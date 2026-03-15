[← Back to ML-Specific Math](../README.md) | [Next: Activation Functions →](../02-Activation-Functions/notes.md)

---

# Loss Functions for Machine Learning

> _"To define is to choose a loss function. Every other choice follows."_
> — paraphrasing Vladimir Vapnik, *The Nature of Statistical Learning Theory* (1995)

## Overview

Loss functions are the mathematical heart of supervised learning. They translate the vague instruction "make accurate predictions" into a precise scalar objective that gradient descent can minimise. Every weight update in every neural network, from a two-layer MLP to a 70-billion-parameter LLM, flows from a single gradient: $\nabla_{\boldsymbol{\theta}} \mathcal{L}$. The choice of $\mathcal{L}$ determines what the model is actually learning — not just how fast it converges.

The mathematics of loss functions connects four major areas: **probability theory** (losses as negative log-likelihoods of generative models), **information theory** (cross-entropy as a KL divergence from true to model distribution), **convex analysis** (which losses have global optima and smooth gradient landscapes), and **decision theory** (Bayes risk, proper scoring rules, regret). Understanding these connections transforms loss function selection from a heuristic choice into a principled design decision.

This section covers the full spectrum from classical regression losses (MSE, MAE, Huber) through classification objectives (cross-entropy, focal, hinge) to modern generative and alignment losses (ELBO, InfoNCE, DPO). We analyse each from four angles: its probabilistic interpretation, its gradient geometry, its calibration properties, and its role in 2026-era AI systems.

## Prerequisites

- **Probability and statistics** (Chapter 06–07): probability densities, MLE, KL divergence, Gaussian distribution
- **Information theory** (Chapter 09): entropy, cross-entropy, mutual information
- **Optimisation** (Chapter 08): gradient descent, convexity, Lagrangian duality
- **Functional analysis** (Chapter 12): norms, Lipschitz continuity, function spaces
- **Linear models** (§14-01): logistic regression, softmax, ERM framework

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive derivations: loss landscapes, gradient geometry, calibration curves, InfoNCE temperature, DPO margin analysis |
| [exercises.ipynb](exercises.ipynb) | 10 graded problems covering Bayes risk, Huber derivation, focal loss, ELBO, InfoNCE, label smoothing, DPO, and multi-task balancing |

## Learning Objectives

After completing this section, you will:

- Derive MSE, MAE, and cross-entropy as negative log-likelihoods of Gaussian, Laplace, and Bernoulli/Categorical distributions
- State and apply the Bayes risk theorem: the optimal predictor for MSE is the conditional mean; for MAE it is the conditional median
- Prove that cross-entropy is a proper scoring rule and explain why this implies calibration
- Analyse the gradient geometry of hinge, focal, and label-smoothed losses and connect each to a training pathology it solves
- Derive the InfoNCE loss as a lower bound on mutual information and explain the role of temperature $\tau$
- Derive the DPO loss from the Bradley-Terry preference model without a separate reward model
- Apply the log-sum-exp trick and numerically stable softmax to prevent NaN losses in practice
- Explain uncertainty weighting for multi-task losses (Kendall et al. 2018) and its Bayesian interpretation
- Classify any loss as convex/non-convex, proper/improper, Fisher-consistent/inconsistent, and Lipschitz/non-Lipschitz
- Connect the autoregressive LM cross-entropy loss to perplexity and bits-per-byte

---

## Table of Contents

- [1. Intuition and Motivation](#1-intuition-and-motivation)
  - [1.1 What a Loss Function Actually Does](#11-what-a-loss-function-actually-does)
  - [1.2 The Risk Minimisation Framework](#12-the-risk-minimisation-framework)
  - [1.3 Loss Functions in Modern AI](#13-loss-functions-in-modern-ai)
  - [1.4 Historical Timeline 1805–2024](#14-historical-timeline-18052024)
- [2. Formal Framework](#2-formal-framework)
  - [2.1 Rigorous Definition](#21-rigorous-definition)
  - [2.2 Empirical Risk Minimisation](#22-empirical-risk-minimisation)
  - [2.3 Bayes Risk and Optimal Predictors](#23-bayes-risk-and-optimal-predictors)
  - [2.4 Surrogate Losses and Consistency](#24-surrogate-losses-and-consistency)
- [3. Regression Losses](#3-regression-losses)
  - [3.1 Mean Squared Error](#31-mean-squared-error)
  - [3.2 Mean Absolute Error](#32-mean-absolute-error)
  - [3.3 Huber Loss](#33-huber-loss)
  - [3.4 Quantile Loss](#34-quantile-loss)
  - [3.5 Log-Cosh and Pseudo-Huber](#35-log-cosh-and-pseudo-huber)
  - [3.6 Heteroscedastic Gaussian NLL](#36-heteroscedastic-gaussian-nll)
- [4. Classification Losses](#4-classification-losses)
  - [4.1 Binary Cross-Entropy](#41-binary-cross-entropy)
  - [4.2 Categorical Cross-Entropy](#42-categorical-cross-entropy)
  - [4.3 Hinge Loss and the SVM Connection](#43-hinge-loss-and-the-svm-connection)
  - [4.4 Focal Loss](#44-focal-loss)
  - [4.5 Label Smoothing](#45-label-smoothing)
  - [4.6 Temperature Scaling and Knowledge Distillation](#46-temperature-scaling-and-knowledge-distillation)
- [5. Probabilistic and Generative Losses](#5-probabilistic-and-generative-losses)
  - [5.1 Negative Log-Likelihood as Unifying Framework](#51-negative-log-likelihood-as-unifying-framework)
  - [5.2 KL Divergence: Forward vs Reverse](#52-kl-divergence-forward-vs-reverse)
  - [5.3 Evidence Lower Bound (ELBO)](#53-evidence-lower-bound-elbo)
  - [5.4 Wasserstein Distance and WGAN](#54-wasserstein-distance-and-wgan)
  - [5.5 Autoregressive LM Loss](#55-autoregressive-lm-loss)
- [6. Ranking and Contrastive Losses](#6-ranking-and-contrastive-losses)
  - [6.1 Pairwise Ranking Loss](#61-pairwise-ranking-loss)
  - [6.2 Triplet Loss](#62-triplet-loss)
  - [6.3 InfoNCE and NT-Xent](#63-infonce-and-nt-xent)
  - [6.4 DPO Loss](#64-dpo-loss)
- [7. Sequence and Structured Losses](#7-sequence-and-structured-losses)
  - [7.1 Sequence Cross-Entropy and Teacher Forcing](#71-sequence-cross-entropy-and-teacher-forcing)
  - [7.2 CTC Loss](#72-ctc-loss)
  - [7.3 REINFORCE and Policy Gradient Loss](#73-reinforce-and-policy-gradient-loss)
  - [7.4 Reward Model Loss in RLHF](#74-reward-model-loss-in-rlhf)
- [8. Loss Function Properties](#8-loss-function-properties)
  - [8.1 Convexity](#81-convexity)
  - [8.2 Calibration and Proper Scoring Rules](#82-calibration-and-proper-scoring-rules)
  - [8.3 Fisher Consistency](#83-fisher-consistency)
  - [8.4 Lipschitz Continuity of Gradients](#84-lipschitz-continuity-of-gradients)
  - [8.5 Robustness to Outliers](#85-robustness-to-outliers)
- [9. Loss Landscape Geometry](#9-loss-landscape-geometry)
  - [9.1 Gradient and Hessian Analysis](#91-gradient-and-hessian-analysis)
  - [9.2 Saddle Points vs Local Minima](#92-saddle-points-vs-local-minima)
  - [9.3 Loss Landscape Visualisation](#93-loss-landscape-visualisation)
  - [9.4 Sharpness-Aware Minimisation](#94-sharpness-aware-minimisation)
- [10. Regularisation as Loss Terms](#10-regularisation-as-loss-terms)
  - [10.1 L2 / Weight Decay](#101-l2--weight-decay)
  - [10.2 L1 Regularisation](#102-l1-regularisation)
  - [10.3 Elastic Net](#103-elastic-net)
  - [10.4 Dropout as Regularisation](#104-dropout-as-regularisation)
- [11. Computational Considerations](#11-computational-considerations)
  - [11.1 Numerical Stability](#111-numerical-stability)
  - [11.2 Multi-Task Loss Balancing](#112-multi-task-loss-balancing)
  - [11.3 Loss Reduction Modes](#113-loss-reduction-modes)
  - [11.4 Mixed Precision and Loss Scaling](#114-mixed-precision-and-loss-scaling)
- [12. Common Mistakes](#12-common-mistakes)
- [13. Exercises](#13-exercises)
- [14. Why This Matters for AI (2026 Perspective)](#14-why-this-matters-for-ai-2026-perspective)
- [15. Conceptual Bridge](#15-conceptual-bridge)

---

## 1. Intuition and Motivation

### 1.1 What a Loss Function Actually Does

A loss function converts the abstract goal "make good predictions" into a concrete number that optimisation algorithms can minimise. Without this conversion, gradient descent has nothing to differentiate. With the wrong conversion, it optimises the wrong thing — a phenomenon so common it has a name: Goodhart's Law ("when a measure becomes a target, it ceases to be a good measure").

Concretely, a loss function $\ell: \hat{\mathcal{Y}} \times \mathcal{Y} \to \mathbb{R}_{\geq 0}$ measures the **cost of predicting $\hat{y}$ when the true answer is $y$**. It must satisfy:
- **Non-negativity**: $\ell(\hat{y}, y) \geq 0$ for all $\hat{y}, y$
- **Zero at perfection**: $\ell(y, y) = 0$

These two properties do not uniquely determine a loss — there are infinitely many valid choices, and the choice matters enormously. To see why, consider three models for the same regression problem:

- **MSE-trained model**: predicts the conditional mean $\mathbb{E}[y \mid \mathbf{x}]$. If the target distribution is bimodal (e.g., a coin flip that gives 0 or 10), the model predicts 5 — a value that never actually occurs.
- **MAE-trained model**: predicts the conditional median. For the same bimodal distribution, it predicts 0 or 10 (depending on the skew), which at least occurs in the data.
- **NLL-trained model**: directly minimises the negative log-likelihood of the data under a parametric family. If the family is correctly specified, this yields the most statistically efficient estimator.

**For AI**: In LLMs, the loss is autoregressive cross-entropy over tokens. This forces the model to assign high probability to the exact next token, not just a semantically similar one. This is why perplexity (the exponentiated average cross-entropy) is the standard LLM evaluation metric — it is directly tied to the training objective.

Loss functions also determine **what information the model extracts from data**. MSE ignores the phase of prediction errors — over- and under-predictions are penalised equally. Quantile loss breaks this symmetry: predicting the 90th percentile penalises under-prediction more than over-prediction. This asymmetry is precisely what makes quantile regression useful for risk management, where false negatives (underestimating risk) are more costly than false positives.

### 1.2 The Risk Minimisation Framework

The statistical learning theory framework formalises the connection between loss functions and generalisation. Let $\mathcal{X}$ be the input space, $\mathcal{Y}$ the output space, and $P$ an unknown joint distribution over $\mathcal{X} \times \mathcal{Y}$.

**Definition (Expected Risk)**. The expected risk of a predictor $f: \mathcal{X} \to \hat{\mathcal{Y}}$ under loss $\ell$ is:
$$R(f) = \mathbb{E}_{(\mathbf{x}, y) \sim P}[\ell(f(\mathbf{x}), y)]$$

The **Bayes optimal predictor** $f^* = \arg\min_f R(f)$ achieves the **Bayes risk** $R^* = R(f^*)$, the irreducible error due to label noise and inherent uncertainty. No learner, regardless of data or compute, can achieve expected risk below $R^*$.

Since $P$ is unknown, we cannot compute $R(f)$ directly. Instead, given a dataset $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$, we minimise the **empirical risk**:
$$\hat{R}_n(f) = \frac{1}{n} \sum_{i=1}^n \ell(f(\mathbf{x}^{(i)}), y^{(i)})$$

The gap between the true risk and Bayes risk decomposes as:
$$R(f) - R^* = \underbrace{[R(f) - \inf_{f \in \mathcal{H}} R(f)]}_{\text{estimation error}} + \underbrace{[\inf_{f \in \mathcal{H}} R(f) - R^*]}_{\text{approximation error}}$$

**Estimation error** vanishes as $n \to \infty$ for sufficiently rich hypothesis classes (uniform law of large numbers). **Approximation error** vanishes as $\mathcal{H}$ grows (e.g., deeper networks), but this increases estimation error. This tension is the **bias-variance tradeoff** expressed in terms of risk rather than squared error.

**For AI**: Modern LLMs operate at a regime where the approximation error is nearly zero (transformers are universal approximators for sequence distributions) but estimation error remains significant — especially for low-frequency knowledge. This is why scaling laws (Hoffmann et al. 2022, Chinchilla) focus on the interplay between model size and dataset size: both affect estimation error.

### 1.3 Loss Functions in Modern AI

Every major AI system encodes its task specification as a loss function. Understanding these connections reveals *why* each system behaves as it does:

| System | Loss Function | What It Optimises |
|---|---|---|
| GPT-4, Llama, Mistral | Autoregressive cross-entropy | Next-token probability |
| CLIP (Radford et al. 2021) | InfoNCE / NT-Xent | Image-text alignment via MI maximisation |
| Stable Diffusion | Denoising ELBO | Evidence lower bound on image log-likelihood |
| AlphaFold 2 | Frame-aligned point error + LDDT | Protein structure deviation |
| RLHF reward model | Bradley-Terry cross-entropy | Human preference ranking |
| DPO (Rafailov et al. 2023) | Log-ratio preference loss | Policy alignment without a separate reward model |
| SAM (Foret et al. 2021) | Perturbed empirical risk | Sharpness of the loss landscape |
| LoRA fine-tuning | Task CE + optional KL | Instruction following while preserving pretrained distribution |

**The choice of loss is a statement about what matters**. CLIP's InfoNCE treats every non-matching pair as a negative — implicitly asserting that images and texts that aren't explicitly paired should be pushed apart. This works because the 400M training pairs are sufficient to define a meaningful similarity structure. For a medical imaging task with 10K paired images, this assumption would be catastrophically wrong.

### 1.4 Historical Timeline 1805–2024

```
LOSS FUNCTION DEVELOPMENT TIMELINE
════════════════════════════════════════════════════════════════════════

  1805  Legendre — Method of least squares (minimising sum of squared
        residuals for planetary orbit fitting)

  1809  Gauss — Probabilistic justification: LSQ = MLE under Gaussian
        noise (published in Theoria Motus Corporum Coelestium)

  1936  Fisher — Maximum likelihood as general principle for estimation

  1951  Kullback & Leibler — KL divergence; cross-entropy as KL + entropy

  1958  Rosenblatt — Perceptron with binary loss (misclassification count)

  1963  Vapnik & Chervonenkis — Hinge loss + max-margin for SVMs

  1974  Werbos — Backpropagation through cross-entropy loss

  1986  Rumelhart, Hinton, Williams — Backprop + sigmoid CE published widely

  1991  Huber — Robust estimation with the eponymous Huber loss (1964 paper)

  2003  Joachims — RankNet pairwise ranking loss for information retrieval

  2006  Hinton & Salakhutdinov — Contrastive divergence for RBMs

  2010  Collobert & Weston — NLP with hinge loss on word embeddings

  2014  Goodfellow et al. — GAN adversarial loss (minimax / Jensen-Shannon)

  2015  Hinton et al. — Knowledge distillation with soft cross-entropy

  2015  Lin et al. — Focal loss for dense object detection (RetinaNet)

  2017  Vaswani et al. — Transformer with label-smoothed cross-entropy

  2018  Oord et al. — InfoNCE as MI lower bound (CPC)

  2020  Chen et al. — NT-Xent in SimCLR for self-supervised learning

  2021  Radford et al. — CLIP with symmetric InfoNCE over image-text pairs

  2022  Kingma et al. — ELBO for VAEs; extended to diffusion models

  2022  Foret et al. — SAM: sharpness-aware minimisation

  2023  Rafailov et al. — DPO: direct preference optimisation

  2024  Grover et al. — Reward-free RLHF with implicit preference models

════════════════════════════════════════════════════════════════════════
```

---

## 2. Formal Framework

### 2.1 Rigorous Definition

**Definition (Loss Function)**. A loss function is a measurable map $\ell: \hat{\mathcal{Y}} \times \mathcal{Y} \to [0, \infty)$ satisfying:
1. $\ell(\hat{y}, y) \geq 0$ for all $\hat{y} \in \hat{\mathcal{Y}}, y \in \mathcal{Y}$
2. $\ell(y, y) = 0$ for all $y \in \mathcal{Y}$ (where $y$ is embedded in $\hat{\mathcal{Y}}$)

Note that $\hat{\mathcal{Y}}$ (the prediction space) may differ from $\mathcal{Y}$ (the label space). For classification with $K$ classes, $\mathcal{Y} = [K]$ but $\hat{\mathcal{Y}} = \Delta^{K-1}$ (the probability simplex).

**Examples** (satisfying definition):
- $\ell(\hat{y}, y) = (\hat{y} - y)^2$ — MSE loss on $\mathbb{R}$
- $\ell(\hat{p}, y) = -\log \hat{p}_y$ — cross-entropy on $\Delta^{K-1} \times [K]$
- $\ell(\hat{y}, y) = \max(0, 1 - y\hat{y})$ — hinge on $\mathbb{R} \times \{-1,+1\}$

**Non-examples** (violating definition):
- $\ell(\hat{y}, y) = \hat{y} - y$ — not non-negative (can be negative when $\hat{y} < y$)
- $\ell(\hat{y}, y) = (\hat{y} - y)^2 + 1$ — not zero at $\hat{y} = y$
- $\ell(\hat{y}, y) = -\log(1 - |\hat{y} - y|)$ — unbounded below and undefined for large errors

**The aggregate training loss** over dataset $\mathcal{D}$ with reduction "mean":
$$\mathcal{L}(\boldsymbol{\theta}; \mathcal{D}) = \frac{1}{n} \sum_{i=1}^n \ell(f_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}), y^{(i)})$$

This is also the empirical risk $\hat{R}_n(f_{\boldsymbol{\theta}})$. The gradient $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ is what backpropagation computes.

### 2.2 Empirical Risk Minimisation

**Theorem (ERM Consistency, Vapnik 1991)**. If the hypothesis class $\mathcal{H}$ has finite VC dimension $d$, then for any $\delta > 0$, with probability at least $1 - \delta$:
$$R(\hat{f}_n) - \inf_{f \in \mathcal{H}} R(f) \leq 2\sqrt{\frac{d \log(2n/d) + \log(2/\delta)}{n}}$$

where $\hat{f}_n = \arg\min_{f \in \mathcal{H}} \hat{R}_n(f)$ is the ERM solution.

**Interpretation**: With $n$ training samples and VC dimension $d$, the estimation error (generalisation gap) shrinks as $O(\sqrt{d/n})$. For neural networks with $P$ parameters, the effective VC dimension is $O(P \log P)$ — but modern over-parameterised networks generalise far better than this bound predicts, indicating that implicit regularisation (from SGD, architecture, initialisation) plays a crucial role.

**For AI**: ERM is the formal justification for training on a finite dataset and expecting generalisation. The **double descent phenomenon** (Belkin et al. 2019) shows that for neural networks, test risk can *decrease* as model size grows past the interpolation threshold — contradicting classical ERM theory. This happens because over-parameterised networks find interpolating solutions with additional desirable inductive biases.

### 2.3 Bayes Risk and Optimal Predictors

**Theorem (Bayes Optimal Predictors)**. For a loss function $\ell$, the pointwise optimal prediction given $\mathbf{x}$ is:
$$f^*(\mathbf{x}) = \arg\min_{\hat{y} \in \hat{\mathcal{Y}}} \mathbb{E}_{y \mid \mathbf{x}}[\ell(\hat{y}, y)]$$

The specific form of $f^*$ depends on $\ell$:

| Loss | Bayes Optimal Predictor $f^*(\mathbf{x})$ |
|---|---|
| $\ell = (\hat{y} - y)^2$ | $\mathbb{E}[y \mid \mathbf{x}]$ — conditional mean |
| $\ell = \lvert\hat{y} - y\rvert$ | $\operatorname{median}(y \mid \mathbf{x})$ — conditional median |
| $\ell_\tau$ (quantile) | $\tau$-th quantile of $P(y \mid \mathbf{x})$ |
| $\ell = -\log \hat{p}_y$ | $P(y \mid \mathbf{x})$ — true conditional distribution |
| $\ell = \mathbf{1}[\hat{y} \neq y]$ | $\arg\max_k P(y = k \mid \mathbf{x})$ — mode |

**Proof sketch for MSE**: $\mathbb{E}[(y - \hat{y})^2] = \mathbb{E}[(y - \mathbb{E}[y|\mathbf{x}])^2] + (\mathbb{E}[y|\mathbf{x}] - \hat{y})^2$. The first term is irreducible (Bayes risk); the second is zero iff $\hat{y} = \mathbb{E}[y|\mathbf{x}]$.

**Proof sketch for MAE**: The derivative of $\mathbb{E}[|y - \hat{y}|]$ with respect to $\hat{y}$ is $\mathbb{E}[\operatorname{sign}(\hat{y} - y)] = P(y \leq \hat{y}) - P(y > \hat{y})$. Setting to zero gives $P(y \leq \hat{y}) = 1/2$, i.e., $\hat{y}$ is the median.

**For AI**: LLMs trained on cross-entropy learn the conditional token distribution $P(\text{token} \mid \text{context})$ — this is exactly the Bayes optimal predictor for NLL. Beam search then approximates $\arg\max P(\text{sequence} \mid \text{prompt})$, while sampling with temperature approximates drawing from this distribution.

### 2.4 Surrogate Losses and Consistency

The **0-1 loss** $\ell_{0\text{-}1}(\hat{y}, y) = \mathbf{1}[\hat{y} \neq y]$ is the natural classification objective, but it is non-convex and discontinuous — gradient descent cannot minimise it directly. **Surrogate losses** are convex upper bounds used in place of the 0-1 loss.

**Definition (Fisher Consistency / Classification-Calibrated)**. A surrogate loss $\ell$ is **Fisher consistent** for binary classification (with labels $\pm 1$) if:
$$\underset{f}{\arg\min}\; \mathbb{E}_{y \mid \mathbf{x}}[\ell(yf(\mathbf{x}))] = \operatorname{sign}(\eta(\mathbf{x}) - 1/2)$$

where $\eta(\mathbf{x}) = P(y = 1 \mid \mathbf{x})$ is the class probability.

**Theorem (Bartlett, Jordan, McAuliffe 2006)**. A convex loss $\ell: \mathbb{R} \to \mathbb{R}_+$ that is differentiable at 0 with $\ell'(0) < 0$ is Fisher consistent for binary classification. This covers:
- Cross-entropy: $\ell(z) = \log(1 + e^{-z})$ — consistent
- Hinge: $\ell(z) = \max(0, 1-z)$ — consistent
- Exponential: $\ell(z) = e^{-z}$ — consistent
- Squared loss: $\ell(z) = (1-z)^2$ — consistent (but less robust)

**Non-example**: The squared loss $\ell(\hat{p}, y) = (\hat{p} - y)^2$ in the probability domain (not the margin domain) is not Fisher consistent when $\eta(\mathbf{x}) \neq 0.5$, because penalising $(p - 1)^2$ and $(p - 0)^2$ equally does not reflect the asymmetry in class probabilities near the boundary.

**For AI**: All major classification losses used in neural networks (cross-entropy, focal, hinge) are Fisher consistent. This guarantees that minimising the surrogate loss eventually yields the Bayes classifier — a crucial theoretical property that justifies the entire supervised learning pipeline.


---

## 3. Regression Losses

### 3.1 Mean Squared Error

**Definition**. The Mean Squared Error (MSE) loss for a single prediction:
$$\ell_{\text{MSE}}(\hat{y}, y) = (\hat{y} - y)^2$$

and over a dataset:
$$\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)})^2$$

**Probabilistic Derivation**. Assume the data-generating process is:
$$y = f_{\boldsymbol{\theta}}(\mathbf{x}) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

The log-likelihood of the dataset is:
$$\log p(\mathcal{D} \mid \boldsymbol{\theta}) = \sum_{i=1}^n \log \mathcal{N}(y^{(i)}; f_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}), \sigma^2)
= -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y^{(i)} - f_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}))^2$$

Maximising this log-likelihood is equivalent to minimising the MSE (the $\sigma^2$ term is a constant w.r.t. $\boldsymbol{\theta}$). **MSE = MLE under Gaussian noise.** This is not a coincidence — it is the assumption that makes MSE the right choice.

**Gradient and Geometry**:
$$\nabla_{\hat{y}} \ell_{\text{MSE}} = 2(\hat{y} - y)$$

The gradient is linear in the residual $r = \hat{y} - y$. Large residuals produce large gradients — this is why MSE is sensitive to outliers. A single training point with $|r| = 10$ contributes a gradient 100$\times$ larger than a point with $|r| = 1$.

**Properties**:
- Convex and strongly convex on $\mathbb{R}$ (with $\lambda = 2$)
- $C^\infty$ smooth: infinitely differentiable
- Optimal prediction: $f^*(\mathbf{x}) = \mathbb{E}[y \mid \mathbf{x}]$
- **Breakdown point**: 0% (a single outlier can move the optimal predictor arbitrarily)
- **Sensitivity**: quadratically penalises errors — errors $> 1$ grow faster than linearly

**For AI**: MSE is rarely used as a primary classification loss in modern deep learning, but it appears as:
- The loss for value-function regression in reinforcement learning (DQN: Mnih et al. 2015)
- The denoising objective in diffusion models (simplified DDPM loss): $\mathbb{E}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)\|^2]$
- Regression heads in multi-task models (bounding box coordinates in DETR)
- The student-teacher distillation loss on intermediate representations (e.g., PKD: Patient Knowledge Distillation)

### 3.2 Mean Absolute Error

**Definition**:
$$\ell_{\text{MAE}}(\hat{y}, y) = \lvert\hat{y} - y\rvert$$

**Probabilistic Derivation**. MAE is MLE under **Laplace noise**:
$$y = f_{\boldsymbol{\theta}}(\mathbf{x}) + \varepsilon, \quad \varepsilon \sim \operatorname{Laplace}(0, b)$$

$$p(y \mid \mathbf{x}; \boldsymbol{\theta}) = \frac{1}{2b}\exp\!\left(-\frac{|y - f_{\boldsymbol{\theta}}(\mathbf{x})|}{b}\right)$$

Maximising the log-likelihood gives $\min \sum |y^{(i)} - f_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})|$.

**Subgradient**:
$$\partial \ell_{\text{MAE}} / \partial \hat{y} = \operatorname{sign}(\hat{y} - y) = \begin{cases} +1 & \hat{y} > y \\ -1 & \hat{y} < y \\ [-1, 1] & \hat{y} = y \end{cases}$$

The subgradient is constant in magnitude — a 10$\times$ larger residual does not produce a 10$\times$ larger gradient. This is what makes MAE robust to outliers.

**Properties**:
- Convex but **not differentiable** at $\hat{y} = y$ (subgradient exists)
- Optimal prediction: $f^*(\mathbf{x}) = \operatorname{median}(y \mid \mathbf{x})$
- **Breakdown point**: 50% — up to half the data can be outliers without moving the optimal predictor to $\pm\infty$
- **Slower convergence** than MSE near the optimum (constant gradient vs. vanishing gradient)

**Non-example**: MAE is **not** the right loss when outliers encode meaningful signal. In fraud detection, extreme transaction amounts are informative, not noise — using MAE would suppress their gradient contribution.

**For AI**: MAE (as Smooth L1 / Huber) is the default for bounding-box regression in object detection (Faster R-CNN, YOLO, DETR). The constant gradient near outliers prevents large bounding-box errors from destabilising the feature pyramid training.

### 3.3 Huber Loss

The Huber loss (Huber 1964) interpolates between MSE (near the optimum) and MAE (far from the optimum):

$$\ell_\delta(\hat{y}, y) = \begin{cases}
\frac{1}{2}r^2 & \lvert r\rvert \leq \delta \\
\delta\lvert r\rvert - \frac{1}{2}\delta^2 & \lvert r\rvert > \delta
\end{cases}$$

where $r = \hat{y} - y$ and $\delta > 0$ is the transition threshold.

**Gradient**:
$$\frac{\partial \ell_\delta}{\partial \hat{y}} = \begin{cases}
r & \lvert r\rvert \leq \delta \\
\delta \cdot \operatorname{sign}(r) & \lvert r\rvert > \delta
\end{cases}$$

The gradient is clipped to $[-\delta, \delta]$ for large residuals — Huber loss performs **implicit gradient clipping**.

**Key properties**:
- Differentiable everywhere (at $|r| = \delta$: gradient is $\delta \cdot \operatorname{sign}(r)$ from both sides, continuous)
- **$\delta \to 0$**: approaches MAE (gradient always $\pm\delta \to 0$, but normalised: slope = 1)
- **$\delta \to \infty$**: approaches $\frac{1}{2}$MSE (the quadratic region covers all residuals)
- Smooth at the transition point: $\frac{1}{2}r^2|_{r=\delta} = \frac{1}{2}\delta^2$ matches $\delta|r| - \frac{1}{2}\delta^2|_{r=\delta} = \frac{1}{2}\delta^2$

**Choosing $\delta$**: Set $\delta$ to approximately the expected noise scale. If residuals are Gaussian with $\sigma$, set $\delta \approx \sigma$ to get MSE-like behaviour in the normal regime and MAE-like behaviour for outliers beyond $\sigma$.

**Connection to gradient clipping**: If you use MSE with gradient clipping at threshold $c$, this is approximately equivalent to Huber loss with $\delta = c$. Many practitioners implement one when they intend the other.

**For AI**: The **Smooth L1 loss** (PyTorch's `nn.SmoothL1Loss`) is Huber loss with $\delta = 1$ by default (or $\delta = 1/\text{beta}$ in older versions). It is the standard regression loss for bounding boxes and depth estimation in modern vision models.

### 3.4 Quantile Loss

For a target quantile $\tau \in (0,1)$, the **quantile loss** (also called **pinball loss**) is:

$$\ell_\tau(\hat{y}, y) = \begin{cases}
\tau(y - \hat{y}) & y \geq \hat{y} \\
(1-\tau)(\hat{y} - y) & y < \hat{y}
\end{cases} = \max(\tau(y - \hat{y}),\, (1-\tau)(\hat{y} - y))$$

Equivalently: $\ell_\tau = (y - \hat{y})(\tau - \mathbf{1}_{y < \hat{y}})$.

**Optimality**: The minimiser of $\mathbb{E}[\ell_\tau(\hat{y}, y)]$ over constants $\hat{y}$ is the $\tau$-th quantile of $P(y)$. For $\tau = 0.5$, this is the median — recovering MAE.

**Gradient**:
$$\frac{\partial \ell_\tau}{\partial \hat{y}} = \begin{cases} -\tau & y \geq \hat{y} \\ 1-\tau & y < \hat{y} \end{cases}$$

For $\tau = 0.9$: gradient is $-0.9$ when under-predicting, $+0.1$ when over-predicting. This 9:1 ratio means the model is penalised 9$\times$ more for under-predictions — pushing predictions toward the 90th percentile.

**Applications**:
- **Conformal prediction**: fit a model with $\tau = 1 - \alpha$ to get prediction intervals with coverage $1-\alpha$
- **Weather forecasting**: predict the 10th and 90th percentile of rainfall to give uncertainty intervals
- **RL value functions**: quantile regression DQN (QR-DQN, Dabney et al. 2017) models the full return distribution instead of the mean

**For AI**: **Implicit Quantile Networks** (IQN, Dabney et al. 2018) use quantile loss to represent the full distribution over returns in reinforcement learning. **Conformal prediction** (Angelopoulos & Bates 2023) uses quantile loss to give distribution-free guarantees on prediction intervals for any model, making it important for deployment safety in LLM applications.

### 3.5 Log-Cosh and Pseudo-Huber

**Log-Cosh Loss**:
$$\ell_{\text{log-cosh}}(\hat{y}, y) = \log(\cosh(\hat{y} - y))$$

Using the Taylor expansion $\cosh(r) = 1 + r^2/2 + r^4/24 + \ldots$:
- For small $r$: $\log(1 + r^2/2 + \ldots) \approx r^2/2$ — behaves like $\frac{1}{2}$MSE
- For large $|r|$: $\cosh(r) \approx e^{|r|}/2$, so $\log\cosh(r) \approx |r| - \log 2$ — behaves like MAE

**Gradient**: $\partial\ell/\partial\hat{y} = \tanh(\hat{y} - y)$, which is bounded in $(-1, 1)$ — implicitly clips gradients.

**Pseudo-Huber Loss** (differentiable approximation):
$$\ell_{\text{ph},\delta}(\hat{y}, y) = \delta^2\left(\sqrt{1 + (r/\delta)^2} - 1\right)$$

Gradient: $r / \sqrt{1 + (r/\delta)^2}$, bounded by $\delta$ in magnitude.

Both log-cosh and pseudo-Huber are infinitely differentiable, making them preferable in settings where higher-order gradient information is needed (e.g., second-order optimisers, natural gradient methods).

### 3.6 Heteroscedastic Gaussian NLL

In standard regression, we assume fixed noise variance $\sigma^2$. **Heteroscedastic regression** lets the model predict both mean $\mu(\mathbf{x})$ and variance $\sigma^2(\mathbf{x})$:

$$p(y \mid \mathbf{x}; \boldsymbol{\theta}) = \mathcal{N}(y;\; \mu_{\boldsymbol{\theta}}(\mathbf{x}),\; \sigma^2_{\boldsymbol{\theta}}(\mathbf{x}))$$

The negative log-likelihood loss is:
$$\ell(\hat{\mu}, \hat{\sigma}^2, y) = \frac{1}{2}\log(2\pi\hat{\sigma}^2) + \frac{(y - \hat{\mu})^2}{2\hat{\sigma}^2}$$

In practice, we predict $\hat{v} = \log\hat{\sigma}^2$ (log-variance) for numerical stability, giving:
$$\ell = \frac{1}{2}e^{-\hat{v}}(y - \hat{\mu})^2 + \frac{1}{2}\hat{v} + \text{const}$$

**Interpretation**: The model can reduce loss by increasing $\hat{v}$ (predicting higher uncertainty) when the residual is large. But increasing $\hat{v}$ also increases the regularisation term $\frac{1}{2}\hat{v}$. The optimal $\hat{v}^* = \log(y - \hat{\mu})^2$, i.e., the log-squared residual.

**For AI**: Heteroscedastic NLL is used in:
- **Laplace Redux** (Immer et al. 2021): efficient approximate Bayesian deep learning
- **Uncertainty estimation in autonomous driving**: the model predicts confidence in its own bounding box predictions (Kendall & Gal 2017)
- **Value function uncertainty in RL**: predicting both Q-value mean and variance for risk-sensitive policies

---

## 4. Classification Losses

### 4.1 Binary Cross-Entropy

For binary classification with label $y \in \{0, 1\}$ and predicted probability $\hat{p} = \sigma(z) \in (0,1)$ (where $\sigma$ is the sigmoid function):

$$\ell_{\text{BCE}}(\hat{p}, y) = -y\log\hat{p} - (1-y)\log(1-\hat{p})$$

**Probabilistic Derivation**: Assume $y \sim \operatorname{Bern}(\hat{p})$. Then:
$$-\log p(y \mid \hat{p}) = -y\log\hat{p} - (1-y)\log(1-\hat{p}) = \ell_{\text{BCE}}$$

BCE is the NLL of the Bernoulli distribution. **BCE = MLE under Bernoulli assumption.**

**Gradient in logit space** (numerically preferred): Let $z = \operatorname{logit}(\hat{p}) = \log(\hat{p}/(1-\hat{p}))$. Then:
$$\frac{\partial \ell_{\text{BCE}}}{\partial z} = \hat{p} - y = \sigma(z) - y$$

This is an elegant result: the gradient is simply the prediction error. When $\hat{p} = y$, the gradient is zero; when $\hat{p} = 1, y = 0$ (confident and wrong), the gradient is $+1$.

**Numerically stable implementation**: Never compute `log(sigmoid(z))` directly. Use:
$$\ell_{\text{BCE}}(z, y) = \max(z, 0) - zy + \log(1 + e^{-|z|})$$

This avoids overflow when $z \gg 0$ and underflow when $z \ll 0$.

**For AI**: BCE is the loss for:
- Binary classification heads in BERT (e.g., Next Sentence Prediction)
- Reward model training: the probability that response A is preferred over B
- Sigmoid cross-entropy in multi-label classification (each class independently)
- Discriminator loss in GANs (original formulation)

### 4.2 Categorical Cross-Entropy

For $K$-class classification with one-hot label $\mathbf{y} \in \{0,1\}^K$ and predicted probabilities $\hat{\mathbf{p}} = \operatorname{softmax}(\mathbf{z}) \in \Delta^{K-1}$:

$$\ell_{\text{CE}}(\hat{\mathbf{p}}, \mathbf{y}) = -\sum_{k=1}^K y_k \log \hat{p}_k = -\log \hat{p}_{y^*}$$

where $y^* = \arg\max_k y_k$ is the true class index.

**Information-Theoretic View**. Cross-entropy decomposes as:
$$H(\mathbf{y}, \hat{\mathbf{p}}) = H(\mathbf{y}) + D_{\mathrm{KL}}(\mathbf{y} \| \hat{\mathbf{p}})$$

Since $H(\mathbf{y}) = 0$ for one-hot labels, minimising cross-entropy is equivalent to minimising $D_{\mathrm{KL}}(\mathbf{y} \| \hat{\mathbf{p}})$ — i.e., making the model distribution match the empirical label distribution.

**Gradient in logit space**: For logits $\mathbf{z}$ with $\hat{\mathbf{p}} = \operatorname{softmax}(\mathbf{z})$:
$$\frac{\partial \ell_{\text{CE}}}{\partial z_k} = \hat{p}_k - y_k$$

The gradient is the residual between predictions and one-hot targets. This is one of the most important equations in deep learning — it means that the gradient of cross-entropy through softmax is perfectly clean.

**Numerical stability**: Always compute `log_softmax` before the cross-entropy rather than `softmax` then `log`. In PyTorch: `F.cross_entropy(logits, targets)` handles this automatically via the log-sum-exp trick:
$$\log \hat{p}_k = z_k - \log\sum_j e^{z_j} = z_k - \left(m + \log\sum_j e^{z_j - m}\right)$$

where $m = \max_j z_j$.

**For AI**: Categorical cross-entropy is **the** loss for LLM training. Over a sequence of $T$ tokens:
$$\mathcal{L}_{\text{LM}} = -\frac{1}{T}\sum_{t=1}^T \log p_{\boldsymbol{\theta}}(x_t \mid x_1, \ldots, x_{t-1})$$

For a vocabulary of $V \approx 50{,}000$ tokens, this is a 50K-way softmax at every position. The gradient $\hat{\mathbf{p}} - \mathbf{y}$ tells the model to increase probability on the correct token and decrease it on all others.

### 4.3 Hinge Loss and the SVM Connection

For binary classification with margin labels $y \in \{-1, +1\}$ and raw score $\hat{y} \in \mathbb{R}$:

$$\ell_{\text{hinge}}(\hat{y}, y) = \max(0, 1 - y\hat{y}) = [1 - y\hat{y}]_+$$

**Geometric interpretation**: The hinge loss is zero when the prediction $\hat{y}$ has the correct sign **and** magnitude $\geq 1$ (i.e., correct with margin). It is positive when the margin is violated.

**Multi-class hinge loss** (Crammer & Singer):
$$\ell_{\text{hinge}}(\hat{\mathbf{y}}, y^*) = \max_{k \neq y^*}(0, \hat{y}_k - \hat{y}_{y^*} + 1)$$

This penalises any class score that comes within 1 unit of the correct class score.

**Properties**:
- Convex, piecewise linear (not smooth at $y\hat{y} = 1$)
- **Sparse gradient**: zero gradient for correctly classified examples with margin — only support vectors get non-zero gradients
- Equivalent to L1 norm of slack variables in the SVM primal problem

**SVM Connection**: The SVM primal problem with soft margin:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i \quad \text{s.t.} \quad y^{(i)}(\mathbf{w}^\top\mathbf{x}^{(i)} + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

is equivalent to:
$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i [1 - y^{(i)}(\mathbf{w}^\top\mathbf{x}^{(i)} + b)]_+$$

which is L2 regularisation + hinge loss with $\lambda = 1/(2C)$.

**For AI**: Hinge loss appears in:
- Training contrastive embedding models with margin (e.g., word2vec negative sampling can be interpreted as hinge-like)
- **Pairwise ranking loss** in information retrieval (RankSVM)
- **Wasserstein GANs**: the critic's objective is a form of hinge loss on the Wasserstein-1 distance

### 4.4 Focal Loss

The **focal loss** (Lin et al. 2017, RetinaNet) addresses **class imbalance** in object detection by down-weighting easy examples:

$$\ell_{\text{focal}}(\hat{p}, y) = -\alpha_y (1 - \hat{p}_y)^\gamma \log \hat{p}_y$$

where:
- $\hat{p}_y$ is the predicted probability for the true class
- $\gamma \geq 0$ is the **focusing parameter** (default: 2)
- $\alpha_y$ is a class-specific weight (default: 0.25 for positives in detection)

**Why it works**: In object detection, 99%+ of anchor boxes contain background. Cross-entropy loss assigns these easy negatives (correctly classified with high confidence) large cumulative loss — swamping the gradient signal from the few hard positive examples (actual objects). Focal loss suppresses the easy negative contribution:

| $\hat{p}_y$ | $(1-\hat{p}_y)^\gamma$ ($\gamma=2$) | Effective weight |
|---|---|---|
| 0.95 (easy correct) | $(0.05)^2 = 0.0025$ | 0.25% of original |
| 0.5 (uncertain) | $(0.5)^2 = 0.25$ | 25% of original |
| 0.05 (hard wrong) | $(0.95)^2 = 0.9025$ | 90.25% of original |

**Gradient**: $\partial\ell_{\text{focal}}/\partial z_y = \alpha_y(1-\hat{p}_y)^\gamma[\gamma\hat{p}_y\log\hat{p}_y + \hat{p}_y - 1]$

**For AI**: Focal loss is standard in vision models with class imbalance. More broadly, the principle of **re-weighting by difficulty** appears in:
- **Curriculum learning**: train on easy examples first, then hard ones
- **Hard negative mining**: explicitly sample hard negatives in contrastive learning
- **Self-paced learning**: weight loss by model confidence

### 4.5 Label Smoothing

**Label smoothing** (Szegedy et al. 2016, Müller et al. 2019) replaces one-hot targets with a soft distribution:
$$y_k^{\text{smooth}} = (1-\varepsilon) y_k + \frac{\varepsilon}{K}$$

The smoothed cross-entropy is:
$$\ell_{\text{LS}}(\hat{\mathbf{p}}, y^*) = (1-\varepsilon)\ell_{\text{CE}}(\hat{\mathbf{p}}, y^*) + \varepsilon \cdot \frac{1}{K}\sum_k (-\log\hat{p}_k)$$
$$= \ell_{\text{CE}}(\hat{\mathbf{p}}, y^*) + \varepsilon [H(\hat{\mathbf{p}}) - H(\hat{\mathbf{p}}, \mathbf{u})]$$

where $\mathbf{u}$ is the uniform distribution.

**What it prevents**: Without label smoothing, cross-entropy drives logits toward $z_{y^*} \to \infty$, $z_k \to -\infty$ for $k \neq y^*$. The model becomes **overconfident**. Label smoothing penalises overconfidence by adding an entropy regularisation term.

**Effect on calibration**: Müller et al. (2019) showed that label smoothing improves calibration (reduces Expected Calibration Error) because it prevents logit saturation.

**Effect on representations**: Label smoothing makes the penultimate layer representations of the same class cluster tightly together and different classes push apart — similar to supervised contrastive loss (Khosla et al. 2020).

**For AI**: Label smoothing ($\varepsilon = 0.1$) is standard in:
- Transformer training (Vaswani et al. 2017 original transformer paper)
- Image classification (ImageNet training recipes)
- Machine translation (prevents the model from being overconfident on rare words)

### 4.6 Temperature Scaling and Knowledge Distillation

**Temperature scaling** modifies logits before softmax:
$$\hat{p}_k^{(\tau)} = \frac{e^{z_k/\tau}}{\sum_j e^{z_j/\tau}}$$

- $\tau < 1$: sharpens the distribution (more confident)
- $\tau > 1$: flattens the distribution (more uncertain, more entropy)
- $\tau \to 0$: argmax (one-hot)
- $\tau \to \infty$: uniform distribution

**Knowledge Distillation** (Hinton et al. 2015): Train a small student model to match a large teacher's soft predictions:
$$\ell_{\text{KD}} = (1-\alpha)\ell_{\text{CE}}(\hat{\mathbf{p}}_{\text{student}}, \mathbf{y}) + \alpha\tau^2 D_{\mathrm{KL}}(\hat{\mathbf{p}}_{\text{teacher}}^{(\tau)} \| \hat{\mathbf{p}}_{\text{student}}^{(\tau)})$$

The $\tau^2$ factor compensates for the gradient scaling introduced by temperature.

**Why soft targets help**: The teacher's soft predictions contain **dark knowledge** — information about the relative similarity between classes. A model that assigns 0.01 to "cat" and 0.001 to "dog" when the true class is "car" is telling the student that cars look more like cats than dogs. Hard labels throw away this information.

**For AI**: Knowledge distillation is central to LLM compression. DistilBERT (Sanh et al. 2019) is 40% smaller than BERT with only 3% accuracy loss. TinyLlama (Zhang et al. 2024) uses token-level distillation to match Llama 2 performance at 1.1B parameters.


---

## 5. Probabilistic and Generative Losses

### 5.1 Negative Log-Likelihood as Unifying Framework

**Theorem (NLL Unification)**. Every standard regression and classification loss is a special case of the negative log-likelihood (NLL) under a specific distributional assumption:

| Loss | Distributional Assumption | NLL Form |
|---|---|---|
| MSE | $y \sim \mathcal{N}(f_{\boldsymbol{\theta}}(\mathbf{x}), \sigma^2)$ | $\frac{(y-\hat{y})^2}{2\sigma^2} + \text{const}$ |
| MAE | $y \sim \operatorname{Laplace}(f_{\boldsymbol{\theta}}(\mathbf{x}), b)$ | $\frac{|y-\hat{y}|}{b} + \text{const}$ |
| BCE | $y \sim \operatorname{Bern}(\hat{p})$ | $-y\log\hat{p} - (1-y)\log(1-\hat{p})$ |
| CE | $y \sim \operatorname{Cat}(\hat{\mathbf{p}})$ | $-\log\hat{p}_{y^*}$ |
| Poisson NLL | $y \sim \operatorname{Poisson}(\hat{\lambda})$ | $\hat{\lambda} - y\log\hat{\lambda} + \log y!$ |

This unification clarifies that **choosing a loss is equivalent to choosing a noise model**. When you use MSE, you are asserting that your targets are corrupted by Gaussian noise. If this is wrong — e.g., the residuals are heavy-tailed or discrete — you should use a different loss.

**MLE vs MAP**: Adding an L2 regulariser to the NLL corresponds to placing a Gaussian prior on the parameters (MAP estimation). Adding L1 regularisation corresponds to a Laplace prior.

**For AI**: The factored form of the LM loss:
$$\mathcal{L}_{\text{LM}} = -\frac{1}{T}\sum_{t=1}^T \log p(x_t \mid x_{<t}; \boldsymbol{\theta})$$

is the NLL of the autoregressive decomposition $p(\mathbf{x}) = \prod_t p(x_t \mid x_{<t})$. The distributional assumption is that the data generating process is a discrete-time Markov chain with a learnable transition kernel — a much weaker assumption than any parametric noise model.

### 5.2 KL Divergence: Forward vs Reverse

**Definition**. The KL divergence from distribution $q$ to distribution $p$:
$$D_{\mathrm{KL}}(p \| q) = \mathbb{E}_{x \sim p}\left[\log\frac{p(x)}{q(x)}\right] = \int p(x)\log\frac{p(x)}{q(x)}\,dx$$

**Properties**:
- $D_{\mathrm{KL}}(p \| q) \geq 0$ with equality iff $p = q$ a.e. (Gibbs' inequality)
- **Asymmetric**: $D_{\mathrm{KL}}(p \| q) \neq D_{\mathrm{KL}}(q \| p)$ in general
- Not a metric (triangle inequality fails)
- Decomposition: $D_{\mathrm{KL}}(p \| q) = H(p, q) - H(p)$ — cross-entropy minus entropy

**Forward KL** ($D_{\mathrm{KL}}(p_{\text{data}} \| q_{\boldsymbol{\theta}})$, minimised by MLE):
$$\arg\min_{\boldsymbol{\theta}} D_{\mathrm{KL}}(p_{\text{data}} \| q_{\boldsymbol{\theta}}) = \arg\max_{\boldsymbol{\theta}} \mathbb{E}_{x \sim p_{\text{data}}}[\log q_{\boldsymbol{\theta}}(x)]$$

The forward KL is **mode-covering**: it forces $q_{\boldsymbol{\theta}}$ to assign positive probability wherever $p_{\text{data}} > 0$. If $q_{\boldsymbol{\theta}}$ assigns zero mass to a region with positive $p_{\text{data}}$, the KL is infinite.

**Reverse KL** ($D_{\mathrm{KL}}(q_{\boldsymbol{\theta}} \| p_{\text{data}})$, used in variational inference):
$$\arg\min_{\boldsymbol{\theta}} D_{\mathrm{KL}}(q_{\boldsymbol{\theta}} \| p) = \arg\max_{\boldsymbol{\theta}} \mathbb{E}_{x \sim q_{\boldsymbol{\theta}}}[\log p(x) - \log q_{\boldsymbol{\theta}}(x)]$$

The reverse KL is **mode-seeking**: $q_{\boldsymbol{\theta}}$ concentrates on regions where $p > 0$ and avoids regions where $q_{\boldsymbol{\theta}} > 0$ but $p \approx 0$.

```
FORWARD vs REVERSE KL: MODE-COVERING vs MODE-SEEKING
════════════════════════════════════════════════════════════════════════

  Bimodal target p(x):
                     *   *
                   * * * * *           *   *
                 * * * * * * *       * * * * *
           ──────────────────────────────────────→ x

  Forward KL minimiser q(x):     Reverse KL minimiser q(x):
  (mode-covering)                (mode-seeking)

         *   *   *   *   *              *   *
       * * * * * * * * * * *          * * * * *
       ──────────────────────          ────────

  q covers both modes             q concentrates on ONE mode
  (broad, may have low peaks)     (sharp, but misses other modes)

════════════════════════════════════════════════════════════════════════
```

**For AI**:
- **Forward KL (MLE)**: used in LLM pretraining. Forces the model to cover all modes of the training data distribution.
- **Reverse KL**: used in RLHF KL penalty: $\mathcal{L}_{\text{RLHF}} = -R(\mathbf{y}) + \beta D_{\mathrm{KL}}(\pi_{\boldsymbol{\theta}} \| \pi_{\text{ref}})$. Prevents the fine-tuned policy $\pi_{\boldsymbol{\theta}}$ from drifting too far from the reference policy.
- **Jensen-Shannon divergence** (GAN objective): symmetric version $D_{\text{JS}} = \frac{1}{2}D_{\mathrm{KL}}(p \| m) + \frac{1}{2}D_{\mathrm{KL}}(q \| m)$ where $m = (p+q)/2$.

### 5.3 Evidence Lower Bound (ELBO)

For a **Variational Autoencoder** (VAE, Kingma & Welling 2013), we want to maximise the log-evidence $\log p_{\boldsymbol{\theta}}(\mathbf{x})$ but this is intractable due to the latent variable integral. The ELBO provides a tractable lower bound.

**Derivation via Jensen's inequality**:
$$\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \log \int p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})\,d\mathbf{z}
= \log \int q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\,d\mathbf{z}$$
$$\geq \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\left[\log\frac{p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}\right] = \mathcal{L}_{\text{ELBO}}$$

**Decomposition**:
$$\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x})}[\log p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})]}_{\text{reconstruction term}} - \underbrace{D_{\mathrm{KL}}(q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}))}_{\text{regularisation term}}$$

**Gap**: $\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \mathcal{L}_{\text{ELBO}} + D_{\mathrm{KL}}(q_{\boldsymbol{\phi}} \| p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x}))$. The gap is zero iff $q_{\boldsymbol{\phi}} = p_{\boldsymbol{\theta}}(\mathbf{z} \mid \mathbf{x})$ — i.e., the variational posterior matches the true posterior.

**$\beta$-VAE** (Higgins et al. 2017): weight the KL term by $\beta > 1$:
$$\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}[\log p(\mathbf{x} \mid \mathbf{z})] - \beta D_{\mathrm{KL}}(q \| p)$$

Higher $\beta$ forces the latent code to be more disentangled (each dimension is independent and meaningful) at the cost of reconstruction quality.

**For AI**: The ELBO is the loss for:
- VAEs and their variants (VQ-VAE, hierarchical VAEs like NVAE, VDVAE)
- **Latent diffusion models**: the encoder in Stable Diffusion is trained with an ELBO objective
- **Evidence lower bounds in LLMs**: IWAE (importance-weighted ELBO) for better likelihood estimation

### 5.4 Wasserstein Distance and WGAN

The **Wasserstein-1 distance** (Earth Mover's Distance) between distributions $p_r$ and $p_g$:
$$W_1(p_r, p_g) = \inf_{\gamma \in \Pi(p_r, p_g)} \mathbb{E}_{(x,y) \sim \gamma}[\|x - y\|]$$

where $\Pi$ is the set of all joint distributions with marginals $p_r$ and $p_g$.

**Kantorovich-Rubinstein duality**:
$$W_1(p_r, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_r}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]$$

where the supremum is over 1-Lipschitz functions $f$.

**WGAN Loss** (Arjovsky et al. 2017): replace the GAN discriminator with a 1-Lipschitz critic $f_w$:
$$\mathcal{L}_{\text{WGAN}} = \mathbb{E}_{x \sim p_r}[f_w(x)] - \mathbb{E}_{z \sim p(z)}[f_w(G_\theta(z))]$$

**Advantage over JSD**: The Wasserstein distance is finite even when $p_r$ and $p_g$ have non-overlapping support (unlike JSD, which saturates at $\log 2$). This resolves the **vanishing gradient problem** of vanilla GANs.

### 5.5 Autoregressive LM Loss

For a sequence $\mathbf{x} = (x_1, \ldots, x_T)$ with vocabulary $\mathcal{V}$, the **language modelling loss** is:
$$\mathcal{L}_{\text{LM}} = -\frac{1}{T}\sum_{t=1}^T \log p_{\boldsymbol{\theta}}(x_t \mid x_1, \ldots, x_{t-1})$$

**Perplexity**:
$$\operatorname{PPL}(\boldsymbol{\theta}) = \exp(\mathcal{L}_{\text{LM}}) = \exp\!\left(-\frac{1}{T}\sum_{t=1}^T \log p_{\boldsymbol{\theta}}(x_t \mid x_{<t})\right)$$

Perplexity is the geometric mean of $1/p(x_t \mid x_{<t})$: the average number of tokens the model considers equally likely as the next token. GPT-2 achieved PPL~18 on PTB; GPT-4 achieves PPL<10 on most benchmarks.

**Bits per byte (BPB)**: Platform-agnostic version:
$$\operatorname{BPB} = \frac{\mathcal{L}_{\text{LM}}}{\log 2 \cdot \bar{b}}$$

where $\bar{b}$ is the average bytes per token. Allows comparing models with different tokenisations.

**Masked LM loss** (BERT, Devlin et al. 2018): mask 15% of tokens and predict only the masked positions:
$$\mathcal{L}_{\text{MLM}} = -\frac{1}{|M|}\sum_{t \in M} \log p_{\boldsymbol{\theta}}(x_t \mid \mathbf{x}_{\setminus M})$$

This is ~6.7$\times$ more data-efficient per token but requires 2-stage pretraining (MLM then fine-tuning).

**For AI**: The autoregressive LM loss scales predictably with compute (Kaplan et al. 2020 scaling laws). The critical insight is that cross-entropy loss on next-token prediction is a proxy for general intelligence: a model that can predict any continuation of any text must understand language, facts, reasoning, and code — making it a surprisingly powerful proxy objective.

---

## 6. Ranking and Contrastive Losses

### 6.1 Pairwise Ranking Loss

For two examples $x_i, x_j$ where $x_i$ should rank higher (score $s_i > s_j$):
$$\ell_{\text{rank}}(s_i, s_j) = \max(0, m - s_i + s_j)$$

where $m > 0$ is a margin. The loss is zero when $s_i - s_j \geq m$ and positive otherwise.

**RankNet** (Burges et al. 2005) uses a probabilistic formulation:
$$P_{ij} = \sigma(s_i - s_j), \quad \ell = -\bar{P}_{ij}\log P_{ij} - (1-\bar{P}_{ij})\log(1-P_{ij})$$

where $\bar{P}_{ij}$ is the empirical probability that $i$ ranks above $j$.

### 6.2 Triplet Loss

For anchor $a$, positive $p$ (same class), and negative $n$ (different class):
$$\ell_{\text{triplet}}(a, p, n) = \max(0,\; \lVert f(a) - f(p) \rVert_2^2 - \lVert f(a) - f(n) \rVert_2^2 + m)$$

**What it enforces**: The anchor–positive distance must be at least margin $m$ smaller than the anchor–negative distance in embedding space.

**Online triplet mining**: With a batch of $B$ samples, there are $O(B^3)$ triplets — most are trivially easy (already satisfied). **Hard negative mining** selects the hardest negatives (closest negatives in the current embedding space) to form difficult triplets. **Semi-hard negatives** (negatives farther than the positive but within the margin) are often more stable.

**For AI**: Triplet loss is used in:
- **FaceNet** (Schroff et al. 2015): the original large-scale face recognition system
- **Sentence embeddings**: training sentence-BERT (SBERT) with NLI triplets
- **Code retrieval**: CodeBERT trained with code-docstring triplets

### 6.3 InfoNCE and NT-Xent

**InfoNCE** (Oord et al. 2018, Contrastive Predictive Coding):
$$\ell_{\text{InfoNCE}} = -\log\frac{\exp(f(\mathbf{x})^\top g(\mathbf{x}^+)/\tau)}{\exp(f(\mathbf{x})^\top g(\mathbf{x}^+)/\tau) + \sum_{j=1}^{K} \exp(f(\mathbf{x})^\top g(\mathbf{x}_j^-)/\tau)}$$

where $\mathbf{x}^+$ is a positive (matched) sample, $\mathbf{x}_j^-$ are $K$ negatives, and $\tau$ is temperature.

**NT-Xent** (Normalised Temperature-scaled Cross Entropy, Chen et al. 2020, SimCLR): with a batch of $N$ examples and their augmented views:
$$\ell_{i,j} = -\log\frac{\exp(\operatorname{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k=1}^{2N}\mathbf{1}_{k\neq i}\exp(\operatorname{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}$$

**Mutual Information Interpretation** (van den Oord et al. 2018): InfoNCE is a lower bound on the mutual information between positive pairs:
$$I(\mathbf{x}; \mathbf{x}^+) \geq \log K - \ell_{\text{InfoNCE}}$$

Maximising InfoNCE maximises a lower bound on MI. This gives a principled reason why contrastive learning works: it learns representations that preserve information about the positive-pair relationship.

**Temperature $\tau$ effect**:
- $\tau \to 0$: loss concentrates on the hardest negative — equivalent to hard-negative mining
- $\tau \to \infty$: all negatives are treated equally — equivalent to random negative sampling
- Optimal $\tau$ balances hard-negative focus with training stability

**CLIP Loss** (Radford et al. 2021): symmetric InfoNCE over image-text pairs:
$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2}\left[\ell_{\text{InfoNCE}}(\mathbf{z}^I \to \mathbf{z}^T) + \ell_{\text{InfoNCE}}(\mathbf{z}^T \to \mathbf{z}^I)\right]$$

With batch size $N$, there are $N^2$ possible pairs; only $N$ are positive. This scales as $O(N)$ negatives per positive.

### 6.4 DPO Loss

**Direct Preference Optimisation** (Rafailov et al. 2023) aligns language models to human preferences without a separate reward model.

**Bradley-Terry preference model**: Given two responses $y_w$ (preferred) and $y_l$ (dispreferred) to prompt $x$:
$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l))$$

where $r$ is an implicit reward.

**Key insight**: Under the RLHF optimal policy $\pi^*$, the reward can be written as:
$$r(x, y) = \beta\log\frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta\log Z(x)$$

Substituting into the Bradley-Terry model and taking log:
$$\mathcal{L}_{\text{DPO}}(\pi_{\boldsymbol{\theta}}) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log\sigma\!\left(\beta\log\frac{\pi_{\boldsymbol{\theta}}(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta\log\frac{\pi_{\boldsymbol{\theta}}(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

**Interpretation**: DPO increases the log-likelihood of preferred responses while decreasing the log-likelihood of dispreferred responses, weighted by how much the model currently deviates from reference.

**For AI**: DPO and its variants (IPO, ORPO, SimPO) have largely replaced PPO-based RLHF for preference alignment due to simpler training:
- No separate reward model required
- No on-policy sampling during training
- More stable (no RL instability)
- Used in Llama 3 instruct, Phi-3, Zephyr, and most modern aligned models

---

## 7. Sequence and Structured Losses

### 7.1 Sequence Cross-Entropy and Teacher Forcing

For sequence-to-sequence tasks (translation, summarisation), the **sequence cross-entropy**:
$$\mathcal{L}_{\text{seq}} = -\sum_{t=1}^T \log p_{\boldsymbol{\theta}}(y_t \mid y_1, \ldots, y_{t-1}, \mathbf{x})$$

**Teacher forcing**: During training, the decoder receives **gold tokens** $y_{<t}$ from the reference sequence as input, not its own predictions $\hat{y}_{<t}$. This prevents error accumulation but creates a **train-test mismatch** (exposure bias): at test time, the model must use its own (potentially incorrect) predictions.

**Scheduled sampling** (Bengio et al. 2015): interpolate between teacher forcing and self-prediction during training. With probability $\epsilon_t$ (decreasing), use the model's own prediction; otherwise use the gold token.

**Minimum Bayes Risk (MBR) decoding**: Instead of maximising $p(\mathbf{y} \mid \mathbf{x})$, select the hypothesis with the lowest expected loss under a reference metric (BLEU, METEOR, COMET). This decouples the training loss from the evaluation metric.

### 7.2 CTC Loss

**Connectionist Temporal Classification** (Graves et al. 2006) trains sequence models without alignment labels. Given input frame sequence $\mathbf{x} = (x_1, \ldots, x_T)$ and output label sequence $\mathbf{y} = (y_1, \ldots, y_U)$ with $U \leq T$:

$$p(\mathbf{y} \mid \mathbf{x}) = \sum_{\pi \in \mathcal{B}^{-1}(\mathbf{y})} \prod_{t=1}^T p(\pi_t \mid x_t)$$

where $\mathcal{B}^{-1}(\mathbf{y})$ is the set of all valid CTC paths that collapse to $\mathbf{y}$ after removing blanks and repeated characters.

**Forward-backward algorithm**: $p(\mathbf{y} \mid \mathbf{x})$ is computed in $O(TU)$ via dynamic programming, analogous to the HMM forward-backward pass.

**For AI**: CTC is used in **Whisper** (OpenAI, Radford et al. 2022) as one training objective for speech recognition without forced alignment. Modern ASR models often combine CTC with attention decoder losses.

### 7.3 REINFORCE and Policy Gradient Loss

For non-differentiable objectives (e.g., BLEU score, human feedback), we use the **REINFORCE** estimator (Williams 1992):
$$\nabla_{\boldsymbol{\theta}} \mathbb{E}_{\mathbf{y} \sim \pi_{\boldsymbol{\theta}}}[R(\mathbf{y})] = \mathbb{E}_{\mathbf{y} \sim \pi_{\boldsymbol{\theta}}}[R(\mathbf{y}) \nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(\mathbf{y})]$$

This is the **score function estimator** (also called REINFORCE or log-derivative trick). The loss to minimise is:
$$\mathcal{L}_{\text{REINFORCE}} = -R(\mathbf{y}) \log \pi_{\boldsymbol{\theta}}(\mathbf{y})$$

**High variance problem**: $R(\mathbf{y})$ can vary widely across samples, leading to unstable training. **Baseline subtraction**: use $R(\mathbf{y}) - b$ where $b$ is a baseline (e.g., average reward or critic estimate) that reduces variance without introducing bias.

**PPO-Clip** (Schulman et al. 2017): the standard policy gradient in RLHF:
$$\mathcal{L}_{\text{PPO}} = -\mathbb{E}_t\left[\min\!\left(r_t(\boldsymbol{\theta})\hat{A}_t,\; \operatorname{clip}(r_t(\boldsymbol{\theta}), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\right)\right]$$

where $r_t(\boldsymbol{\theta}) = \pi_{\boldsymbol{\theta}}(a_t \mid s_t) / \pi_{\text{old}}(a_t \mid s_t)$ is the probability ratio and $\hat{A}_t$ is the advantage estimate.

### 7.4 Reward Model Loss in RLHF

In RLHF (Ouyang et al. 2022, InstructGPT), a reward model $r_\phi$ is trained on preference data:
$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log\sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

This is the **Bradley-Terry cross-entropy**: the loss is minimised when $r_\phi(x, y_w) - r_\phi(x, y_l) \to \infty$ (perfect separation).

**Calibration concern**: The Bradley-Terry model makes strong assumptions. If the reward model is overfit to the annotator's preferences (rather than the true objective), it will be exploited by the RL optimisation — a manifestation of **reward hacking**.

**For AI**: Modern RLHF pipelines (Claude 3, GPT-4, Llama 3) use either:
1. **Reward model + PPO**: train a separate reward model, then use PPO to optimise the LM against it
2. **DPO/ORPO**: directly optimise the preference loss without a reward model
3. **Constitutional AI** (Anthropic): generate critique and revision pairs, use them as preference data


---

## 8. Loss Function Properties

### 8.1 Convexity

**Definition (Convex Loss)**. A loss function $\ell(\hat{y}, y)$ is **convex in $\hat{y}$** if for all $\hat{y}_1, \hat{y}_2$ and $\alpha \in [0,1]$:
$$\ell(\alpha\hat{y}_1 + (1-\alpha)\hat{y}_2, y) \leq \alpha\ell(\hat{y}_1, y) + (1-\alpha)\ell(\hat{y}_2, y)$$

**Why convexity matters**: For a convex loss and a convex model class (e.g., linear models), the training objective $\mathcal{L}(\boldsymbol{\theta})$ is convex — any local minimum is a global minimum, and gradient descent is guaranteed to converge.

**Convexity classification table**:

| Loss | Convex in $\hat{y}$? | Notes |
|---|---|---|
| MSE | Yes (strongly) | Hessian $= 2I$ |
| MAE | Yes | Not strictly convex |
| Huber | Yes | Smooth, with Lipschitz gradient |
| Quantile | Yes | Piecewise linear |
| BCE | Yes (in logit $z$) | Convex in $z = \log(\hat{p}/(1-\hat{p}))$ |
| Categorical CE | Yes (in logits) | Softmax-CE is jointly convex in logits |
| Hinge | Yes | Piecewise linear |
| Focal loss | No (in logits) | $(1-\hat{p})^\gamma$ introduces non-convexity |
| Triplet loss | No | Non-convex in embedding space |
| InfoNCE | No | Denominator creates non-convexity |

**For neural networks**: Even with convex losses, the training objective $\mathcal{L}(\boldsymbol{\theta})$ is non-convex in $\boldsymbol{\theta}$ (because the model $f_{\boldsymbol{\theta}}$ is non-linear). Convexity of the loss is still valuable: it ensures clean gradients and curvature analysis at the final layer.

### 8.2 Calibration and Proper Scoring Rules

**Definition (Proper Scoring Rule)**. A loss $\ell: \hat{\mathcal{Y}} \times \mathcal{Y} \to \mathbb{R}$ is a **proper scoring rule** if the expected loss is minimised by the true conditional distribution:
$$\mathbb{E}_{y \sim P(y \mid \mathbf{x})}[\ell(P(y \mid \mathbf{x}), y)] \leq \mathbb{E}_{y \sim P(y \mid \mathbf{x})}[\ell(q, y)] \quad \forall q \neq P(y \mid \mathbf{x})$$

**Theorem**: A loss $\ell(\hat{p}, y)$ is a proper scoring rule iff minimising $\mathbb{E}[\ell(\hat{p}, y)]$ with respect to $\hat{p}$ yields $\hat{p} = P(y \mid \mathbf{x})$.

**Examples of proper scoring rules**:
- Cross-entropy: $-\log\hat{p}_y$ (proper)
- Brier score: $\sum_k (\hat{p}_k - y_k)^2$ (proper)
- Log-loss: equivalent to cross-entropy (proper)

**Non-example (improper)**:
- Accuracy: $\mathbf{1}[\hat{y} = y]$ — improper because maximising it encourages outputting confident predictions, not calibrated probabilities

**Expected Calibration Error (ECE)**:
$$\operatorname{ECE} = \sum_{m=1}^M \frac{|B_m|}{n}\left|\operatorname{acc}(B_m) - \operatorname{conf}(B_m)\right|$$

where $B_m$ are bins of confidence levels. ECE measures how much the predicted confidence deviates from actual accuracy. A well-calibrated model has $\operatorname{conf}(B_m) \approx \operatorname{acc}(B_m)$ for all bins.

**For AI**: Modern LLMs are often **overconfident** (ECE > 0). Temperature scaling (Guo et al. 2017) calibrates post-hoc by finding $\tau^*$ that minimises NLL on a held-out set. Label smoothing reduces overconfidence during training.

### 8.3 Fisher Consistency

**Definition**. A loss $\ell$ is **Fisher consistent** (or Bayes consistent) for a task if minimising the expected loss yields the Bayes optimal predictor for that task.

**Theorem (Fisher consistency for cross-entropy)**. The categorical cross-entropy is Fisher consistent for multi-class classification: the minimiser of $\mathbb{E}[-\log\hat{p}_{y}]$ over $\hat{\mathbf{p}} \in \Delta^{K-1}$ is $\hat{p}_k = P(y = k \mid \mathbf{x})$ for each $k$.

**Proof**: By Lagrangian optimisation with the simplex constraint $\sum_k \hat{p}_k = 1$:
$$\nabla_{\hat{\mathbf{p}}} \left[\mathbb{E}\left[-\sum_k P(y=k|\mathbf{x})\log\hat{p}_k\right] + \lambda\left(\sum_k \hat{p}_k - 1\right)\right] = 0$$
$$\Rightarrow -P(y=k|\mathbf{x})/\hat{p}_k + \lambda = 0 \Rightarrow \hat{p}_k \propto P(y=k|\mathbf{x}) \Rightarrow \hat{p}_k = P(y=k|\mathbf{x})$$

**For AI**: Fisher consistency is why cross-entropy is the standard for classification. When the model capacity is sufficient and training converges, cross-entropy training directly learns the true class probabilities — the foundation of all downstream probabilistic reasoning.

### 8.4 Lipschitz Continuity of Gradients

**Definition ($L$-smooth loss)**. A loss $\mathcal{L}$ is **$L$-smooth** if its gradient is $L$-Lipschitz:
$$\lVert \nabla \mathcal{L}(\boldsymbol{\theta}_1) - \nabla \mathcal{L}(\boldsymbol{\theta}_2) \rVert \leq L\lVert \boldsymbol{\theta}_1 - \boldsymbol{\theta}_2 \rVert$$

Equivalently, the Hessian satisfies $\lVert H_{\mathcal{L}} \rVert_2 \leq L$.

**Consequence for gradient descent**: If $\mathcal{L}$ is $L$-smooth, then gradient descent with step size $\eta \leq 1/L$ is guaranteed to decrease the loss:
$$\mathcal{L}(\boldsymbol{\theta} - \eta\nabla\mathcal{L}(\boldsymbol{\theta})) \leq \mathcal{L}(\boldsymbol{\theta}) - \frac{\eta}{2}\lVert \nabla\mathcal{L}(\boldsymbol{\theta})\rVert^2$$

**Smoothness of common losses**:
- MSE: $L$-smooth with $L = 2$ (constant Hessian)
- BCE (in logit space): $L$-smooth with $L = 1/4$ (max second derivative of sigmoid is $1/4$)
- MAE: **not smooth** (subgradient discontinuous at zero)
- Huber: $L$-smooth with $L = 1$ (gradient clipped to $[-\delta,\delta]$, Lipschitz constant $= 1$)

**For AI**: Smooth losses enable convergence guarantees for Adam and other adaptive optimisers. Non-smooth losses (MAE) require subgradient methods or smooth approximations (Huber, log-cosh). **Gradient clipping** artificially enforces a smoothness condition by capping the gradient norm.

### 8.5 Robustness to Outliers

**Breakdown point** of a loss: the fraction of arbitrarily corrupted data points that can be added before the optimal predictor is driven to infinity.

- MSE: 0% breakdown point (one outlier with $|r| \to \infty$ shifts the mean to $\pm\infty$)
- MAE: 50% breakdown point (up to half the data can be outliers)
- Huber: same as MAE for large $\delta$; as $\delta \to 0$, approaches MAE's robustness
- Cross-entropy: 0% for confident wrong predictions (one sample with $y = 1, \hat{p} = 0$ gives loss $= \infty$)

**Influence function** (Koh & Liang 2017): measures how much removing a training point $z$ changes the model parameters:
$$\mathcal{I}(z) = H_{\boldsymbol{\theta}^*}^{-1} \nabla_{\boldsymbol{\theta}} \ell(z; \boldsymbol{\theta}^*)$$

where $H_{\boldsymbol{\theta}^*}$ is the Hessian at convergence. This is the formal tool for understanding which training examples are "outliers" in the model's view.

**For AI**: Influence functions were used to identify **poisoning attacks** on NLP models (Wallace et al. 2021): specific training examples that disproportionately affect model behaviour. Robust losses (Huber, symmetric cross-entropy) are important for training on web-scraped data with noisy labels.

---

## 9. Loss Landscape Geometry

### 9.1 Gradient and Hessian Analysis

The **loss landscape** is the hypersurface $\{(\boldsymbol{\theta}, \mathcal{L}(\boldsymbol{\theta})) : \boldsymbol{\theta} \in \mathbb{R}^P\}$. Its geometry determines:
- Whether gradient descent converges
- The quality of the minimum found
- The generalisation of the trained model

**Gradient**: $\nabla\mathcal{L}(\boldsymbol{\theta}) \in \mathbb{R}^P$ — points in the direction of steepest ascent.

**Hessian**: $H \in \mathbb{R}^{P \times P}$ where $H_{ij} = \partial^2\mathcal{L}/\partial\theta_i\partial\theta_j$ — captures curvature.

**Key Hessian quantities**:
- $\lambda_{\max}(H)$: largest eigenvalue — determines the maximum useful learning rate ($\eta < 2/\lambda_{\max}$)
- $\lambda_{\min}(H)$: smallest eigenvalue — if negative, indicates a saddle point
- $\kappa(H) = \lambda_{\max}/\lambda_{\min}$: condition number — high $\kappa$ means the landscape is narrow in some directions and flat in others, causing slow convergence

**Critical points**: $\nabla\mathcal{L}(\boldsymbol{\theta}) = 0$. Classification:
- **Local minimum**: all eigenvalues of $H$ positive ($H \succ 0$)
- **Saddle point**: $H$ has both positive and negative eigenvalues
- **Local maximum**: all eigenvalues of $H$ negative ($H \prec 0$)

**For AI**: Dauphin et al. (2014) showed empirically that in high-dimensional networks, critical points with high loss are almost always **saddle points**, not local minima. This is because for a random matrix in high dimension, having all positive eigenvalues is exponentially unlikely. Gradient descent escapes saddle points via stochastic noise from minibatches.

### 9.2 Saddle Points vs Local Minima

**Empirical observations** from neural network training:
1. Local minima found by SGD generalise well — gradient descent has a "flat minimum bias"
2. Sharp minima (high $\lambda_{\max}$) generalise poorly; flat minima (low $\lambda_{\max}$) generalise well (Hochreiter & Schmidhuber 1997, Keskar et al. 2017)
3. Wide minima are found by small-batch SGD; large-batch SGD finds sharp minima

**Loss of plasticity**: Models trained for too long on a fixed distribution become "stuck" in sharp minima and cannot adapt to new data — relevant to **continual learning** and **LLM fine-tuning**.

### 9.3 Loss Landscape Visualisation

The **filter normalisation** method (Li et al. 2018) projects the high-dimensional landscape onto 2D:

Choose two random direction vectors $\boldsymbol{\delta}, \boldsymbol{\eta} \in \mathbb{R}^P$ (normalised filter-wise). Visualise:
$$\mathcal{L}(\boldsymbol{\theta}^* + \alpha\boldsymbol{\delta} + \beta\boldsymbol{\eta})$$

as a function of $\alpha, \beta \in [-1, 1]$.

**Key finding**: ResNets have smooth, bowl-like landscapes; plain networks without residual connections have chaotic landscapes with many ridges and valleys. This explains why residual connections dramatically improve trainability.

### 9.4 Sharpness-Aware Minimisation

**SAM** (Foret et al. 2021) minimises the worst-case loss in a neighbourhood of $\boldsymbol{\theta}$:
$$\mathcal{L}_{\text{SAM}}(\boldsymbol{\theta}) = \max_{\lVert\boldsymbol{\epsilon}\rVert \leq \rho} \mathcal{L}(\boldsymbol{\theta} + \boldsymbol{\epsilon})$$

**SAM update**: Approximate the inner maximum with one gradient step:
$$\hat{\boldsymbol{\epsilon}} = \rho \frac{\nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})}{\lVert \nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta})\rVert}$$

Then update $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta} + \hat{\boldsymbol{\epsilon}})$.

**For AI**: SAM consistently improves generalisation by 1–3% on ImageNet and 0.5–1% on standard NLP benchmarks. It has been integrated into training recipes for ViT, ResNet, and language model fine-tuning.

---

## 10. Regularisation as Loss Terms

### 10.1 L2 / Weight Decay

Adding L2 regularisation to any loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2}\lVert\boldsymbol{\theta}\rVert_2^2$$

**Bayesian interpretation**: MAP estimation with a Gaussian prior $p(\boldsymbol{\theta}) = \mathcal{N}(\mathbf{0}, \lambda^{-1}I)$:
$$\log p(\boldsymbol{\theta} \mid \mathcal{D}) \propto \log p(\mathcal{D} \mid \boldsymbol{\theta}) + \log p(\boldsymbol{\theta}) = -\mathcal{L}_{\text{data}} - \frac{\lambda}{2}\lVert\boldsymbol{\theta}\rVert_2^2$$

**Weight decay in adaptive optimisers**: L2 regularisation in Adam scales the regularisation by the adaptive learning rate — this is **not** equivalent to weight decay. **AdamW** (Loshchilov & Hutter 2018) decouples weight decay from the adaptive update:
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \varepsilon} + \lambda\boldsymbol{\theta}_t\right)$$

AdamW is the standard optimiser for LLM training (GPT series, Llama, Mistral all use AdamW).

### 10.2 L1 Regularisation

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda\lVert\boldsymbol{\theta}\rVert_1$$

**Bayesian interpretation**: MAP with Laplace prior $p(\theta_i) \propto \exp(-\lambda|\theta_i|)$.

**Sparsity**: The L1 penalty has a non-differentiable corner at $\theta_i = 0$. The subgradient at $\theta_i = 0$ is any value in $[-\lambda, \lambda]$. This creates a **soft-thresholding** effect: parameters with gradient magnitude below $\lambda$ are driven exactly to zero. This is the basis of LASSO regression.

**For AI**: L1 regularisation is used in:
- **Sparse Autoencoders** (Anthropic 2023): learn sparse feature representations of LLM residual stream activations; the sparsity is enforced by L1 penalty on activations
- **Magnitude pruning**: L1 regularisation + threshold pruning reduces model size by eliminating small-magnitude weights

### 10.3 Elastic Net

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_1\lVert\boldsymbol{\theta}\rVert_1 + \lambda_2\lVert\boldsymbol{\theta}\rVert_2^2$$

Combines sparsity (L1) with coefficient stability (L2). Particularly useful when features are correlated — L1 tends to pick one from a correlated group arbitrarily, while elastic net tends to retain all correlated features with equal small coefficients.

### 10.4 Dropout as Regularisation

**Dropout** (Srivastava et al. 2014) during training randomly zeros activations with probability $p$:
$$\tilde{h}_i = h_i \cdot m_i, \quad m_i \sim \operatorname{Bern}(1-p)$$

**Regularisation interpretation**: Dropout can be viewed as adding noise to the activations, equivalent to implicit L2 regularisation on the weights (Wager et al. 2013). The effective regulariser is:
$$\Omega(\boldsymbol{\theta}) = p(1-p)\sum_{l,i,j} w^{[l]}_{ij}\sum_k (h^{[l-1]}_k)^2$$

**Bayesian interpretation**: Dropout training approximates variational inference in a Gaussian process (Gal & Ghahramani 2016). **MC Dropout** at test time (running multiple forward passes with dropout active) gives uncertainty estimates.

---

## 11. Computational Considerations

### 11.1 Numerical Stability

**Problem**: Computing $\log\sum_k e^{z_k}$ directly overflows when $z_k > 88$ (float32 range) and underflows when all $z_k \ll 0$.

**Log-sum-exp trick**:
$$\log\sum_k e^{z_k} = m + \log\sum_k e^{z_k - m}, \quad m = \max_k z_k$$

Since $e^{z_k - m} \leq 1$, no overflow occurs. And $e^{z_k - m} = 1$ for the maximum term, preventing underflow.

**Numerically stable cross-entropy**: Never compute `softmax(z)` then `log(...)`. Instead:
$$-\log\hat{p}_{y^*} = -z_{y^*} + \log\sum_k e^{z_k} = -(z_{y^*} - m) + \log\sum_k e^{z_k - m}$$

**Numerically stable BCE**: Use `log1p(-p)` instead of `log(1-p)` for small $p$.

**For AI**: PyTorch's `F.cross_entropy` and `F.binary_cross_entropy_with_logits` implement numerically stable versions automatically. Always use these instead of manual `softmax + log`.

### 11.2 Multi-Task Loss Balancing

For $T$ tasks with losses $\mathcal{L}_1, \ldots, \mathcal{L}_T$:
$$\mathcal{L}_{\text{total}} = \sum_{t=1}^T w_t \mathcal{L}_t$$

**Naive approach**: Set $w_t = 1$. Problem: tasks with different magnitudes or gradient scales dominate.

**Uncertainty weighting** (Kendall et al. 2018): Model each task with a homoscedastic uncertainty $\sigma_t^2$:
$$\mathcal{L}_{\text{total}} = \sum_{t=1}^T \frac{1}{2\sigma_t^2}\mathcal{L}_t + \log\sigma_t$$

Minimising over $\sigma_t$ gives $\sigma_t^2 \propto \mathcal{L}_t$ at optimum — the model automatically learns to weight tasks by their uncertainty.

**GradNorm** (Chen et al. 2018): Normalise gradient magnitudes across tasks:
$$w_t \leftarrow w_t \cdot \frac{\tilde{G}_t}{G_W(t)}$$

where $G_W(t) = \lVert\nabla\mathcal{L}_t\rVert_2$ and $\tilde{G}_t$ is the target gradient norm.

**For AI**: Multi-task losses are critical in:
- **Instruction-following LLMs**: balance helpfulness vs harmlessness objectives
- **Vision-language models**: balance image captioning, VQA, OCR tasks
- **AlphaFold 2**: balance structure prediction, distogram, and frame alignment losses

### 11.3 Loss Reduction Modes

When aggregating per-sample losses:
- **`reduction='mean'`**: $\mathcal{L} = \frac{1}{n}\sum_i \ell_i$ — default in most frameworks. Learning rate is scale-invariant to batch size.
- **`reduction='sum'`**: $\mathcal{L} = \sum_i \ell_i$ — the learning rate must scale with $1/n$ to be equivalent.
- **`reduction='none'`**: Returns per-sample losses $(\ell_1, \ldots, \ell_n)$ — used for sample reweighting, curriculum learning, or custom aggregation.

**Masked losses**: For sequence models with padding, mask out padding tokens:
```python
# PyTorch pattern
loss = F.cross_entropy(logits, targets, reduction='none')  # (B, T)
loss = (loss * mask).sum() / mask.sum()  # average over non-padding tokens
```

### 11.4 Mixed Precision and Loss Scaling

Training with FP16 (half-precision) reduces memory and increases throughput but introduces numerical issues:
- FP16 range: approximately $[10^{-8}, 65504]$ — gradients can **underflow** to zero
- **Dynamic loss scaling**: multiply the loss by a large scalar $S$ before backward, then divide gradients by $S$ after. If overflow occurs (gradient is NaN/Inf), skip the update and reduce $S$.

$$\mathcal{L}_{\text{scaled}} = S \cdot \mathcal{L}, \quad \nabla_{\boldsymbol{\theta}} \leftarrow \nabla_{\boldsymbol{\theta}} / S$$

**BF16 advantage**: BFloat16 has the same exponent range as FP32 but reduced mantissa precision — it rarely underflows and does not require loss scaling. This is why modern training (GPT-4, Llama 3) uses BF16 instead of FP16.

---

## 12. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Using MSE for classification | MSE is not a proper scoring rule for binary outputs; it converges to the wrong Bayes predictor ($\mathbb{E}[y]$ not $P(y)$) | Use BCE or CE which are proper and Fisher consistent |
| 2 | Computing `log(softmax(z))` separately | Numerically unstable: softmax saturates at large $z$, then log($\approx$1) loses precision | Use `log_softmax(z)` or `F.cross_entropy(logits, targets)` directly |
| 3 | Ignoring the sign of KL divergence direction | $D_{\mathrm{KL}}(p \| q) \neq D_{\mathrm{KL}}(q \| p)$; mode-covering vs mode-seeking has opposite implications | Explicitly specify: "KL from data to model" vs "KL from model to data" |
| 4 | Using L2 regularisation with Adam and calling it weight decay | L2 reg scales differently with Adam's adaptive learning rates — the effective regularisation strength varies per-parameter | Use AdamW (decoupled weight decay) for proper weight decay in Adam |
| 5 | Reducing sequence losses over all positions including padding | Padding tokens inflate the normalisation constant, diluting the actual signal | Mask padding positions and compute mean only over valid tokens |
| 6 | Setting temperature $\tau = 0$ in InfoNCE | As $\tau \to 0$, loss focuses entirely on the single hardest negative — extremely high variance gradients, training instability | Use $\tau \in [0.05, 0.5]$; tune on a validation metric |
| 7 | Forgetting the $\tau^2$ scaling factor in knowledge distillation | The KL distillation term has gradients scaled by $1/\tau^2$ relative to the hard-label CE term — they become unbalanced | Include the $\tau^2$ multiplier: $\alpha\tau^2 D_{\mathrm{KL}}(\text{teacher}\|\text{student})$ |
| 8 | Using CE without label smoothing and reporting calibration | Plain CE converges to overconfident logits; ECE will be high even if accuracy is good | Apply label smoothing ($\varepsilon = 0.1$) during training or temperature scale post-hoc |
| 9 | Interpreting high loss as evidence of wrong architecture | Loss magnitude depends on task scale — a CE loss of 2.0 for 100 classes (log(100)$\approx$4.6 is chance) is actually good | Normalise: compare to the uniform-prediction baseline loss |
| 10 | Using `reduction='sum'` without adjusting learning rate | Sum loss scales linearly with batch size — the gradient is $B$ times larger than with mean reduction | Use `reduction='mean'` or scale learning rate by $1/B$ when using sum |

---

## 13. Exercises

**Exercise 1** ★ — **MSE as Gaussian MLE**

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$ with $y_i = f_{\boldsymbol{\theta}}(x_i) + \varepsilon_i$, $\varepsilon_i \sim \mathcal{N}(0, \sigma^2)$.

(a) Write the log-likelihood $\log p(\mathcal{D} \mid \boldsymbol{\theta})$.

(b) Show that maximising the log-likelihood is equivalent to minimising the MSE.

(c) What happens to the MLE estimate of $\boldsymbol{\theta}$ if one data point has $y_i$ replaced by $y_i + M$ for very large $M$?

**Exercise 2** ★ — **Bayes Optimal Predictors**

For a distribution $P(y \mid x)$ where $y \in \{0, 5, 10\}$ with probabilities $\{0.1, 0.5, 0.4\}$:

(a) Compute the Bayes optimal prediction under MSE loss.

(b) Compute the Bayes optimal prediction under MAE loss.

(c) Compute the Bayes optimal prediction under 0-1 loss.

(d) Which predictor minimises the expected Huber loss (hint: it depends on $\delta$)?

**Exercise 3** ★ — **Cross-Entropy Gradient**

For logits $\mathbf{z} = (z_1, \ldots, z_K)$ and true class $y^* = k$, with $\hat{\mathbf{p}} = \operatorname{softmax}(\mathbf{z})$:

(a) Show that $\partial \ell_{\text{CE}} / \partial z_k = \hat{p}_k - \mathbf{1}[k = y^*]$.

(b) What is the gradient when the model is perfectly calibrated ($\hat{p}_{y^*} = 1$)?

(c) What is the maximum gradient magnitude for any single logit?

**Exercise 4** ★★ — **Huber Loss: Gradient Clipping Connection**

(a) Write out the Huber loss $\ell_\delta$ and its gradient for $\delta = 1$.

(b) Show that gradient descent on Huber loss is equivalent to gradient descent on MSE with gradient clipping at magnitude $\delta$.

(c) If the residuals follow a Student's $t$-distribution with 3 degrees of freedom, what value of $\delta$ would you choose and why?

**Exercise 5** ★★ — **Fisher Consistency of Hinge Loss**

(a) For binary classification with labels $y \in \{-1, +1\}$, write the expected hinge loss as a function of $f(\mathbf{x})$ and $\eta(\mathbf{x}) = P(y = 1 \mid \mathbf{x})$.

(b) Find the minimiser $f^*(\mathbf{x})$ of the expected hinge loss for a given $\eta$.

(c) Show that $\operatorname{sign}(f^*(\mathbf{x})) = \operatorname{sign}(\eta(\mathbf{x}) - 1/2)$, confirming Fisher consistency.

**Exercise 6** ★★ — **InfoNCE and Temperature**

Given $N = 256$ samples with one positive and $N-1$ negatives per query:

(a) Write the InfoNCE loss for a single query $\mathbf{q}$ with positive $\mathbf{k}^+$ and negatives $\mathbf{k}_j^-$.

(b) Compute the gradient of InfoNCE with respect to the positive similarity $s^+ = \mathbf{q}^\top\mathbf{k}^+/\tau$.

(c) Show that as $\tau \to 0$, the gradient concentrates on the single hardest negative.

(d) What is the minimum possible InfoNCE loss (over all $\tau$)?

**Exercise 7** ★★ — **ELBO Decomposition**

For a VAE with encoder $q_{\boldsymbol{\phi}}(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\phi}}, \operatorname{diag}(\boldsymbol{\sigma}^2_{\boldsymbol{\phi}}))$ and prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$:

(a) Derive the closed-form KL divergence $D_{\mathrm{KL}}(q_{\boldsymbol{\phi}} \| p)$ for a $d$-dimensional latent space.

(b) For $\boldsymbol{\mu} = \mathbf{0}$ and $\boldsymbol{\sigma}^2 = \mathbf{1}$, verify that $D_{\mathrm{KL}} = 0$.

(c) What is the maximum value of the KL term when $\lVert\boldsymbol{\mu}\rVert_2 = 1$ and $\boldsymbol{\sigma}^2 = \mathbf{1}$?

**Exercise 8** ★★★ — **DPO Loss Analysis**

(a) Starting from the RLHF optimal policy equation $r(x,y) = \beta\log(\pi^*(y|x)/\pi_{\text{ref}}(y|x)) + \beta\log Z(x)$, derive the DPO loss.

(b) Show that the DPO gradient increases $\pi_{\boldsymbol{\theta}}(y_w \mid x)$ and decreases $\pi_{\boldsymbol{\theta}}(y_l \mid x)$ simultaneously.

(c) What pathology can occur when $\pi_{\boldsymbol{\theta}}(y_w \mid x)$ is already much larger than $\pi_{\text{ref}}(y_w \mid x)$?

**Exercise 9** ★★★ — **Label Smoothing and Calibration**

(a) For $K = 10$ classes and smoothing $\varepsilon = 0.1$, write the smoothed target distribution $\mathbf{y}^{\text{smooth}}$.

(b) Show that the optimal logit gap (between the true class and others) under label-smoothed CE is $(1-\varepsilon)/(\varepsilon/(K-1)) = K-1$ times larger than for the other classes.

(c) Implement label smoothing and verify with a toy example that the cross-entropy with soft targets gives lower ECE than with hard targets.

**Exercise 10** ★★★ — **Multi-Task Uncertainty Weighting**

Given two tasks with losses $\mathcal{L}_1 = 0.5$ (regression) and $\mathcal{L}_2 = 1.2$ (classification):

(a) Using uncertainty weighting, write the combined loss with learnable $\sigma_1, \sigma_2$.

(b) Taking derivatives w.r.t. $\sigma_1$ and $\sigma_2$, find the optimal values $\sigma_1^*, \sigma_2^*$.

(c) What is the effective weight ratio $w_1/w_2$ at the optimum, and does this make intuitive sense?

---

## 14. Why This Matters for AI (2026 Perspective)

| Concept | AI / LLM Impact |
|---|---|
| Autoregressive CE loss | The single loss function behind GPT-4, Llama 3, Gemini, Claude — everything emerges from predicting the next token |
| Proper scoring rules | Guarantee that CE-trained LLMs learn calibrated token probabilities, enabling reliable softmax-based reasoning |
| InfoNCE / NT-Xent | Foundation of all contrastive pretraining: CLIP, ALIGN, DALL-E, ImageBind — the embedding models behind multimodal AI |
| DPO / preference losses | Replace PPO-RLHF in most 2024 aligned models; simpler, stabler, and equally effective for preference alignment |
| KL penalty in RLHF | Prevents reward hacking by keeping the policy close to the pretrained reference; controlled by the $\beta$ hyperparameter |
| ELBO / diffusion ELBO | Loss function for all latent diffusion models (Stable Diffusion, FLUX, Sora); directly connects to VAE theory |
| Label smoothing | Standard in all transformer training recipes since Vaswani et al. 2017; reduces overconfidence, improves calibration |
| Multi-task balancing | Critical for instruction-tuning: helpfulness, harmlessness, and honesty objectives must be balanced |
| Huber / smooth L1 | Default for regression in vision models (bounding boxes, depth, optical flow) — stable even with label noise |
| Fisher consistency | Theoretical justification that CE training converges to the correct Bayes classifier given enough capacity |
| Numerical stability | Prevented training failures in trillion-parameter scale models; log-softmax + CE is now boilerplate code |
| Focal loss | Enabled modern one-stage detectors (YOLO series) by solving class imbalance; idea extended to hard example mining in LLMs |

---

## 15. Conceptual Bridge

Loss functions are the point where mathematics meets measurement — the precise specification of what "learning" means for a given task. Looking backward, this section synthesises ideas from three major preceding chapters: **probability theory** (Chapter 06) provided the MLE interpretation of every standard loss; **information theory** (Chapter 09) gave the cross-entropy / KL framework that unifies classification and generative objectives; **optimisation** (Chapter 08) showed why convexity and smoothness of the loss determine whether gradient descent converges and how fast.

Looking forward, every section in Chapter 13 and beyond builds on loss functions. **Activation functions** (§13-02) directly interact with loss gradient flow — sigmoid saturation kills gradients in BCE; ReLU preserves them. **Normalisation techniques** (§13-03) alter the loss landscape curvature. **Chapter 14** (Math for Specific Models) shows how each model architecture combines a loss with an inductive bias: transformers pair self-attention with cross-entropy; VAEs pair an encoder-decoder architecture with the ELBO; GANs pair a generator with an adversarial loss. **Chapter 15** (Math for LLMs) will revisit the autoregressive CE loss in detail, connecting it to perplexity, scaling laws, and the emergent capabilities that arise from minimising it at scale.

The deeper lesson is that **every design choice in AI is a loss function choice in disguise**. Choosing to use cosine similarity instead of dot product in attention is choosing a different geometry on the embedding space. Choosing temperature scaling at inference is reparameterising the sharpness of the CE loss. Choosing PPO vs DPO is choosing between online and offline gradient estimation for the same underlying preference objective. Understanding losses mathematically means understanding AI systems at their foundations.

```
POSITION IN CURRICULUM: ML-SPECIFIC MATH
════════════════════════════════════════════════════════════════════════

  Prerequisites (from which we draw)        What this section enables
  ═════════════════════════════════          ═════════════════════════

  §06 Probability Theory                    §13-02 Activation Functions
  → MLE, Bayes risk, NLL                    → gradient flow through loss

  §07 Statistics                            §13-03 Normalisation Techniques
  → Estimators, calibration                 → loss landscape curvature

  §08 Optimisation                          §13-04 Sampling Methods
  → Gradient descent on L(theta)            → MCMC for NLL integrals

  §09 Information Theory                    §14 Math for Specific Models
  → KL, cross-entropy, MI                   → ELBO, CE, DPO, InfoNCE

  §12 Functional Analysis                   §15 Math for LLMs
  → Lipschitz, proper scoring rules         → scaling laws, perplexity

                    ┌─────────────────────────────────┐
                    │   13-01 Loss Functions  ◄ YOU   │
                    │   The core objective of every    │
                    │   AI system — in one section.    │
                    └─────────────────────────────────┘

════════════════════════════════════════════════════════════════════════
```

### References

1. Vapnik, V. (1995). *The Nature of Statistical Learning Theory*. Springer.
2. Huber, P. J. (1964). Robust estimation of a location parameter. *Annals of Statistics*.
3. Bartlett, P., Jordan, M., & McAuliffe, J. (2006). Convexity, classification, and risk bounds. *JASA*.
4. Lin, T.-Y., et al. (2017). Focal loss for dense object detection. *ICCV*. (RetinaNet)
5. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NIPS Workshop*.
6. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
7. Oord, A., et al. (2018). Representation learning with contrastive predictive coding. *arXiv:1807.03748*.
8. Chen, T., et al. (2020). A simple framework for contrastive learning. *ICML*. (SimCLR)
9. Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*. (CLIP)
10. Arjovsky, M., et al. (2017). Wasserstein GAN. *ICML*.
11. Rafailov, R., et al. (2023). Direct preference optimization. *NeurIPS*. (DPO)
12. Kingma, D. & Welling, M. (2013). Auto-encoding variational Bayes. *ICLR*.
13. Ouyang, L., et al. (2022). Training language models to follow instructions. *NeurIPS*. (InstructGPT)
14. Foret, P., et al. (2021). Sharpness-aware minimization for efficiently improving generalization. *ICLR*.
15. Kendall, A. & Gal, Y. (2017). Multi-task learning using uncertainty to weigh losses. *CVPR*.
16. Guo, C., et al. (2017). On calibration of modern neural networks. *ICML*.
17. Müller, R., et al. (2019). When does label smoothing help? *NeurIPS*.
18. Belkin, M., et al. (2019). Reconciling modern machine learning practice and the classical bias-variance trade-off. *PNAS*.

---

## Appendix A: Loss Function Zoo — Complete Reference

The following table provides a complete reference for all loss functions discussed in this section:

```
REGRESSION LOSSES: SIDE-BY-SIDE COMPARISON
════════════════════════════════════════════════════════════════════════

  Loss          Formula              Optimal f*(x)   Breakdown   Smooth
  ────────────────────────────────────────────────────────────────────
  MSE           (y - ŷ)²             E[y|x]          0%          Yes
  MAE           |y - ŷ|              median(y|x)     50%         No
  Huber(δ)      ½r² / δ|r|-½δ²      ~median         50%         Yes
  Quantile(τ)   (y-ŷ)(τ-1{y<ŷ})     τ-quantile      —           No
  Log-Cosh      log(cosh(r))         ~median         ~50%        Yes
  Gauss NLL     ½log(σ²)+(y-μ)²/2σ² mean + var      0%          Yes
  ────────────────────────────────────────────────────────────────────

  r = ŷ - y  (residual)

════════════════════════════════════════════════════════════════════════
```

```
CLASSIFICATION LOSSES: SIDE-BY-SIDE COMPARISON
════════════════════════════════════════════════════════════════════════

  Loss            Formula              Proper?   Consistent?   Convex?
  ────────────────────────────────────────────────────────────────────
  0-1 loss        1[ŷ ≠ y]             No        Yes (trivially) No
  BCE             -y log p -(1-y)log(1-p)  Yes   Yes           Yes
  CE              -log p_{y*}          Yes       Yes           Yes
  Hinge           max(0, 1-yŷ)         No        Yes           Yes
  Focal           -(1-p)^γ log p       No        Yes*          No
  Label-smooth    (1-ε)CE + ε·uniform  Yes*      Yes           Yes
  Brier           Σ(p_k - y_k)²        Yes       Yes           Yes
  ────────────────────────────────────────────────────────────────────

  * Approximately / asymptotically proper with appropriate γ/ε

════════════════════════════════════════════════════════════════════════
```

### A.1 Gradient Magnitude Comparison

For regression losses with residual $r = \hat{y} - y$:

| $\lvert r \rvert$ | MSE gradient | MAE gradient | Huber ($\delta=1$) | Log-cosh |
|---|---|---|---|---|
| 0.1 | 0.2 | 1.0 | 0.1 | 0.100 |
| 0.5 | 1.0 | 1.0 | 0.5 | 0.462 |
| 1.0 | 2.0 | 1.0 | 1.0 | 0.762 |
| 2.0 | 4.0 | 1.0 | 1.0 | 0.964 |
| 5.0 | 10.0 | 1.0 | 1.0 | 1.000 |
| 10.0 | 20.0 | 1.0 | 1.0 | 1.000 |

**Reading**: MSE gradient grows unbounded with $|r|$; all robust losses (MAE, Huber, log-cosh) cap at 1.0 (or $\delta$) for large residuals.

### A.2 Loss Value Comparison

| $\lvert r \rvert$ | MSE | MAE | Huber ($\delta=1$) | Log-cosh |
|---|---|---|---|---|
| 0.1 | 0.010 | 0.100 | 0.005 | 0.005 |
| 0.5 | 0.250 | 0.500 | 0.125 | 0.121 |
| 1.0 | 1.000 | 1.000 | 0.500 | 0.434 |
| 2.0 | 4.000 | 2.000 | 1.500 | 1.326 |
| 5.0 | 25.00 | 5.000 | 4.500 | 4.307 |

**Note**: For $|r| \leq \delta$, Huber $= \frac{1}{2}$MSE; for $|r| > \delta$, Huber $\approx$ MAE (shifted).

---

## Appendix B: Detailed Gradient Derivations

### B.1 Softmax Cross-Entropy Gradient (Full Derivation)

Given logits $\mathbf{z} \in \mathbb{R}^K$ with $\hat{p}_k = e^{z_k}/\sum_j e^{z_j}$, and one-hot label $\mathbf{y}$:

$$\ell = -\sum_k y_k \log \hat{p}_k$$

**Step 1**: $\frac{\partial \hat{p}_k}{\partial z_j}$

$$\frac{\partial \hat{p}_k}{\partial z_j} = \begin{cases} \hat{p}_k(1 - \hat{p}_k) & j = k \\ -\hat{p}_k\hat{p}_j & j \neq k \end{cases} = \hat{p}_k(\mathbf{1}_{j=k} - \hat{p}_j)$$

**Step 2**: $\frac{\partial \ell}{\partial z_j}$

$$\frac{\partial \ell}{\partial z_j} = \sum_k \frac{\partial \ell}{\partial \hat{p}_k} \cdot \frac{\partial \hat{p}_k}{\partial z_j}
= \sum_k \left(-\frac{y_k}{\hat{p}_k}\right) \hat{p}_k(\mathbf{1}_{j=k} - \hat{p}_j)$$
$$= \sum_k -y_k(\mathbf{1}_{j=k} - \hat{p}_j)
= -y_j + \hat{p}_j\sum_k y_k
= \hat{p}_j - y_j$$

**Result**: $\nabla_{\mathbf{z}} \ell = \hat{\mathbf{p}} - \mathbf{y}$ — the gradient is simply the prediction residual.

**Matrix form** (batch): For batch matrix $Z \in \mathbb{R}^{n \times K}$ with targets $Y \in \{0,1\}^{n \times K}$:
$$\frac{\partial \mathcal{L}}{\partial Z} = \frac{1}{n}(\hat{P} - Y) \in \mathbb{R}^{n \times K}$$

### B.2 Focal Loss Gradient (Full Derivation)

For binary focal loss with $z$ the logit and $p = \sigma(z)$, $y \in \{0,1\}$:
$$\ell_{\text{focal}} = -\alpha_y (1-p_y)^\gamma \log p_y$$

where $p_y = p$ if $y=1$, else $1-p$.

For $y=1$: $\ell = -\alpha(1-p)^\gamma \log p$

$$\frac{\partial \ell}{\partial z} = \alpha(1-p)^{\gamma-1}[\gamma p \log p - (1-p)](-1)(-1) \cdot \frac{\partial p}{\partial z}$$

Since $\partial p/\partial z = p(1-p)$:
$$\frac{\partial \ell}{\partial z} = \alpha(1-p)^{\gamma-1}[\gamma p \log p + (p-1)] \cdot p(1-p)$$
$$= \alpha p(1-p)^\gamma[\gamma \log p - (1-p)/(1-p)] \cdot (1-p)^{-1}...$$

Simplified: $\frac{\partial \ell}{\partial z} = \alpha(1-p)^\gamma (p-1+\gamma p \log p)$

**Key property**: When $p \approx 1$ (easy, correct), $(1-p)^\gamma \approx 0$ — gradient is suppressed. When $p \approx 0$ (hard, wrong), $(1-p)^\gamma \approx 1$ — gradient is full strength.

### B.3 InfoNCE Gradient Analysis

For a single query $\mathbf{q}$ with similarity scores $s^+ = \mathbf{q}^\top\mathbf{k}^+/\tau$ and $s_j^- = \mathbf{q}^\top\mathbf{k}_j^-/\tau$:

$$\ell = -s^+ + \log\!\left(e^{s^+} + \sum_j e^{s_j^-}\right)$$

$$\frac{\partial \ell}{\partial s^+} = -1 + \frac{e^{s^+}}{e^{s^+} + \sum_j e^{s_j^-}} = \hat{p}_{+} - 1$$

$$\frac{\partial \ell}{\partial s_j^-} = \frac{e^{s_j^-}}{e^{s^+} + \sum_j e^{s_j^-}} = \hat{p}_j$$

**Interpretation**: The gradient pushes the positive score up by $(1-\hat{p}_+)$ and each negative score down by $\hat{p}_j$. When $\hat{p}_+ \approx 1$ (correct), gradients vanish — the loss is already minimised.

---

## Appendix C: Probabilistic Connections in Depth

### C.1 The Exponential Family and NLL

Many distributions belong to the **exponential family**:
$$p(y; \boldsymbol{\eta}) = h(y)\exp(\boldsymbol{\eta}^\top T(y) - A(\boldsymbol{\eta}))$$

where $\boldsymbol{\eta}$ are natural parameters, $T(y)$ are sufficient statistics, and $A(\boldsymbol{\eta})$ is the log-partition function.

**Key property**: The NLL of an exponential family member is:
$$-\log p(y; \boldsymbol{\eta}) = A(\boldsymbol{\eta}) - \boldsymbol{\eta}^\top T(y) - \log h(y)$$

Since $A(\boldsymbol{\eta})$ is always convex, the NLL of any exponential family distribution is **convex in $\boldsymbol{\eta}$** — this is the fundamental reason why logistic regression and Poisson regression are convex problems.

| Distribution | Natural param $\eta$ | Sufficient stat $T(y)$ | NLL loss |
|---|---|---|---|
| Gaussian | $\mu/\sigma^2$ | $y$ | MSE |
| Bernoulli | $\log(p/(1-p))$ | $y$ | BCE |
| Categorical | $\log p_k$ | $\mathbf{1}_{y=k}$ | CE |
| Poisson | $\log\lambda$ | $y$ | $e^\eta - y\eta$ |
| Laplace | $-1/b$ | $|y|$ | MAE |

### C.2 f-Divergences: A General Framework

The **f-divergence** family generalises KL divergence:
$$D_f(p \| q) = \int q(x) f\!\left(\frac{p(x)}{q(x)}\right) dx$$

for a convex function $f$ with $f(1) = 0$:

| $f(t)$ | $D_f$ | Use case |
|---|---|---|
| $t\log t$ | KL divergence $D_{\mathrm{KL}}(p\|q)$ | Variational inference, MLE |
| $-\log t$ | Reverse KL $D_{\mathrm{KL}}(q\|p)$ | Expectation propagation |
| $(t-1)^2$ | $\chi^2$ divergence | Covariate shift correction |
| $\frac{1}{2}\lvert t-1\rvert$ | Total variation | Robust statistics |
| $(\sqrt{t}-1)^2$ | Squared Hellinger | Statistical testing |
| $\frac{1}{2}(t-1)\log t - \frac{1}{2}(t^2-1)$ + ... | Jensen-Shannon | GAN objective |

**GAN Connection**: The original GAN (Goodfellow et al. 2014) minimises the Jensen-Shannon divergence between real and generated distributions — a special f-divergence. WGAN switches to the Wasserstein distance, which is not an f-divergence but a transport distance, making it better-behaved when distributions have non-overlapping support.

### C.3 Proper Scoring Rules: Complete Theory

**Theorem (Characterisation of Proper Scoring Rules, Savage 1971)**. A scoring rule $S(p, y)$ is proper iff there exists a convex function $G: \Delta^{K-1} \to \mathbb{R}$ such that:
$$S(p, y) = G(p) + \nabla G(p)^\top (\mathbf{e}_y - p)$$

where $\mathbf{e}_y$ is the one-hot vector for class $y$.

**Examples**:
- Cross-entropy: $G(p) = -H(p) = \sum_k p_k\log p_k$ (negative entropy, convex)
- Brier score: $G(p) = \lVert p\rVert_2^2 - 1 = \sum_k p_k^2 - 1$ (squared norm, convex)

**Strictly proper**: A proper scoring rule is **strictly proper** if the minimum is achieved only at $q = p$ (not at any other distribution). Cross-entropy and Brier score are both strictly proper.

**Implications for AI training**: Strictly proper scoring rules guarantee that the model cannot "game" the loss by outputting biased confidence estimates. This is why cross-entropy is theoretically preferred over accuracy for classification — accuracy is not proper.

---

## Appendix D: Loss Functions in the LLM Training Pipeline

### D.1 Pretraining Loss

During LLM pretraining, the loss is:
$$\mathcal{L}_{\text{pretrain}} = -\frac{1}{|\mathcal{D}|}\sum_{d \in \mathcal{D}}\sum_{t=1}^{T_d} \log p_{\boldsymbol{\theta}}(x_t^{(d)} \mid x_{<t}^{(d)})$$

Practical considerations:
- **Sequence packing**: multiple documents are concatenated and packed into fixed-length contexts, with causal masking preventing cross-document attention
- **Padding masking**: positions corresponding to padding tokens are excluded from the loss computation
- **Document boundary masking**: in some implementations, attention across document boundaries is masked (caution: affects which gradient signals flow)

### D.2 Instruction Fine-Tuning Loss

For supervised fine-tuning (SFT) on instruction-response pairs:
$$\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{response positions}} \log p_{\boldsymbol{\theta}}(x_t \mid x_{<t})$$

**Only the response tokens** contribute to the loss — prompt tokens are masked. This prevents the model from forgetting how to follow prompts.

### D.3 RLHF Pipeline Losses

**Stage 1: SFT** — standard CE as above.

**Stage 2: Reward Model** — Bradley-Terry cross-entropy on preference pairs.

**Stage 3: RL** — PPO with KL penalty:
$$\mathcal{L}_{\text{PPO+KL}} = -r_\phi(\mathbf{x}, \mathbf{y}) + \beta D_{\mathrm{KL}}(\pi_{\boldsymbol{\theta}} \| \pi_{\text{SFT}})$$

**Combined objective** (InstructGPT):
$$\mathcal{L} = \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \pi_{\boldsymbol{\theta}}}[r_\phi(\mathbf{x}, \mathbf{y})] - \beta D_{\mathrm{KL}}(\pi_{\boldsymbol{\theta}} \| \pi_{\text{SFT}}) + \gamma \mathcal{L}_{\text{pretrain}}$$

The final term prevents **alignment tax** — catastrophic forgetting of pretraining capabilities during RL fine-tuning.

### D.4 Constitutional AI Losses

Anthropic's Constitutional AI (Bai et al. 2022) uses two stages:
1. **Critique and revision (SL-CAI)**: generate critique + revised response pairs; fine-tune on revisions with CE
2. **RLAIF**: use a feedback model (instead of humans) to generate preference labels; train reward model with BCE, then PPO

The key insight: replacing human feedback with AI feedback scales the RLHF process while maintaining the same loss functions.

---

## Appendix E: Connections to Statistical Decision Theory

### E.1 Loss Functions and Risk

Statistical decision theory provides the most general framework for understanding loss functions. A **decision rule** $\delta: \mathcal{X} \to \hat{\mathcal{Y}}$ maps observations to decisions. The **risk** of $\delta$ under loss $\ell$ is:
$$R(\theta, \delta) = \mathbb{E}_{\mathbf{x} \sim P_\theta}[\ell(\delta(\mathbf{x}), \theta)]$$

**Frequentist approach**: Find $\delta^*$ that minimises the worst-case risk (minimax):
$$\delta^*_{\text{minimax}} = \arg\min_\delta \sup_\theta R(\theta, \delta)$$

**Bayesian approach**: Minimise the Bayes risk under prior $\pi(\theta)$:
$$\delta^*_{\text{Bayes}} = \arg\min_\delta \mathbb{E}_{\theta \sim \pi}[R(\theta, \delta)]$$

**Connection to ML**: Empirical risk minimisation is approximately Bayes risk minimisation with a uniform prior and the empirical data distribution as the likelihood.

### E.2 Admissibility and Stein's Phenomenon

A decision rule $\delta_1$ **dominates** $\delta_2$ if $R(\theta, \delta_1) \leq R(\theta, \delta_2)$ for all $\theta$ (with strict inequality for some $\theta$). A rule is **admissible** if no rule dominates it.

**Stein's paradox** (Stein 1956): For estimating the mean of $\mathcal{N}(\boldsymbol{\mu}, I)$ in $d \geq 3$ dimensions under squared error loss, the MLE $\hat{\boldsymbol{\mu}} = \mathbf{x}$ is **inadmissible**. The James-Stein estimator:
$$\hat{\boldsymbol{\mu}}_{\text{JS}} = \left(1 - \frac{d-2}{\lVert\mathbf{x}\rVert^2}\right)\mathbf{x}$$

has strictly lower risk everywhere. This demonstrates that shrinkage (implicit regularisation) is always beneficial in high dimensions — a fact that motivates weight decay in neural networks.

**For AI**: Stein's paradox explains why L2 regularisation always helps in high-dimensional parameter spaces: the unregularised MLE is inadmissible, and the regularised estimator (MAP with Gaussian prior) dominates it under squared error.

### E.3 The Rashomon Set

The **Rashomon set** (Breiman 2001) is the set of models within $\varepsilon$ of the optimal training loss:
$$\mathcal{R}_\varepsilon = \{f \in \mathcal{H} : \hat{R}_n(f) \leq \hat{R}_n(f^*) + \varepsilon\}$$

In modern over-parameterised networks, $\mathcal{R}_\varepsilon$ is enormous — there are many models with near-identical training loss but very different predictions on out-of-distribution data. This has two implications:

1. **Good news**: Implicit regularisation (SGD bias toward flat minima) selects "nice" members of $\mathcal{R}_\varepsilon$
2. **Bad news**: Multiple models from $\mathcal{R}_\varepsilon$ can have very different behaviours on tail distributions, making it hard to guarantee robustness

**For AI**: The Rashomon set perspective explains why fine-tuning a pretrained LLM for instruction following can achieve similar validation loss to the base model while exhibiting very different behaviour: both models are in $\mathcal{R}_\varepsilon$ for the training loss, but they make different trade-offs on the test distribution.


---

## Appendix F: Implementation Patterns

### F.1 PyTorch Loss Implementation Reference

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Regression Losses ──────────────────────────────────────────────────

# MSE — numerically straightforward
mse = nn.MSELoss(reduction='mean')  # or 'sum', 'none'

# MAE
mae = nn.L1Loss(reduction='mean')

# Huber (Smooth L1)
huber = nn.HuberLoss(delta=1.0, reduction='mean')
# Equivalently: nn.SmoothL1Loss(beta=1.0)

# Heteroscedastic Gaussian NLL
gnll = nn.GaussianNLLLoss(full=False, reduction='mean')
# Usage: gnll(pred_mean, target, pred_var)

# Quantile loss (not in PyTorch natively — implement manually)
def quantile_loss(pred, target, tau=0.9):
    residual = target - pred
    return torch.mean(torch.max(tau * residual, (tau - 1) * residual))

# ── Classification Losses ──────────────────────────────────────────────

# Categorical cross-entropy from logits (numerically stable)
ce = nn.CrossEntropyLoss(
    weight=None,           # class weights for imbalance
    label_smoothing=0.1,   # ε for label smoothing
    reduction='mean'
)
# Usage: ce(logits, targets)  — logits are BEFORE softmax

# Binary cross-entropy from logits
bce = nn.BCEWithLogitsLoss(
    pos_weight=None,       # weight for positive class
    reduction='mean'
)
# NEVER use: nn.BCELoss(sigmoid(logits), ...) — numerically unstable

# ── Numerically Stable Patterns ────────────────────────────────────────

# Always prefer log_softmax + nll_loss over softmax + log + nll
log_probs = F.log_softmax(logits, dim=-1)     # numerically stable
loss = F.nll_loss(log_probs, targets)         # same as cross_entropy

# Equivalent one-liner (preferred)
loss = F.cross_entropy(logits, targets)

# ── Contrastive / Metric Losses ────────────────────────────────────────

# InfoNCE / NT-Xent
def info_nce_loss(z_i, z_j, temperature=0.07):
    """
    z_i, z_j: (batch_size, embedding_dim) — two views of same batch
    """
    B = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)   # (2B, D)
    z = F.normalize(z, dim=1)           # unit sphere
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)
    
    # Mask out diagonal (self-similarity)
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))
    
    # Positive pairs: (i, B+i) and (B+i, i)
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(B)]).to(z.device)
    return F.cross_entropy(sim, labels)

# Triplet loss
triplet = nn.TripletMarginLoss(margin=1.0, p=2, reduction='mean')
# Usage: triplet(anchor, positive, negative)

# ── Sequence Losses ────────────────────────────────────────────────────

# LM loss with padding mask
def lm_loss(logits, targets, pad_id=0):
    """
    logits:  (B, T, V)
    targets: (B, T)
    """
    B, T, V = logits.shape
    mask = targets != pad_id  # (B, T)
    loss = F.cross_entropy(
        logits.view(-1, V),
        targets.view(-1),
        reduction='none'
    ).view(B, T)  # (B, T)
    return (loss * mask).sum() / mask.sum()  # average over non-pad tokens
```

### F.2 Common Numerical Issues and Fixes

| Issue | Symptom | Fix |
|---|---|---|
| `log(0)` in BCE | `loss = nan` when prediction is exactly 0 or 1 | Use `BCEWithLogitsLoss`; never pass probabilities to `BCELoss` |
| Softmax overflow | `softmax([1000, 1001]) = [nan, nan]` | Use `log_softmax` + `nll_loss` or `cross_entropy(logits, ...)` |
| Gradient explosion in CE | Large loss on confident wrong predictions | Gradient clip or use label smoothing |
| KL = inf | Model assigns 0 prob where data has positive mass | Clip probabilities: `p.clamp(min=1e-7)` |
| FP16 underflow | Gradients become 0 in half precision | Use dynamic loss scaling or switch to BF16 |
| NaN in triplet loss | Distance = 0 when anchor == positive exactly | Add small epsilon: `torch.norm(...) + 1e-8` |

### F.3 Loss Debugging Checklist

When training loss is NaN or diverges:

1. **Check for NaN inputs**: `assert not torch.isnan(logits).any()`
2. **Check label range**: for CE, labels must be in `[0, num_classes)`
3. **Check gradient norms**: `print(max(p.grad.norm() for p in model.parameters()))`
4. **Verify reduction mode**: sum vs mean affects learning rate scaling
5. **Inspect loss magnitude**: compare to log-uniform-baseline ($\log K$ for $K$ classes)
6. **Check batch normalisation mode**: `model.train()` vs `model.eval()` changes BatchNorm behaviour
7. **Verify padding mask**: unmasked padding inflates sequence loss

---

## Appendix G: Loss Functions and Optimisation Theory

### G.1 Gradient Descent Convergence with Different Losses

**For $L$-smooth, convex loss** with gradient descent ($\eta \leq 1/L$):
$$\mathcal{L}(\boldsymbol{\theta}_T) - \mathcal{L}^* \leq \frac{\lVert\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\rVert^2}{2\eta T}$$

Convergence rate $O(1/T)$.

**For $L$-smooth, $\mu$-strongly convex loss**:
$$\lVert\boldsymbol{\theta}_T - \boldsymbol{\theta}^*\rVert^2 \leq \left(1 - \frac{\mu}{L}\right)^T \lVert\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\rVert^2$$

Linear (exponential) convergence rate. Condition number $\kappa = L/\mu$ determines the rate.

| Loss | Smooth ($L$) | Strongly convex ($\mu$) | Convergence |
|---|---|---|---|
| MSE (linear model) | 2σ²_max | 2σ²_min | Linear ($\kappa = \sigma_{\max}^2/\sigma_{\min}^2$) |
| BCE (logistic) | 1/4 | 0 (unless regularised) | $O(1/T)$ |
| MAE (linear model) | ∞ (not smooth) | 0 | Subgradient: $O(1/\sqrt{T})$ |
| CE (neural network) | Unknown (non-convex) | 0 | No global guarantees |

### G.2 Second-Order Methods and the Fisher Information Matrix

For the NLL loss, the **Fisher information matrix** is:
$$F(\boldsymbol{\theta}) = \mathbb{E}_{(\mathbf{x},y) \sim p_{\boldsymbol{\theta}}}\left[\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(y \mid \mathbf{x})\,\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(y \mid \mathbf{x})^\top\right]$$

Under certain regularity conditions: $F(\boldsymbol{\theta}) = -\mathbb{E}[H_{\mathcal{L}}(\boldsymbol{\theta})]$ — the Fisher equals the expected negative Hessian.

**Natural gradient descent** (Amari 1998): use the Fisher as a preconditioner:
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta F(\boldsymbol{\theta}_t)^{-1} \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$$

Natural gradient descent is invariant to reparameterisation — it converges at the same rate regardless of the parameterisation of the model. Standard gradient descent is not invariant.

**For AI**: K-FAC (Kronecker-Factored Approximate Curvature, Martens & Grosse 2015) approximates the Fisher matrix efficiently for neural networks. It is used in large-scale LLM training to precondition gradients and achieve faster convergence.

### G.3 Saddle-Free Newton Method

For non-convex losses with indefinite Hessians, the standard Newton step $-H^{-1}g$ may **ascend** the loss at saddle points (negative Hessian eigenvalues invert the direction). The **saddle-free Newton method** (Dauphin et al. 2014) replaces $H$ with $|H|$ (absolute value of eigenvalues):
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta |H(\boldsymbol{\theta}_t)|^{-1} \nabla\mathcal{L}(\boldsymbol{\theta}_t)$$

This escapes saddle points by ascending in negative-curvature directions. Practical approximations (CRNLoss, etc.) make this feasible for large models.

---

## Appendix H: Domain-Specific Loss Functions

### H.1 Computer Vision

| Task | Loss | Notes |
|---|---|---|
| Image classification | CE with label smoothing | Standard ResNet, ViT recipes |
| Object detection (boxes) | Smooth L1 / Huber | Coordinates regression |
| Object detection (IoU) | IoU loss, GIoU, DIoU, CIoU | Geometric overlap metrics |
| Semantic segmentation | CE + Dice | Dice handles class imbalance |
| Instance segmentation | Mask CE + box L1 | Mask R-CNN |
| Depth estimation | BerHu loss (reverse Huber) | More weight to large errors |
| Optical flow | Robust loss (Charbonnier) | Handles occlusions |
| Image generation | Adversarial + perceptual + L1 | GAN training + quality terms |
| Diffusion models | Denoising MSE | DDPM simplified loss |

### H.2 Natural Language Processing

| Task | Loss | Notes |
|---|---|---|
| Language modelling | Autoregressive CE | GPT, Llama |
| Masked LM | Masked CE (15% positions) | BERT, RoBERTa |
| Machine translation | Label-smoothed CE | Transformer original |
| Named entity recognition | Token-level CE | Sequence labelling |
| Span extraction (QA) | CE on start + end positions | BERT for QA |
| Text generation (RLHF) | PPO + KL, or DPO | Alignment |
| Retrieval | InfoNCE / DPR loss | Dense passage retrieval |
| Sentence similarity | Cosine MSE or NT-Xent | SBERT |

### H.3 Reinforcement Learning

| Objective | Loss | Notes |
|---|---|---|
| Value function | MSE (TD error) | DQN, SAC critic |
| Policy gradient | -log π × advantage | REINFORCE, A3C |
| Clipped policy | PPO clip loss | Stable policy gradient |
| Q-function | Huber (Smooth L1) | Robust to large TD errors |
| Return distribution | Quantile regression | QR-DQN, IQN |
| World model | MSE on latent predictions | DreamerV3 |


---

## Appendix I: Quick Reference — Key Equations

### I.1 Regression Loss Summary

$$\ell_{\text{MSE}} = r^2, \quad \ell_{\text{MAE}} = |r|, \quad \ell_\delta = \begin{cases}\frac{1}{2}r^2 & |r| \leq \delta \\ \delta|r| - \frac{1}{2}\delta^2 & |r| > \delta \end{cases}$$

$$\ell_\tau = \max(\tau r,\, (\tau-1)r), \quad \ell_{\text{log-cosh}} = \log\cosh(r), \quad r = \hat{y} - y$$

$$\ell_{\text{Gauss-NLL}} = \frac{1}{2}\log(2\pi\sigma^2) + \frac{r^2}{2\sigma^2} = \frac{1}{2}e^{-v}r^2 + \frac{1}{2}v, \quad v = \log\sigma^2$$

### I.2 Classification Loss Summary

$$\ell_{\text{BCE}} = -y\log\sigma(z) - (1-y)\log(1-\sigma(z)), \quad \nabla_z\ell = \sigma(z) - y$$

$$\ell_{\text{CE}} = -\log\hat{p}_{y^*} = -z_{y^*} + \log\sum_k e^{z_k}, \quad \nabla_{\mathbf{z}}\ell = \hat{\mathbf{p}} - \mathbf{y}$$

$$\ell_{\text{hinge}} = \max(0, 1 - y\hat{y}), \quad \ell_{\text{focal}} = -(1-\hat{p}_y)^\gamma\log\hat{p}_y$$

$$\ell_{\text{LS}} = (1-\varepsilon)\ell_{\text{CE}} + \frac{\varepsilon}{K}\sum_k(-\log\hat{p}_k), \quad y^{\text{smooth}}_k = (1-\varepsilon)y_k + \varepsilon/K$$

### I.3 Probabilistic Loss Summary

$$D_{\mathrm{KL}}(p \| q) = \mathbb{E}_p\!\left[\log\frac{p}{q}\right], \quad D_{\mathrm{KL}}(q\|p) \neq D_{\mathrm{KL}}(p\|q)$$

$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{\mathrm{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

$$\ell_{\text{InfoNCE}} = -\log\frac{e^{s^+/\tau}}{e^{s^+/\tau} + \sum_j e^{s_j^-/\tau}}, \quad \nabla_{s^+}\ell = \hat{p}_+ - 1$$

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

$$\mathcal{L}_{\text{LM}} = -\frac{1}{T}\sum_{t=1}^T\log p_\theta(x_t|x_{<t}), \quad \operatorname{PPL} = e^{\mathcal{L}_{\text{LM}}}$$

### I.4 Regularisation Summary

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \frac{\lambda}{2}\lVert\boldsymbol{\theta}\rVert_2^2 \text{ (L2/weight decay)}$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda\lVert\boldsymbol{\theta}\rVert_1 \text{ (L1/Lasso)}$$

$$\mathcal{L}_{\text{MTL}} = \sum_t \frac{1}{2\sigma_t^2}\mathcal{L}_t + \log\sigma_t \text{ (uncertainty weighting)}$$

### I.5 Numerical Stability Identities

$$\log\operatorname{softmax}(\mathbf{z})_k = z_k - m - \log\sum_j e^{z_j - m}, \quad m = \max_j z_j$$

$$\ell_{\text{BCE}}(z, y) = \max(z, 0) - zy + \log(1 + e^{-|z|})$$

$$D_{\mathrm{KL}}(\mathcal{N}(\boldsymbol{\mu}, \operatorname{diag}(\boldsymbol{\sigma}^2)) \| \mathcal{N}(\mathbf{0}, I)) = \frac{1}{2}\sum_j(\sigma_j^2 + \mu_j^2 - 1 - \log\sigma_j^2)$$

---

## Appendix J: Connections Between Loss Functions

```
LOSS FUNCTION FAMILY TREE
════════════════════════════════════════════════════════════════════════

                    Negative Log-Likelihood (NLL)
                    ──────────────────────────────
                    p(y|x; θ) = ?
                        │
          ┌─────────────┼──────────────────┐
          ▼             ▼                  ▼
    Gaussian noise  Bernoulli          Categorical
       p(y) = N      p(y) = Bern        p(y) = Cat
          │             │                  │
          ▼             ▼                  ▼
         MSE           BCE               CE
    (squared loss)  (log-loss)      (cross-entropy)
          │
     Robust version
          │
    ┌─────┴──────┐
    ▼            ▼
   Huber        MAE
  (smooth L1)  (Laplace)

             CE family
             ──────────
                 CE
            ┌────┴────┐
            ▼         ▼
      Label-smooth   Focal
      (regularised)  (reweighted)
            │
            ▼
       Distillation
     (soft targets)

            Contrastive family
            ──────────────────
                 Triplet
                    │
                    ▼
                InfoNCE ───► CLIP
            (temperature τ)
                    │
                    ▼
                  NT-Xent
              (SimCLR variant)

            Preference family
            ─────────────────
              Bradley-Terry
                    │
                 ┌──┴──┐
                 ▼     ▼
                RLHF   DPO
              (reward  (direct
               model)  policy)

════════════════════════════════════════════════════════════════════════
```

This diagram shows that most modern loss functions are specialisations or extensions of a few core ideas: NLL unifies the regression and classification families; proper scoring rules generalise CE; contrastive losses generalise triplet via the softmax structure; preference losses extend the Bradley-Terry model. Understanding this family tree means understanding why each loss was invented and what problem it solves.


---

## Appendix K: Self-Assessment Checklist

After completing this section, verify you can answer these questions without notes:

**Foundations**
- [ ] State the two required properties of any loss function
- [ ] Define Bayes risk and explain why it is irreducible
- [ ] Write the ERM formulation and state the uniform convergence guarantee
- [ ] Explain the trade-off between approximation and estimation error

**Regression Losses**
- [ ] Derive MSE from the Gaussian likelihood assumption
- [ ] Prove that the Bayes optimal predictor under MAE is the conditional median
- [ ] Explain why Huber loss performs implicit gradient clipping
- [ ] State when you would use quantile loss over MSE

**Classification Losses**
- [ ] Derive the gradient $\nabla_\mathbf{z}\ell_{\text{CE}} = \hat{\mathbf{p}} - \mathbf{y}$ from scratch
- [ ] Explain why focal loss solves class imbalance in object detection
- [ ] Describe the effect of label smoothing on logit saturation
- [ ] State the knowledge distillation loss and explain the $\tau^2$ factor

**Probabilistic Losses**
- [ ] Explain forward vs reverse KL and their mode-covering/seeking behaviour
- [ ] Derive the ELBO from Jensen's inequality and state the two terms
- [ ] Write the InfoNCE loss and its gradient with respect to positive similarity
- [ ] Derive the DPO loss from the Bradley-Terry preference model

**Properties**
- [ ] Classify cross-entropy and hinge loss by: convexity, properness, Fisher consistency
- [ ] Define proper scoring rules and give two examples
- [ ] Explain the breakdown point of MSE vs MAE
- [ ] State the condition for L-smooth convergence in gradient descent

**Practical**
- [ ] Write the numerically stable BCE formula in logit space
- [ ] Explain why `log(softmax(z))` is numerically unstable and how to fix it
- [ ] Describe uncertainty weighting for multi-task losses and find the optimal weights
- [ ] Identify the loss function reduction mode and its effect on learning rate scaling

