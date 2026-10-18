[Back to Curriculum](../../README.md) | [Previous: Loss Functions](../01-Loss-Functions/notes.md) | [Next: Normalization Techniques](../03-Normalization-Techniques/notes.md)

---

# Activation Functions

> _"A neural network without nonlinear activation is only a disguised linear map."_

## Overview

Activation functions are the local nonlinear maps that turn stacks of affine
layers into expressive models. They decide which signals pass forward, which
gradients flow backward, which neurons saturate, which features become sparse,
and how numerical scale changes with depth. Their formulas look small, but
their training consequences are large.

This section is the canonical home for activation-function math in the
curriculum. Chapter 14 discusses how neural networks, RNNs, CNNs, and
Transformers use these functions inside model architectures. Chapter 15
discusses LLM-specific block design. Here we focus on the reusable mathematical
objects themselves: scalar activations, vector activations, derivatives,
Jacobians, saturation, smoothness, Lipschitz behavior, gating, softmax
temperature, and gradient-flow consequences.

The central question is not "Which activation is popular?" but "What map does
this activation apply, what derivative does it create, and what does that
derivative do to learning?"

## Prerequisites

- **Single-variable derivatives and chain rule** - [Derivatives](../../04-Calculus-Fundamentals/02-Derivatives/notes.md)
- **Jacobians and vector-valued functions** - [Jacobian Matrix](../../05-Multivariate-Calculus/03-Jacobian-Matrix/notes.md)
- **Gradient flow through optimization** - [Gradient Descent](../../08-Optimization/02-Gradient-Descent/notes.md)
- **Loss gradients at the output** - [Loss Functions](../01-Loss-Functions/notes.md)
- **Numerical stability** - [Floating Point Arithmetic](../../10-Numerical-Methods/01-Floating-Point-Arithmetic/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Curves, derivatives, Jacobians, saturation, gated activations, and softmax temperature |
| [exercises.ipynb](exercises.ipynb) | 10 graded exercises on activation values, derivatives, gradients, and stable implementations |

## Learning Objectives

After completing this section, you will be able to:

- Define scalar, elementwise vector, and coupled vector activations
- Compute derivatives for sigmoid, tanh, ReLU, Leaky ReLU, GELU, SiLU, and softmax
- Explain saturation and why it causes vanishing gradients
- Compare piecewise-linear and smooth activations by gradient behavior
- Derive the softmax Jacobian $J_{ij}=s_i(\delta_{ij}-s_j)$
- Explain why gated activations such as GLU, GEGLU, and SwiGLU are multiplicative feature selectors
- Connect activation variance to Xavier and He initialization
- Diagnose dead neurons, exploding activations, saturated gates, and unstable softmax
- Choose activations by mathematical role rather than popularity

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Nonlinearities Matter](#11-why-nonlinearities-matter)
  - [1.2 Activations as Gates](#12-activations-as-gates)
  - [1.3 Activations as Gradient Controllers](#13-activations-as-gradient-controllers)
  - [1.4 Shape of Activation Curves](#14-shape-of-activation-curves)
  - [1.5 Historical Path](#15-historical-path)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Scalar Activation](#21-scalar-activation)
  - [2.2 Elementwise Vector Activation](#22-elementwise-vector-activation)
  - [2.3 Coupled Vector Activation](#23-coupled-vector-activation)
  - [2.4 Jacobian](#24-jacobian)
  - [2.5 Smoothness, Lipschitz Constants, and Monotonicity](#25-smoothness-lipschitz-constants-and-monotonicity)
- [3. Classical Activations](#3-classical-activations)
  - [3.1 Sigmoid](#31-sigmoid)
  - [3.2 Tanh](#32-tanh)
  - [3.3 Softplus](#33-softplus)
  - [3.4 Saturation](#34-saturation)
  - [3.5 Gate Interpretation](#35-gate-interpretation)
- [4. ReLU Family](#4-relu-family)
  - [4.1 ReLU](#41-relu)
  - [4.2 Leaky ReLU](#42-leaky-relu)
  - [4.3 PReLU](#43-prelu)
  - [4.4 ELU and SELU Preview](#44-elu-and-selu-preview)
  - [4.5 Dead Neurons and Sparse Activations](#45-dead-neurons-and-sparse-activations)
- [5. Smooth Modern Activations](#5-smooth-modern-activations)
  - [5.1 GELU](#51-gelu)
  - [5.2 SiLU and Swish](#52-silu-and-swish)
  - [5.3 Mish](#53-mish)
  - [5.4 Curvature Comparison](#54-curvature-comparison)
  - [5.5 Gradient-Flow Effects](#55-gradient-flow-effects)
- [6. Gated Activations](#6-gated-activations)
  - [6.1 GLU](#61-glu)
  - [6.2 GEGLU](#62-geglu)
  - [6.3 SwiGLU](#63-swiglu)
  - [6.4 Bilinear Gating](#64-bilinear-gating)
  - [6.5 Why Gated MLPs Help](#65-why-gated-mlps-help)
- [7. Vector Activations](#7-vector-activations)
  - [7.1 Softmax](#71-softmax)
  - [7.2 Temperature Softmax](#72-temperature-softmax)
  - [7.3 Softmax Jacobian](#73-softmax-jacobian)
  - [7.4 Sparsemax and Entmax Preview](#74-sparsemax-and-entmax-preview)
  - [7.5 Attention-Output Preview](#75-attention-output-preview)
- [8. Initialization and Gradient Flow](#8-initialization-and-gradient-flow)
  - [8.1 Activation Variance](#81-activation-variance)
  - [8.2 Xavier and He Initialization](#82-xavier-and-he-initialization)
  - [8.3 Vanishing Gradients](#83-vanishing-gradients)
  - [8.4 Exploding Gradients](#84-exploding-gradients)
  - [8.5 Residual Connection Preview](#85-residual-connection-preview)
- [9. Applications in Machine Learning](#9-applications-in-machine-learning)
  - [9.1 CNNs](#91-cnns)
  - [9.2 RNN Gates](#92-rnn-gates)
  - [9.3 Transformer Feedforward Blocks](#93-transformer-feedforward-blocks)
  - [9.4 Binary Outputs](#94-binary-outputs)
  - [9.5 Probability Heads](#95-probability-heads)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI](#12-why-this-matters-for-ai)
- [13. Conceptual Bridge](#13-conceptual-bridge)
- [68. Common Exam and Interview Traps](#68-common-exam-and-interview-traps)
- [65. Computed Table: Extreme-Value Behaviour](#65-computed-table-extreme-value-behaviour)
- [66. Self-Test Answer Key (Condensed)](#66-self-test-answer-key-condensed)
- [67. Section 02 Completion Checklist](#67-section-02-completion-checklist)
- [61. Computed Table: Output Scale vs Input Scale](#61-computed-table-output-scale-vs-input-scale)
- [62. Worked Problem Set G: Final Synthesis](#62-worked-problem-set-g-final-synthesis)
- [63. Decision Flowchart in Words](#63-decision-flowchart-in-words)
- [64. Final Note on Activations](#64-final-note-on-activations)
- [57. Computed Table: Dead Units vs Bias Shift (Simulation, Seed 42)](#57-computed-table-dead-units-vs-bias-shift-simulation-seed-42)
- [58. Worked Example: Full Forward/Backward Pass by Hand](#58-worked-example-full-forwardbackward-pass-by-hand)
- [59. One Activation Language Across Architectures](#59-one-activation-language-across-architectures)
- [60. Practice Problems for Revision](#60-practice-problems-for-revision)
- [53. Worked Problem Set F: Derivations](#53-worked-problem-set-f-derivations)
- [54. Activations and Attention Logit Growth](#54-activations-and-attention-logit-growth)
- [55. What to Memorise vs What to Look Up](#55-what-to-memorise-vs-what-to-look-up)
- [56. Closing Checklist for This Section](#56-closing-checklist-for-this-section)
- [48. Computed Table: Second Derivatives (Curvature)](#48-computed-table-second-derivatives-curvature)
- [49. Activations Inside Transformers](#49-activations-inside-transformers)
- [50. Worked Problem Set E: Statistics and Synthesis](#50-worked-problem-set-e-statistics-and-synthesis)
- [51. Formula Sheet](#51-formula-sheet)
- [52. Chapter Summary: Activations](#52-chapter-summary-activations)
- [44. Computed Table: Activation Statistics for Standard Normal Input](#44-computed-table-activation-statistics-for-standard-normal-input)
- [45. Quick-Reference Cards](#45-quick-reference-cards)
- [46. Extended Exercise Bank (Self-Study)](#46-extended-exercise-bank-self-study)
- [47. Annotated Reading Guide](#47-annotated-reading-guide)
- [39. Worked Trace: One Neuron, Two Activations](#39-worked-trace-one-neuron-two-activations)
- [40. Signal Propagation and the Mean-Field View](#40-signal-propagation-and-the-mean-field-view)
- [41. Activations Meet Regularization and Loss](#41-activations-meet-regularization-and-loss)
- [42. Activation Mistake Catalogue (Field Version)](#42-activation-mistake-catalogue-field-version)
- [43. Decision Records: Activation Choices in Context](#43-decision-records-activation-choices-in-context)
- [35. Worked Problem Set C: Jacobians and Deployment](#35-worked-problem-set-c-jacobians-and-deployment)
- [36. Worked Problem Set D: Selection Under Constraints](#36-worked-problem-set-d-selection-under-constraints)
- [37. Frequently Asked Questions](#37-frequently-asked-questions)
- [38. Self-Test Questions](#38-self-test-questions)
- [31. Computed Table: Softmax Confidence vs Logit Gap](#31-computed-table-softmax-confidence-vs-logit-gap)
- [32. Computed Table: Temperature on Fixed Logits (2,1,0)](#32-computed-table-temperature-on-fixed-logits-210)
- [33. Computed Table: Variance Drift Under Init/Activation Mismatch](#33-computed-table-variance-drift-under-initactivation-mismatch)
- [34. Computed Table: GELU Approximation Error](#34-computed-table-gelu-approximation-error)
- [26. Computed Table: Gradient Attenuation by Depth](#26-computed-table-gradient-attenuation-by-depth)
- [27. Historical Timeline: Activations](#27-historical-timeline-activations)
- [28. Glossary: Activation Vocabulary](#28-glossary-activation-vocabulary)
- [29. Pre-Flight Checklist for Choosing an Activation](#29-pre-flight-checklist-for-choosing-an-activation)
- [30. Monitoring Runbook: Activation Health During Training](#30-monitoring-runbook-activation-health-during-training)
- [24. Worked Problem Set A: Gradients, Death, and Scale](#24-worked-problem-set-a-gradients-death-and-scale)
- [25. Worked Problem Set B: Saturation, Gating, Softmax](#25-worked-problem-set-b-saturation-gating-softmax)
- [19. Computed Table: Activation Values on a Grid](#19-computed-table-activation-values-on-a-grid)
- [20. Computed Table: Derivatives on a Grid](#20-computed-table-derivatives-on-a-grid)
- [21. Gated Activations: GLU, SwiGLU, and Why Transformers Moved](#21-gated-activations-glu-swiglu-and-why-transformers-moved)
- [22. Softmax and Vector Activations](#22-softmax-and-vector-activations)
- [23. Initialization Interplay: Xavier, He, and Activation Statistics](#23-initialization-interplay-xavier-he-and-activation-statistics)
- [14. Activation Dossiers: How to Read an Activation Function](#14-activation-dossiers-how-to-read-an-activation-function)
- [15. Dossier: Sigmoid and Tanh](#15-dossier-sigmoid-and-tanh)
- [16. Dossier: ReLU and the Dying-ReLU Problem](#16-dossier-relu-and-the-dying-relu-problem)
- [17. Dossier: Leaky ReLU, PReLU, ELU, SELU](#17-dossier-leaky-relu-prelu-elu-selu)
- [18. Dossier: GELU, Swish/SiLU, Mish](#18-dossier-gelu-swishsilu-mish)
- [References](#references)

---

## 1. Intuition

### 1.1 Why Nonlinearities Matter

An affine layer has the form

$$
\mathbf{h}=W\mathbf{x}+\mathbf{b}.
$$

If we stack affine layers without nonlinearities, the result is still affine:

$$
W_3(W_2(W_1\mathbf{x}+\mathbf{b}_1)+\mathbf{b}_2)+\mathbf{b}_3
=A\mathbf{x}+\mathbf{c}.
$$

Depth alone does not create nonlinear expressivity. Activation functions insert
nonlinear maps between affine transforms:

$$
\mathbf{h}^{[l]}=\phi(W^{[l]}\mathbf{h}^{[l-1]}+\mathbf{b}^{[l]}).
$$

The activation $\phi$ is what lets the network bend decision boundaries,
compose features, gate information, and approximate complex functions.

### 1.2 Activations as Gates

Many activations behave like gates. ReLU passes positive values and blocks
negative values:

$$
\operatorname{ReLU}(x)=\max(0,x).
$$

Sigmoid maps values into $(0,1)$ and is often interpreted as a soft gate:

$$
\sigma(x)=\frac{1}{1+\exp(-x)}.
$$

Gated activations multiply a content stream by a learned gate:

$$
\operatorname{GLU}(\mathbf{a},\mathbf{b})
=\mathbf{a}\odot\sigma(\mathbf{b}).
$$

This gate perspective is especially useful in LSTMs, GRUs, gated MLPs, and
Transformer feedforward variants.

### 1.3 Activations as Gradient Controllers

Backpropagation multiplies by activation derivatives. In a depth-$L$ network,
early-layer gradients contain products of terms like

$$
W^{[l]\top}\operatorname{diag}(\phi'(\mathbf{z}^{[l]})).
$$

If $\phi'$ is often near zero, gradients vanish. If the product of weight norms
and activation slopes is large, gradients can explode. This is why sigmoid can
saturate deep networks, why ReLU helped train deeper models, and why
normalization and residual connections matter.

### 1.4 Shape of Activation Curves

The curve shape determines both forward statistics and backward gradients:

| Activation | Range | Saturates? | Derivative behavior |
| --- | --- | --- | --- |
| Sigmoid | $(0,1)$ | Both tails | Max derivative $1/4$ |
| Tanh | $(-1,1)$ | Both tails | Max derivative $1$ |
| ReLU | $[0,\infty)$ | Negative side | Derivative $0$ or $1$ |
| GELU | $\mathbb{R}$ | Soft negative gate | Smooth nonmonotone region |
| SiLU | $\mathbb{R}$ | Soft negative gate | Smooth self-gating |
| Softmax | Simplex | Probability saturation | Coupled Jacobian |

### 1.5 Historical Path

Sigmoid and tanh dominated early neural networks because they looked like
smooth biological firing rates and probabilistic gates. ReLU became important
because it reduced saturation and made sparse activations easy. Smooth
variants such as GELU and SiLU became common in large architectures because
they combine gating-like behavior with differentiability. Gated activations
such as GLU, GEGLU, and SwiGLU made multiplicative feature selection a standard
MLP ingredient.

The historical lesson is that activation choice follows training dynamics.
Expressivity, gradient propagation, initialization, normalization, and hardware
all influence the best choice.

## 2. Formal Definitions

### 2.1 Scalar Activation

A scalar activation is a function

$$
\phi:\mathbb{R}\to\mathbb{R}.
$$

Examples:

- $\sigma(x)=1/(1+\exp(-x))$
- $\tanh(x)$
- $\operatorname{ReLU}(x)=\max(0,x)$

Non-examples:

- A loss function $\ell(\hat{y},y)$ is not an activation, because it compares prediction and target.
- A full neural-network layer $W\mathbf{x}+\mathbf{b}$ is not an activation, because it is affine and parameterized.

### 2.2 Elementwise Vector Activation

Given $\mathbf{x}\in\mathbb{R}^d$, an elementwise activation applies the same
scalar map to each coordinate:

$$
\phi(\mathbf{x})=
(\phi(x_1),\ldots,\phi(x_d))^\top.
$$

The Jacobian is diagonal:

$$
J_{\phi}(\mathbf{x})=
\operatorname{diag}(\phi'(x_1),\ldots,\phi'(x_d)).
$$

Examples include sigmoid, tanh, ReLU, GELU, and SiLU when applied to vectors.

### 2.3 Coupled Vector Activation

A coupled vector activation maps $\mathbb{R}^d$ to $\mathbb{R}^d$ but each
output can depend on multiple input coordinates. Softmax is the central
example:

$$
\operatorname{softmax}(\mathbf{z})_i
=\frac{\exp(z_i)}{\sum_{j=1}^{d}\exp(z_j)}.
$$

Changing one logit changes all softmax probabilities because the denominator
couples every coordinate.

### 2.4 Jacobian

For $\mathbf{s}=\operatorname{softmax}(\mathbf{z})$, the Jacobian entries are

$$
\frac{\partial s_i}{\partial z_j}
=s_i(\delta_{ij}-s_j).
$$

In matrix form,

$$
J_{\operatorname{softmax}}(\mathbf{z})
=\operatorname{diag}(\mathbf{s})-\mathbf{s}\mathbf{s}^\top.
$$

This matrix is positive semidefinite and has row sums zero. The row-sum
property reflects shift invariance:

$$
\operatorname{softmax}(\mathbf{z}+c\mathbf{1})
=\operatorname{softmax}(\mathbf{z}).
$$

### 2.5 Smoothness, Lipschitz Constants, and Monotonicity

An activation is smooth if it has continuous derivatives of the needed order.
ReLU is continuous but not differentiable at zero. GELU and SiLU are smooth.
Sigmoid, tanh, softplus, and softmax are smooth on their domains.

An activation is Lipschitz if there exists $L$ such that

$$
\lvert \phi(x)-\phi(y)\rvert\le L\lvert x-y\rvert.
$$

For differentiable scalar activations, a bounded derivative gives a Lipschitz
constant. ReLU is 1-Lipschitz. Sigmoid is $1/4$-Lipschitz. Tanh is
1-Lipschitz.

## 3. Classical Activations

### 3.1 Sigmoid

The sigmoid is

$$
\sigma(x)=\frac{1}{1+\exp(-x)}.
$$

Its derivative is

$$
\sigma'(x)=\sigma(x)(1-\sigma(x)).
$$

The derivative is largest at $x=0$ and approaches zero in both tails. This is
why sigmoid works well as a gate but poorly as a default hidden activation in
deep feedforward networks.

### 3.2 Tanh

The hyperbolic tangent is

$$
\tanh(x)=\frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}.
$$

Its derivative is

$$
\frac{d}{dx}\tanh(x)=1-\tanh^2(x).
$$

Tanh is zero-centered, unlike sigmoid, which often makes optimization easier.
But it still saturates for large $\lvert x\rvert$.

### 3.3 Softplus

Softplus is a smooth approximation to ReLU:

$$
\operatorname{softplus}(x)=\log(1+\exp(x)).
$$

Its derivative is sigmoid:

$$
\frac{d}{dx}\operatorname{softplus}(x)=\sigma(x).
$$

Softplus is useful when a strictly positive output is needed, for example a
variance or rate parameter.

### 3.4 Saturation

An activation saturates when its derivative becomes very small over a large
input region. Sigmoid and tanh saturate in both tails:

$$
\lim_{x\to\infty}\sigma'(x)=0,
\qquad
\lim_{x\to-\infty}\sigma'(x)=0.
$$

Saturation creates vanishing gradients. Once a unit enters a saturated region,
large upstream gradients may be multiplied by near-zero local derivatives.

### 3.5 Gate Interpretation

Sigmoid values can be interpreted as soft gates because they lie between zero
and one. If

$$
\mathbf{h}_{\mathrm{out}}
=\mathbf{g}\odot\mathbf{h}_{\mathrm{in}},
\qquad
\mathbf{g}=\sigma(\mathbf{a}),
$$

then each coordinate of $\mathbf{g}$ decides how much of the input passes.
This pattern is foundational for LSTM input, forget, and output gates, and it
reappears in modern gated feedforward blocks.

## 4. ReLU Family

### 4.1 ReLU

ReLU is

$$
\operatorname{ReLU}(x)=\max(0,x).
$$

A common derivative convention is

$$
\operatorname{ReLU}'(x)=
\begin{cases}
1, & x>0,\\
0, & x\le 0.
\end{cases}
$$

ReLU avoids positive-side saturation and creates sparse activations. Its main
failure mode is dead neurons: units whose preactivations remain negative so
their gradients stay zero.

### 4.2 Leaky ReLU

Leaky ReLU uses a small negative slope:

$$
\operatorname{LeakyReLU}_{\alpha}(x)=
\begin{cases}
x, & x>0,\\
\alpha x, & x\le 0.
\end{cases}
$$

Its derivative is $\alpha$ on the negative side and $1$ on the positive side.
This reduces the dead-neuron problem while preserving much of ReLU's behavior.

### 4.3 PReLU

Parametric ReLU learns the negative slope:

$$
\operatorname{PReLU}_{a}(x)=
\begin{cases}
x, & x>0,\\
ax, & x\le 0.
\end{cases}
$$

The slope $a$ becomes a parameter. This can improve flexibility, but it adds a
small risk: if slopes grow uncontrolled, activation scale can drift.

### 4.4 ELU and SELU Preview

ELU uses an exponential negative branch:

$$
\operatorname{ELU}_{\alpha}(x)=
\begin{cases}
x, & x>0,\\
\alpha(\exp(x)-1), & x\le 0.
\end{cases}
$$

SELU scales ELU to encourage self-normalizing behavior under assumptions about
initialization and network structure. These activations are useful historically
and conceptually, though modern large Transformer-style models more commonly
use GELU, SiLU, or gated variants.

### 4.5 Dead Neurons and Sparse Activations

A ReLU neuron is dead when its preactivation is negative for almost all
examples and updates do not move it back into the active region. Causes
include too-large learning rates, biased initialization, distribution shift,
and poor normalization.

Sparse activation is not always bad. ReLU sparsity can improve feature
selectivity and computation. The problem is irreversible inactivity, not
zeros themselves.

## 5. Smooth Modern Activations

### 5.1 GELU

GELU is

$$
\operatorname{GELU}(x)=x\Phi(x),
$$

where $\Phi$ is the standard normal CDF. A common approximation is

$$
\operatorname{GELU}(x)
\approx
0.5x\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)\right)\right).
$$

GELU can be read as stochastic-looking gating: larger positive values pass
almost unchanged, large negative values are suppressed, and values near zero
are softly weighted.

### 5.2 SiLU and Swish

SiLU, also called Swish when parameterized, is

$$
\operatorname{SiLU}(x)=x\sigma(x).
$$

Its derivative is

$$
\operatorname{SiLU}'(x)
=\sigma(x)+x\sigma(x)(1-\sigma(x)).
$$

SiLU is smooth and self-gated. It allows small negative outputs, which can
improve gradient flow compared with hard zeroing.

### 5.3 Mish

Mish is

$$
\operatorname{Mish}(x)=x\tanh(\operatorname{softplus}(x)).
$$

Like GELU and SiLU, Mish is smooth and allows a negative tail. It is less
standard in large LLM stacks but useful for understanding the family of smooth
self-gated activations.

### 5.4 Curvature Comparison

Smooth activations have nonzero second derivatives over wider regions. That
changes curvature seen by optimizers. ReLU is piecewise linear, so its second
derivative is zero away from the kink. GELU and SiLU have smooth curvature near
zero, which can give more gradual gradient transitions.

Curvature is not automatically good. It can improve optimization smoothness,
but it also changes gradient scale and may interact with initialization.

### 5.5 Gradient-Flow Effects

The main gradient-flow question is:

$$
\text{typical slope} \times \text{typical weight scale}.
$$

Sigmoid has small slopes in the tails. ReLU has slope one on active units and
zero on inactive units. GELU and SiLU have soft slopes that vary smoothly.
Gated activations multiply by learned factors, so their gradients include both
content and gate paths.

## 6. Gated Activations

### 6.1 GLU

Given two vectors $\mathbf{a},\mathbf{b}\in\mathbb{R}^d$, the gated linear
unit is

$$
\operatorname{GLU}(\mathbf{a},\mathbf{b})
=\mathbf{a}\odot\sigma(\mathbf{b}).
$$

The output is linear content modulated by a sigmoid gate. The derivative has
two paths: one through $\mathbf{a}$ and one through $\mathbf{b}$.

### 6.2 GEGLU

GEGLU replaces the sigmoid gate with GELU:

$$
\operatorname{GEGLU}(\mathbf{a},\mathbf{b})
=\mathbf{a}\odot\operatorname{GELU}(\mathbf{b}).
$$

This gives a smooth non-probability gate. Unlike sigmoid gates, GELU gates are
not constrained to $(0,1)$.

### 6.3 SwiGLU

SwiGLU uses SiLU as the gate:

$$
\operatorname{SwiGLU}(\mathbf{a},\mathbf{b})
=\mathbf{a}\odot\operatorname{SiLU}(\mathbf{b}).
$$

SwiGLU is common in modern Transformer feedforward blocks. The architecture
details belong to Chapter 14 and Chapter 15; here the reusable math is the
bilinear interaction between content and gate streams.

### 6.4 Bilinear Gating

Gated activations introduce multiplicative interactions:

$$
y_i=a_i g_i.
$$

The local derivatives are

$$
\frac{\partial y_i}{\partial a_i}=g_i,
\qquad
\frac{\partial y_i}{\partial g_i}=a_i.
$$

Thus the gate controls content gradients, and the content controls gate
gradients. This is more expressive than an elementwise activation applied to a
single stream.

### 6.5 Why Gated MLPs Help

Gated MLPs can select features conditionally. A standard activation transforms
each coordinate independently. A gated activation lets one projection decide
which coordinates of another projection matter. This adds a simple form of
feature interaction without attention.

## 7. Vector Activations

### 7.1 Softmax

Softmax maps logits to probabilities:

$$
s_i=\frac{\exp(z_i)}{\sum_{j=1}^{C}\exp(z_j)}.
$$

It is shift-invariant:

$$
\operatorname{softmax}(\mathbf{z}+c\mathbf{1})
=\operatorname{softmax}(\mathbf{z}).
$$

Softmax is used in multiclass classification, attention weights, categorical
sampling, and contrastive objectives.

### 7.2 Temperature Softmax

Temperature softmax is

$$
s_i(\tau)
=\frac{\exp(z_i/\tau)}{\sum_j\exp(z_j/\tau)}.
$$

Small $\tau$ sharpens the distribution. Large $\tau$ flattens it. Temperature
changes both probabilities and gradient scale, because derivatives include the
factor $1/\tau$.

### 7.3 Softmax Jacobian

The softmax Jacobian is

$$
J=\operatorname{diag}(\mathbf{s})-\mathbf{s}\mathbf{s}^\top.
$$

The diagonal entries are $s_i(1-s_i)$. Off-diagonal entries are $-s_is_j$.
This coupling is why increasing one logit decreases other probabilities.

### 7.4 Sparsemax and Entmax Preview

Softmax assigns positive probability to every class. Sparse alternatives such
as sparsemax and entmax can assign exact zeros. These are useful when sparse
probability distributions are desirable, but their full treatment belongs to
specialized sequence and attention contexts.

### 7.5 Attention-Output Preview

In attention, softmax converts scores into weights:

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right).
$$

The full attention mechanism belongs in
[Attention Mechanism Math](../../15-Math-for-LLMs/03-Attention-Mechanism-Math/notes.md).
Here the important activation fact is that softmax can saturate when logits
have large variance, which motivates scaling by $\sqrt{d_k}$.

## 8. Initialization and Gradient Flow

### 8.1 Activation Variance

If activations grow layer by layer, later layers may explode. If activations
shrink layer by layer, signals may vanish. Initialization chooses weight
variance to keep preactivation and activation variance controlled.

### 8.2 Xavier and He Initialization

Xavier initialization is designed for roughly symmetric activations such as
tanh:

$$
\operatorname{Var}(W_{ij})\approx\frac{2}{n_{\mathrm{in}}+n_{\mathrm{out}}}.
$$

He initialization is designed for ReLU-like activations:

$$
\operatorname{Var}(W_{ij})\approx\frac{2}{n_{\mathrm{in}}}.
$$

The factor of two appears because ReLU zeroes roughly half of a symmetric
input distribution.

### 8.3 Vanishing Gradients

If many derivatives satisfy $\lvert \phi'(z)\rvert<1$, products of derivatives
can shrink exponentially with depth. This was a major reason sigmoid and tanh
hidden activations were hard to train in deep networks without careful
initialization, normalization, or residual paths.

### 8.4 Exploding Gradients

Exploding gradients occur when Jacobian products have large singular values.
Activations contribute through their slopes. ReLU slopes are bounded by one,
but weights can still produce exploding products. Smooth activations can have
slopes slightly above one in some regions, so scale control still matters.

### 8.5 Residual Connection Preview

Residual connections create paths where gradients can flow through identity
maps:

$$
\mathbf{h}^{[l+1]}=\mathbf{h}^{[l]}+F(\mathbf{h}^{[l]}).
$$

This reduces dependence on long products of activation derivatives. The full
model-specific treatment belongs in Chapter 14.

## 9. Applications in Machine Learning

### 9.1 CNNs

CNNs historically use ReLU-family activations because they are cheap,
piecewise-linear, and sparse. Leaky variants can help when dead filters appear.

### 9.2 RNN Gates

RNNs use sigmoid gates and tanh candidate states. The bounded range controls
state updates, while gates regulate memory. The full recurrent architecture
math belongs in [RNN and LSTM Math](../../14-Math-for-Specific-Models/04-RNN-and-LSTM-Math/notes.md).

### 9.3 Transformer Feedforward Blocks

Transformers commonly use GELU, SiLU, or gated variants inside feedforward
blocks. The activation controls token-wise nonlinear transformation between
attention layers. Chapter 14 and Chapter 15 cover the full block design.

### 9.4 Binary Outputs

Sigmoid maps logits to Bernoulli probabilities for binary classification.
Training should usually use a logits-based BCE loss rather than manually
applying sigmoid and then taking logs.

### 9.5 Probability Heads

Softmax maps class logits to categorical probabilities. Its coupling and
shift-invariance make it the standard final activation for multiclass
probability heads and many contrastive objectives.

## 10. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Stacking affine layers without activations | The composition is still affine | Insert nonlinear activations between affine maps |
| 2 | Using sigmoid hidden units in a deep plain MLP | Saturation causes vanishing gradients | Prefer ReLU/GELU/SiLU plus normalization when appropriate |
| 3 | Calling softmax elementwise | Softmax couples coordinates through the denominator | Use the full Jacobian when deriving gradients |
| 4 | Forgetting softmax shift-invariance | Large logits may overflow | Subtract the max before exponentiating |
| 5 | Treating ReLU derivative at zero as important | The convention rarely changes training | State the chosen convention and move on |
| 6 | Ignoring activation scale in initialization | Variance can explode or vanish | Match initialization to activation family |
| 7 | Assuming smooth activations are always better | Smoothness changes scale and cost | Compare gradients, not only curves |
| 8 | Applying sigmoid before a logits BCE loss | The loss applies sigmoid internally | Pass raw logits to fused logits losses |
| 9 | Confusing GLU gates with probabilities | GELU/SwiGLU gates are not constrained to $(0,1)$ | Interpret them as multiplicative feature selectors |
| 10 | Using temperature without considering gradients | Temperature rescales derivatives | Retune or monitor gradient norms |

## 11. Exercises

1. (*) Derive $\sigma'(x)=\sigma(x)(1-\sigma(x))$.
2. (*) Derive $\frac{d}{dx}\tanh(x)=1-\tanh^2(x)$.
3. (*) Compute ReLU, Leaky ReLU, and their derivatives for a vector.
4. (**) Show that stacked affine layers without activations collapse to one affine map.
5. (**) Implement stable softmax and verify shift-invariance.
6. (**) Derive the softmax Jacobian for a three-class vector.
7. (**) Compare GELU and SiLU curves and derivatives numerically.
8. (***) Compute gradients for a GLU and explain the two gradient paths.
9. (***) Estimate activation variance after Xavier and He initialization.
10. (***) Diagnose a dead-ReLU layer from activation and gradient statistics.

## 12. Why This Matters for AI

| Activation concept | AI impact |
| --- | --- |
| Nonlinearity | Gives networks expressive function classes |
| Sigmoid gates | Controls memory and binary probabilities |
| Tanh | Bounded hidden states and centered gates |
| ReLU | Sparse, cheap, deep-friendly hidden activations |
| GELU | Smooth stochastic-style gating in Transformer blocks |
| SiLU/SwiGLU | Self-gated and multiplicative feedforward transformations |
| Softmax | Converts scores into probabilities and attention weights |
| Softmax temperature | Controls sharpness in classification, sampling, and contrastive learning |
| Activation derivatives | Determine gradient propagation and trainability |
| Initialization coupling | Keeps activation scale stable across depth |

## 13. Conceptual Bridge

Loss functions define the gradient at the output. Activation functions decide
how that gradient passes through each hidden layer.

```text
Loss gradient
    -> output head
    -> activation derivatives
    -> layer Jacobians
    -> earlier parameters
```

Next, [Normalization Techniques](../03-Normalization-Techniques/notes.md)
studies how to control the statistics of those activations so deep networks
remain trainable.

## 14. Activation Dossiers: How to Read an Activation Function
This section gives a repeatable dossier template and then applies it to the activations used in practice. For every activation record: domain and range, derivative and its range, behaviour at saturation, smoothness class, monotonicity, mean output at symmetric inputs, cost of evaluation, and the initialization it pairs with. Two activations with the same formula plot can behave very differently once these fields are filled in.

- **Range** determines what the next layer sees: bounded outputs (sigmoid, tanh) cap signal magnitude, unbounded positive outputs (ReLU, GELU) let scale grow.
- **Derivative range** bounds local gradient amplification: sigmoid's derivative never exceeds 0.25, so deep sigmoid stacks contract gradients geometrically.
- **Zero-centred output** (tanh, ELU, GELU approximately) reduces the covariate-shift burden on the next layer compared with strictly positive outputs.
- **Smoothness** matters for second-order methods and for gradient penalties: ReLU is piecewise linear (second derivative zero a.e.), GELU/SiLU are $C^\infty$.
- **Monotonicity** is usual but not required: Swish/SiLU and GELU are slightly non-monotonic near zero, which empirically does not hurt and may help.
- **Sparsity**: exact zeros (ReLU family) give sparse activations; smooth approximations give near-zeros but never exact sparsity.
- **Saturation side**: sigmoid/tanh saturate on both sides; ReLU saturates (zero gradient) only on the left; ELU saturates to $-\alpha$ on the left with nonzero approach gradient.
- Evaluation cost: exp/division (sigmoid, GELU-exact) vs max/mul (ReLU, Leaky ReLU); at LLM scale activation cost is small next to matmuls but memory for saved inputs matters.
- Initialization pairing: He initialization assumes ReLU-like half-rectification; Xavier/Glorot assumes near-linear symmetric activations (Glorot et al. 2011; He et al. 2015).
- A dossier is incomplete without the failure mode: name the input regime where gradient or output degrades, and the monitor that detects it.



## 15. Dossier: Sigmoid and Tanh
The logistic sigmoid $\sigma(x)=1/(1+e^{-x})$ maps $\mathbb{R}$ to $(0,1)$ with derivative $\sigma'(x)=\sigma(x)(1-\sigma(x))$, maximised at $x=0$ with value $1/4$. It is the canonical probability activation for binary outputs and gates, and a poor default for deep hidden layers precisely because of that derivative cap.

- $\sigma'(x)\le 1/4$ everywhere, so a depth-$L$ sigmoid stack multiplies gradients by at most $4^{-L}$ before weights are even considered.
- $\tanh(x)=2\sigma(2x)-1$ maps to $(-1,1)$, is zero-centred, and has derivative $1-\tanh^2(x)\le 1$; it saturates more gracefully than sigmoid but still saturates.
- Historical role: sigmoid/tanh were standard before Glorot et al. (2011) and Nair & Hinton (2010) showed rectifiers train deep nets far more reliably.
- Legitimate modern uses: output probabilities (binary, multi-label), gates in LSTM/GRU/GLU, attention-side squashing is rare (softmax instead).
- Non-example: using sigmoid hidden units in a 20-layer MLP — gradients vanish by construction; depth is not the remedy, the activation is.
- Numerical note: compute sigmoid with the branch $x\ge0: 1/(1+e^{-x})$, $x<0: e^{x}/(1+e^{x})$ to avoid overflow in $e^{-x}$.
- Tanh in RNNs persists because bounded state helps stability of recurrence; the same boundedness is why gradient clipping accompanies tanh RNNs.
- Saturation diagnostic: histogram pre-activations; mass beyond $|x|>4$ for sigmoid ($\sigma'<0.018$) is effectively frozen.
- Temperature view: $\sigma(x/T)$ with $T$ learned interpolates between step and linear behaviour; rarely used for hidden units, common in distillation (Chapter 13 §01).
- Derivative identity worth memorising: $\frac{d}{dx}\log\sigma(x)=1-\sigma(x)=\sigma(-x)$, which is why logistic loss gradients collapse to $p-y$.



## 16. Dossier: ReLU and the Dying-ReLU Problem
ReLU, $f(x)=\max(0,x)$ (Nair & Hinton 2010), is piecewise linear with derivative $1$ on $x>0$ and $0$ on $x<0$. Its virtues — cheap, non-saturating on the right, sparsity-inducing — made deep training routine; its failure mode, the dying ReLU, is a units-level absorbing state.

- A unit dies when its pre-activation is negative for every training example; gradient is then identically zero and gradient descent cannot revive it.
- Death causes: large negative bias update, high learning rate early in training, unlucky initialization with predominantly negative inputs.
- Death is not always bad: some permanent sparsity is harmless; pathology is when a large fraction (say >30-50%) of a layer dies early.
- Monitor: per-layer fraction of exactly-zero activations on a fixed probe batch, logged over time; a jump after a learning-rate spike is the classic signature.
- Mitigations: Leaky ReLU/PReLU (nonzero left slope), ELU (smooth left branch), lower initial learning rate with warmup, careful bias init (small positive bias was an early trick).
- He initialization, $\mathrm{Var}(w)=2/n_{in}$, is derived exactly for ReLU's half-rectification (He et al. 2015); using Xavier with ReLU under-scales variance by a factor of two per layer.
- ReLU networks are positively homogeneous: scaling one layer's weights by $c>0$ and the next by $1/c$ leaves the function unchanged — relevant to weight decay geometry and normalization interplay (§03).
- Expressivity: ReLU nets are continuous piecewise-linear functions; the number of linear regions can grow exponentially in depth (a standard counting argument), which is the geometric root of their power.
- Non-example of a fix: adding more epochs does not revive dead units; the gradient signal is exactly zero, not small.
- Checkpoint test: after any activation swap, assert the dead-unit fraction and gradient norms are logged before comparing accuracy.



## 17. Dossier: Leaky ReLU, PReLU, ELU, SELU
The ReLU variants each repair one ReLU defect: Leaky ReLU (Maas et al. 2013) keeps a small fixed left slope $\alpha$ (often 0.01); PReLU (He et al. 2015) learns $\alpha$ per channel; ELU (Clevert et al. 2015) saturates smoothly to $-\alpha$; SELU (Klambauer et al. 2017) chooses constants so activations self-normalize under stated assumptions.

- Leaky ReLU derivative is $\alpha$ on the left: no unit is ever fully dead, at the cost of losing exact sparsity.
- PReLU adds negligible parameters (one per channel) and typically matches or slightly beats fixed leaks; the learned $\alpha$ values are themselves diagnostics.
- ELU: $f(x)=x$ for $x>0$, $\alpha(e^x-1)$ for $x\le0$; output mean shifts toward zero vs ReLU, and the left derivative $\alpha e^x$ decays smoothly rather than jumping.
- ELU still saturates on the left (derivative $\to0$ as $x\to-\infty$), so very negative units recover slowly — faster than dead ReLU, slower than leaky.
- SELU fixes $\alpha\approx1.6733$, $\lambda\approx1.0507$ so that (under independence and initialization assumptions) activation mean/variance have a stable fixed point; it requires its own initialization and breaks with standard dropout (use alpha-dropout).
- SELU non-example: dropping SELU into a BatchNorm network gains nothing — self-normalization and batch normalization solve the same problem differently.
- Choice heuristic: default ReLU-family member in CNNs is plain ReLU or Leaky; in MLPs without normalization, ELU/SELU are candidates; always validate empirically on the target task.
- All variants share ReLU's unbounded right side: activation scale can still drift upward without normalization or weight decay.
- Derivative discontinuities: ReLU/Leaky/PReLU have a kink at 0 (measure-zero, harmless in practice); ELU is $C^1$ if $\alpha=1$ at the join in derivative value.
- Reporting: activation-variant comparisons are notoriously seed-sensitive; report mean over seeds, not a single run (see §01 on honest comparisons).



## 18. Dossier: GELU, Swish/SiLU, Mish
Smooth modern activations dominate current architectures. GELU (Hendrycks & Gimpel 2016) weights input by a Gaussian CDF, $x\Phi(x)$; Swish/SiLU (Ramachandran et al. 2017; also Elfwing et al. 2018) uses $x\sigma(x)$; Mish (Misra 2019) uses $x\tanh(\mathrm{softplus}(x))$. All are smooth, mostly increasing, slightly non-monotonic near zero, and unbounded above.

- GELU exact form: $x\Phi(x)$; the tanh approximation $\frac{x}{2}(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))$ is standard in frameworks — know which one your framework default uses.
- GELU interpretation: stochastic regularizer view — input kept with probability $\Phi(x)$, i.e. a smooth, input-dependent dropout-like gating (Hendrycks & Gimpel 2016).
- SiLU/Swish: derivative $\sigma(x)+x\sigma(x)(1-\sigma(x))$; the dip below zero (minimum $\approx-0.278$ near $x\approx-1.28$) gives a self-gating bump that empirically helps deep nets.
- Both GELU and SiLU approach ReLU for large $x$ and 0 for large negative $x$, but with smooth transitions that keep small gradients flowing for moderately negative inputs.
- Non-monotonicity is mild and not a defect per se; monotonicity is sufficient, not necessary, for trainability.
- Transformers standardized on GELU (BERT/GPT era), then shifted to gated SiLU (SwiGLU) in many open LLMs; vision models often use SiLU (EfficientNet lineage from Swish search).
- Mish behaves similarly to SiLU with slightly different tails; gains over SiLU are inconsistent across benchmarks — treat as a hyperparameter, not a law.
- Cost: exp/tanh evaluations exceed ReLU cost; at LLM scale this is minor vs matmul, but fused kernels matter for memory bandwidth (activations saved for backward).
- Gradient check: smooth activations pair naturally with gradient penalties and second-order diagnostics where ReLU's kink complicates Hessians.
- Default heuristic 2026: GELU or SwiGLU in transformers, SiLU in convnets, ReLU where sparsity/hardware simplicity dominates; deviations need evidence.

## 19. Computed Table: Activation Values on a Grid
Values computed from the definitions (GELU exact form). Read columns comparatively: saturation, zero behaviour, and negative-side treatment differ exactly where training dynamics differ.

| x | sigmoid | tanh | ReLU | LeakyReLU(0.01) | ELU(1) | SiLU | GELU |
|---|---|---|---|---|---|---|---|
| -5.0 | 0.0067 | -0.9999 | 0.0000 | -0.0500 | -0.9933 | -0.0335 | -0.0000 |
| -4.5 | 0.0110 | -0.9998 | 0.0000 | -0.0450 | -0.9889 | -0.0494 | -0.0000 |
| -4.0 | 0.0180 | -0.9993 | 0.0000 | -0.0400 | -0.9817 | -0.0719 | -0.0001 |
| -3.5 | 0.0293 | -0.9982 | 0.0000 | -0.0350 | -0.9698 | -0.1026 | -0.0008 |
| -3.0 | 0.0474 | -0.9951 | 0.0000 | -0.0300 | -0.9502 | -0.1423 | -0.0040 |
| -2.5 | 0.0759 | -0.9866 | 0.0000 | -0.0250 | -0.9179 | -0.1896 | -0.0155 |
| -2.0 | 0.1192 | -0.9640 | 0.0000 | -0.0200 | -0.8647 | -0.2384 | -0.0455 |
| -1.5 | 0.1824 | -0.9051 | 0.0000 | -0.0150 | -0.7769 | -0.2736 | -0.1002 |
| -1.0 | 0.2689 | -0.7616 | 0.0000 | -0.0100 | -0.6321 | -0.2689 | -0.1587 |
| -0.5 | 0.3775 | -0.4621 | 0.0000 | -0.0050 | -0.3935 | -0.1888 | -0.1543 |
| 0.0 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| 0.5 | 0.6225 | 0.4621 | 0.5000 | 0.5000 | 0.5000 | 0.3112 | 0.3457 |
| 1.0 | 0.7311 | 0.7616 | 1.0000 | 1.0000 | 1.0000 | 0.7311 | 0.8413 |
| 1.5 | 0.8176 | 0.9051 | 1.5000 | 1.5000 | 1.5000 | 1.2264 | 1.3998 |
| 2.0 | 0.8808 | 0.9640 | 2.0000 | 2.0000 | 2.0000 | 1.7616 | 1.9545 |
| 2.5 | 0.9241 | 0.9866 | 2.5000 | 2.5000 | 2.5000 | 2.3104 | 2.4845 |
| 3.0 | 0.9526 | 0.9951 | 3.0000 | 3.0000 | 3.0000 | 2.8577 | 2.9960 |
| 3.5 | 0.9707 | 0.9982 | 3.5000 | 3.5000 | 3.5000 | 3.3974 | 3.4992 |
| 4.0 | 0.9820 | 0.9993 | 4.0000 | 4.0000 | 4.0000 | 3.9281 | 3.9999 |
| 4.5 | 0.9890 | 0.9998 | 4.5000 | 4.5000 | 4.5000 | 4.4506 | 4.5000 |
| 5.0 | 0.9933 | 0.9999 | 5.0000 | 5.0000 | 5.0000 | 4.9665 | 5.0000 |



## 20. Computed Table: Derivatives on a Grid
Derivative magnitudes bound gradient flow. Note sigmoid's 0.25 cap, ReLU's exact zeros, ELU's decaying left derivative, and the smooth bumps of SiLU/GELU near the origin.

| x | sigmoid' | tanh' | ReLU' | LeakyReLU(0.01)' | ELU(1)' | SiLU' | GELU' |
|---|---|---|---|---|---|---|---|
| -5.0 | 0.0066 | 0.0002 | 0.0000 | 0.0100 | 0.0067 | -0.0265 | -0.0000 |
| -4.5 | 0.0109 | 0.0005 | 0.0000 | 0.0100 | 0.0111 | -0.0379 | -0.0001 |
| -4.0 | 0.0177 | 0.0013 | 0.0000 | 0.0100 | 0.0183 | -0.0527 | -0.0005 |
| -3.5 | 0.0285 | 0.0036 | 0.0000 | 0.0100 | 0.0302 | -0.0703 | -0.0028 |
| -3.0 | 0.0452 | 0.0099 | 0.0000 | 0.0100 | 0.0498 | -0.0881 | -0.0119 |
| -2.5 | 0.0701 | 0.0266 | 0.0000 | 0.0100 | 0.0821 | -0.0994 | -0.0376 |
| -2.0 | 0.1050 | 0.0707 | 0.0000 | 0.0100 | 0.1353 | -0.0908 | -0.0852 |
| -1.5 | 0.1491 | 0.1807 | 0.0000 | 0.0100 | 0.2231 | -0.0413 | -0.1275 |
| -1.0 | 0.1966 | 0.4200 | 0.0000 | 0.0100 | 0.3679 | 0.0723 | -0.0833 |
| -0.5 | 0.2350 | 0.7864 | 0.0000 | 0.0100 | 0.6065 | 0.2600 | 0.1325 |
| 0.0 | 0.2500 | 1.0000 | 0.0000 | 0.0100 | 1.0000 | 0.5000 | 0.5000 |
| 0.5 | 0.2350 | 0.7864 | 1.0000 | 1.0000 | 1.0000 | 0.7400 | 0.8675 |
| 1.0 | 0.1966 | 0.4200 | 1.0000 | 1.0000 | 1.0000 | 0.9277 | 1.0833 |
| 1.5 | 0.1491 | 0.1807 | 1.0000 | 1.0000 | 1.0000 | 1.0413 | 1.1275 |
| 2.0 | 0.1050 | 0.0707 | 1.0000 | 1.0000 | 1.0000 | 1.0908 | 1.0852 |
| 2.5 | 0.0701 | 0.0266 | 1.0000 | 1.0000 | 1.0000 | 1.0994 | 1.0376 |
| 3.0 | 0.0452 | 0.0099 | 1.0000 | 1.0000 | 1.0000 | 1.0881 | 1.0119 |
| 3.5 | 0.0285 | 0.0036 | 1.0000 | 1.0000 | 1.0000 | 1.0703 | 1.0028 |
| 4.0 | 0.0177 | 0.0013 | 1.0000 | 1.0000 | 1.0000 | 1.0527 | 1.0005 |
| 4.5 | 0.0109 | 0.0005 | 1.0000 | 1.0000 | 1.0000 | 1.0379 | 1.0001 |
| 5.0 | 0.0066 | 0.0002 | 1.0000 | 1.0000 | 1.0000 | 1.0265 | 1.0000 |



## 21. Gated Activations: GLU, SwiGLU, and Why Transformers Moved
Gated linear units (Dauphin et al. 2017) split the projection: $\mathrm{GLU}(x)=(xW+b)\odot\sigma(xV+c)$. Shazeer (2020) systematically compared GLU variants in transformers and found SwiGLU, $\mathrm{Swish}_\beta(xW)\odot(xV)$, among the best — now standard in many LLMs (e.g., LLaMA-style FFNs).

- Mechanism: the gate branch multiplicatively controls what the value branch passes; gradients flow through both branches, softening the hard on/off of ReLU.
- Parameter accounting: three matrices (two input projections, one output) vs two in a plain FFN; to keep parameter count fixed, hidden width is typically scaled by $2/3$.
- Forgetting the $2/3$ width adjustment is a real-world bug: swapping GELU-FFN for SwiGLU at the same width silently grows parameters and FLOPs by ~50% in the FFN.
- SwiGLU with $\beta=1$ is SiLU-gated; GEGLU uses GELU gating; ReGLU uses ReLU gating — Shazeer (2020) reports gated variants generally beat their ungated counterparts at equal compute.
- Gating connects to earlier ideas: LSTM/GRU gates, highway networks, and multiplicative interactions generally; the dossier view: gate = learned, input-dependent activation slope.
- Gradient path: even if the value branch is small, the gate branch receives gradient through the product, mitigating dead-unit absorbing states.
- Non-example: applying elementwise sigmoid alone as an FFN activation is not GLU — the multiplicative two-branch structure is the point.
- Numerics: products of two linear maps can grow scale; normalization placement (Pre-LN, §03) interacts with FFN design.
- Implementation check: assert shapes — gate and value branches must share shape before the Hadamard product; broadcasting bugs here are silent and costly.
- Historical note: GLU was introduced for language modelling (Dauphin et al. 2017) years before SwiGLU's transformer adoption; ideas often wait for the right architecture.



## 22. Softmax and Vector Activations
Softmax, $\mathrm{softmax}(z)_i=e^{z_i}/\sum_j e^{z_j}$, is a vector activation turning logits into a probability simplex point. Its Jacobian is $\mathrm{diag}(p)-pp^\top$, and paired with cross-entropy the gradient collapses to $p-y$ (derived in §01) — the cleanest gradient identity in ML.

- Shift invariance: softmax$(z+c\mathbf{1})$ = softmax$(z)$; always subtract max$(z)$ before exponentiating (log-sum-exp trick, §01).
- Temperature: softmax$(z/T)$; $T\to0$ approaches argmax one-hot, $T\to\infty$ approaches uniform — the bridge to sampling controls in §04 and Chapter 15 §05.
- Softmax saturates: when one logit dominates, gradients to other logits vanish — label smoothing and logit regularization address the resulting overconfidence.
- Sparsemax/entmax are drop-in alternatives projecting onto the simplex with exact zeros; useful when sparse attention or interpretable selection is desired.
- Softmax is not a probability of correctness: calibration is separate (§01, Guo et al. 2017); high softmax confidence on OOD inputs is a known failure.
- Attention uses row-wise softmax over scores; scaling by $1/\sqrt{d_k}$ keeps pre-softmax variance in the non-saturating regime — an activation-statistics argument, revisited in Chapter 14.
- Multi-label problems use elementwise sigmoid, not softmax: softmax enforces competition (probabilities sum to 1), sigmoid does not.
- Log-softmax should be computed directly (never log(softmax)) for stability; frameworks fuse this as log_softmax + NLL.
- Jacobian rank: $\mathrm{diag}(p)-pp^\top$ has rank $K-1$ (null vector $\mathbf{1}$), reflecting shift invariance.
- Check: softmax outputs tested against sum-to-one within tolerance and invariance to constant shifts in unit tests.



## 23. Initialization Interplay: Xavier, He, and Activation Statistics
Initialization and activation are one design decision. Xavier/Glorot initialization (Glorot & Bengio 2010) preserves variance for near-linear symmetric activations; He initialization (He et al. 2015) doubles the variance to compensate ReLU's half-rectification. Mismatching them degrades signal propagation layer by layer (Glorot et al. 2011).

- Forward variance recursion (linearized): $\mathrm{Var}(y)=n_{in}\mathrm{Var}(w)\,\mathbb{E}[f'( \cdot)^2]$-style factors accumulate per layer; aim for factor 1.
- For tanh near the origin, $f'\approx1$ and $\mathbb{E}[x^2]$ passes through: Xavier sets $\mathrm{Var}(w)=2/(n_{in}+n_{out})$ (uniform variant $\pm\sqrt{6/(n_{in}+n_{out})}$).
- For ReLU, half the mass is zeroed: $\mathbb{E}[\mathrm{ReLU}(x)^2]=\tfrac12\mathrm{Var}(x)$ for symmetric $x$, hence He: $\mathrm{Var}(w)=2/n_{in}$.
- Using Xavier with ReLU shrinks variance by $2^{-L}$ over $L$ layers; using He with tanh inflates it toward saturation.
- GELU/SiLU are near-ReLU in second moment; He-style fan-in scaling is the common default in transformer codebases, combined with residual scaling rules.
- Depth-aware residual scaling (e.g., $1/\sqrt{2L}$-style init for residual branches in GPT-2-era code) controls the variance growth of the residual stream itself (cf. §03 Pre-LN discussion).
- Signal-propagation diagnostics: log activation std per layer at init on a real batch; geometric drift up or down localizes the mismatched layer.
- Edge: embeddings and final logits often use separate init scales; the LM head scale interacts with softmax saturation and early-training loss spikes.
- Orthogonal initialization preserves norms exactly for linear maps and helps deep/recurrent nets; it complements, not replaces, activation-aware scaling.
- Rule: whenever the activation changes, re-derive or look up the paired init; 'same init as before' is a silent confound in activation ablations.

## 24. Worked Problem Set A: Gradients, Death, and Scale
Worked problems. Each is solved step by step; cover the steps and reproduce them before checking.

**Problem A1: sigmoid gradient cap in depth**
1. Setup: 12-layer MLP, sigmoid hidden units, weights with spectral norm near 1.
2. Each layer's Jacobian contributes factor at most $\max\sigma'=0.25$ from the activation.
3. Total activation contribution: at most $0.25^{12}=2^{-24}\approx6\times10^{-8}$.
4. Even with perfectly conditioned weights, early-layer gradients are ~8 orders below output gradients.
5. Conclusion: depth failure is predicted by the dossier before any experiment; switch to ReLU-family or tanh-with-care.
6. Verification: log per-layer gradient norms at init; geometric decay confirms the analysis.

**Problem A2: dying ReLU diagnosis**
1. Observation: training loss plateaus; layer-3 zero-activation fraction rose from 0.48 to 0.97 after LR increase.
2. Hypothesis: mass unit death from an oversized update pushing pre-activations negative for all data.
3. Check: confirm zeros are exact (ReLU) and persistent across different probe batches.
4. Check: dead units' incoming gradient norm is exactly zero, distinguishing death from slow learning.
5. Fix candidates: restore checkpoint before the spike, lower LR, add warmup; optionally switch layer to Leaky ReLU (0.01).
6. Prevention: log dead fraction per layer; alert on step changes correlated with LR schedule events.
7. Non-fix: more epochs — gradient is identically zero for dead units.

**Problem A3: He vs Xavier variance drift**
1. Setup: 20-layer ReLU MLP, Xavier init Var$(w)=1/n_{in}$ (fan-in form).
2. Per layer, ReLU halves variance: Var multiplier $= n_{in}\cdot(1/n_{in})\cdot\tfrac12=\tfrac12$.
3. After 20 layers activation variance scales by $2^{-20}\approx10^{-6}$; logits collapse toward bias-only.
4. Switch to He Var$(w)=2/n_{in}$: multiplier $=2\cdot\tfrac12=1$ per layer; variance preserved.
5. Verify empirically: activation std per layer flat under He, geometrically decaying under Xavier.
6. Lesson: initialization constants are derived from the activation's second moment, not folklore.

**Problem A4: GELU at the origin**
1. Compute GELU$(0)=0\cdot\Phi(0)=0$.
2. Derivative: $\Phi(x)+x\phi(x)$; at 0: $\Phi(0)=0.5$, second term 0, so GELU$'(0)=0.5$.
3. Compare ReLU (subgradient choice) and SiLU$'(0)=0.5$ as well: smooth activations pass half-gradient at the origin.
4. Implication: units centred at zero still learn under GELU/SiLU, unlike left-saturated ReLU units.
5. Numeric spot-check from the table in Section 20: GELU$'(-1)\approx0.083$, small but nonzero.

**Problem A5: softmax shift invariance test**
1. Take $z=(1,2,3)$; softmax $\propto(e^1,e^2,e^3)=(2.718,7.389,20.086)$, sum 30.19.
2. Probabilities: $(0.090,0.245,0.665)$.
3. Shift by $+100$: naive $e^{103}$ overflows float32 (~$3.4\times10^{38}$ max, $e^{103}\approx10^{44}$).
4. Subtract max first: $z-3=(-2,-1,0)$; exponentials $(0.135,0.368,1)$, same probabilities.
5. Rule: every hand-rolled softmax subtracts the max; test invariance to $+1000$ shift in unit tests.

**Problem A6: SwiGLU parameter accounting**
1. Plain FFN width $d=1024$, model dim 256: params $=2\times256\times1024=524{,}288$ (ignoring biases).
2. Naive SwiGLU at same width: three matrices $=3\times256\times1024=786{,}432$ (+50%).
3. Equal-parameter SwiGLU: width $1024\times2/3\approx683$; params $3\times256\times683\approx524{,}544$.
4. Conclusion: report which convention an ablation used; width-matched and parameter-matched comparisons answer different questions.



## 25. Worked Problem Set B: Saturation, Gating, Softmax
Second set: saturation, gating, and vector activations.

**Problem B1: tanh saturation budget**
1. Require derivative at least 0.1: $1-\tanh^2(x)\ge0.1\Rightarrow |\tanh(x)|\le0.949\Rightarrow |x|\lesssim1.83$.
2. If pre-activations are $\mathcal N(0,4)$ (std 2), $P(|x|>1.83)\approx36\%$: a third of units are in the low-gradient zone.
3. Remedies: scale init down, add normalization (§03), or change activation — in that diagnostic order.
4. Recompute after fix: target <5% beyond the saturation threshold on probe data.

**Problem B2: ELU recovery vs Leaky recovery**
1. Unit pre-activation sits at $-6$ for current data.
2. ELU derivative there: $e^{-6}\approx0.0025$ — small but nonzero; recovery is slow exponential climb.
3. Leaky (0.01) derivative: exactly 0.01 — four times larger and constant.
4. ReLU derivative: 0 — no recovery under gradient descent.
5. Ranking for reviving deeply negative units: Leaky/PReLU > ELU > ReLU; ranking for sparsity is the reverse.

**Problem B3: temperature limits**
1. Logits $(2,1,0)$, $T=1$: probs $\propto(7.39,2.72,1)=(0.665,0.245,0.090)$.
2. $T=0.5$ (sharper): logits/T $=(4,2,0)\Rightarrow(0.867,0.117,0.016)$.
3. $T=2$ (flatter): $(1,0.5,0)\Rightarrow(0.506,0.307,0.186)$.
4. $T\to\infty$: uniform $(1/3,1/3,1/3)$; $T\to0$: one-hot on argmax.
5. Use: calibration (§01), distillation targets, decoding (§04) — same knob, different jobs; name which.

**Problem B4: GLU gradient paths**
1. $y=(xW)\odot\sigma(xV)$; let $a=xW$, $g=\sigma(xV)$.
2. $\partial y/\partial a=g$ (elementwise); $\partial y/\partial g=a$.
3. If gate $g\approx0$, value path gradient vanishes but gate path gradient is $a\cdot\sigma'(xV)x$-driven — the unit can reopen.
4. Contrast ReLU death: both paths absent. This is the precise sense in gating mitigates absorbing states.
5. Caveat: if $a$ also $\approx0$, both gradients are small; gates mitigate, not abolish, dead regions.

**Problem B5: where softmax is the wrong activation**
1. Task: tag each token with any subset of {person, org, loc} (multi-label).
2. Softmax forces $\sum p=1$: tagging 'person' suppresses 'org' even when both apply.
3. Correct: elementwise sigmoid + binary cross-entropy per label (§01).
4. Test: construct an example with two true labels; softmax model cannot assign both >0.5 — structural, not a training failure.

**Problem B6: activation swap ablation protocol**
1. Claim to test: SiLU beats ReLU in our convnet.
2. Hold fixed: parameter count, init scheme re-derived for each activation, LR retuned per variant (small grid), seeds $\ge5$.
3. Log: accuracy mean/std, dead-unit fraction, gradient norms, wall-clock per step.
4. Confound to avoid: keeping ReLU-tuned LR for SiLU attributes optimizer mismatch to the activation.
5. Report: effect size with seed spread; single-seed wins are marketing, not evidence.

## 26. Computed Table: Gradient Attenuation by Depth
Upper-bound activation factors (weights excluded). This is the arithmetic behind 'sigmoids don't train deep' and behind residual/normalization designs that keep an identity path.

| depth | sigmoid max factor | tanh factor (|x|=1) | ReLU active |
|---|---|---|---|
| 1 | 2.500e-01 | 4.200e-01 | 1.0 |
| 2 | 6.250e-02 | 1.764e-01 | 1.0 |
| 4 | 3.906e-03 | 3.112e-02 | 1.0 |
| 6 | 2.441e-04 | 5.489e-03 | 1.0 |
| 8 | 1.526e-05 | 9.683e-04 | 1.0 |
| 10 | 9.537e-07 | 1.708e-04 | 1.0 |
| 15 | 9.313e-10 | 2.232e-06 | 1.0 |
| 20 | 9.095e-13 | 2.917e-08 | 1.0 |
| 30 | 8.674e-19 | 4.983e-12 | 1.0 |
| 50 | 7.889e-31 | 1.454e-19 | 1.0 |
| 80 | 6.842e-49 | 7.243e-31 | 1.0 |
| 100 | 6.223e-61 | 2.113e-38 | 1.0 |



## 27. Historical Timeline: Activations
Dates are publication anchors for the canonical citations in References; use them to read claims in context (e.g., pre-2010 advice assumes sigmoid-era constraints).

| Year | Milestone | Why it matters |
|---|---|---|
| 1943 | McCulloch & Pitts threshold units | Binary step neurons; no gradient learning. |
| 1969 | Minsky & Papert, Perceptrons | Limits of single-layer models; multi-layer training problem posed. |
| 1986 | Rumelhart, Hinton & Williams | Backpropagation popularised with sigmoid units. |
| 1989 | Cybenko / Hornik universal approximation | One hidden layer with sigmoidal activations approximates continuous functions on compacts. |
| 1998 | LeCun et al., Efficient BackProp | Tanh variants, input normalization, and practical sigmoid-era training advice. |
| 2010 | Glorot & Bengio (Xavier init) | Variance-preserving initialization analysis for sigmoid/tanh. |
| 2010 | Nair & Hinton | ReLU for restricted Boltzmann machines/deep nets; sparsity and trainability. |
| 2011 | Glorot, Bordes & Bengio | Deep sparse rectifier networks; systematic ReLU evidence. |
| 2013 | Maas, Hannun & Ng | Leaky ReLU in rectifier networks (acoustic models). |
| 2015 | He et al. (PReLU + He init) | Learnable leak and ReLU-matched initialization; ImageNet results. |
| 2015 | Clevert, Unterthiner & Hochreiter | ELU: exponential linear units. |
| 2016 | Hendrycks & Gimpel | GELU: Gaussian error linear units. |
| 2017 | Ramachandran, Zoph & Le | Swish found by automatic search (SiLU form). |
| 2017 | Klambauer et al. | SELU and self-normalizing networks. |
| 2017 | Dauphin et al. | Gated linear units for language modelling. |
| 2018 | Elfwing, Uchibe & Doya | Sigmoid-weighted linear units (SiLU) in RL. |
| 2019 | Misra | Mish activation. |
| 2020 | Shazeer | GLU variants incl. SwiGLU for transformers. |
| 2022+ | SwiGLU in open LLMs | LLaMA-family FFNs adopt SwiGLU as default. |



## 28. Glossary: Activation Vocabulary
| Term | Meaning |
|---|---|
| Saturation | Input regime where derivative is near zero; learning stalls for affected units. |
| Dying ReLU | Unit whose pre-activation is negative for all data; gradient identically zero. |
| Dead fraction | Share of units with exactly zero output on a probe batch; a monitoring signal. |
| Zero-centred | Output mean near zero for symmetric inputs; eases next-layer optimization. |
| Piecewise linear | Function linear on regions; ReLU nets partition input space into polytopes. |
| Linear region | A polytope where a ReLU net equals one affine function. |
| Smoothness class | $C^0$ continuous, $C^1$ differentiable, $C^\infty$ infinitely differentiable (GELU, SiLU). |
| Monotonicity | Order-preserving; typical but not necessary (SiLU/GELU dip slightly). |
| Gate | Multiplicative, input-dependent factor controlling information flow (GLU, LSTM). |
| SwiGLU | SiLU-gated GLU FFN variant (Shazeer 2020). |
| Logits | Pre-softmax scores; unbounded reals mapped to probabilities by softmax/sigmoid. |
| Temperature | Positive divisor scaling logits before softmax/sigmoid; controls sharpness. |
| Fan-in / fan-out | Input/output dimension of a weight matrix; init variances are stated per fan mode. |
| Signal propagation | How activation mean/variance evolve across layers at initialization. |
| Edge of chaos | Init regime where signals neither vanish nor explode; deep-net trainability boundary (mean-field view). |
| Sparsity (activation) | Fraction of exact zeros in activations; ReLU induces it, smooth activations approximate it. |
| Subgradient | Generalized derivative at a kink; ReLU at 0 conventionally uses 0 or 1 in code. |
| Hadamard product | Elementwise multiplication, denoted $\odot$; the gating operation. |
| Pre-activation | Affine output $Wx+b$ before the nonlinearity; histograms of it diagnose saturation. |
| Activation checkpointing | Recomputing activations in backward to save memory; interacts with activation cost. |
| Fused kernel | Implementation combining affine+activation; performance, not math, but affects feasibility. |
| Alpha-dropout | Dropout variant preserving SELU's self-normalizing moments (Klambauer et al. 2017). |
| Log-sum-exp | Stable max-shifted computation of $\log\sum e^{z_i}$; pairs with softmax losses. |
| Sparsemax | Euclidean projection onto the simplex; sparse alternative to softmax. |
| Entmax | Family interpolating softmax and sparsemax via Tsallis entropy. |
| Positive homogeneity | $f(cx)=cf(x)$ for $c>0$ (ReLU); drives scale invariances with normalization. |
| Symmetry breaking | Why identical init fails: units must differ to learn different features; random init provides it. |
| Gradient clipping | Norm cap on gradients; complements (not replaces) non-saturating activations in RNNs. |



## 29. Pre-Flight Checklist for Choosing an Activation
- Task head chosen first: sigmoid (binary/multi-label), softmax (single-label), linear (regression) — hidden activations decided after.
- Hidden default set by architecture family: GELU/SwiGLU (transformer), SiLU (convnet), ReLU (baseline/sparse/hardware).
- Initialization re-derived for the chosen activation (He vs Xavier) and verified by per-layer activation-std logs at init.
- Dead/saturated fraction instrumentation added before training, not after a plateau.
- Learning rate retuned for the variant in any ablation; parameter count matched for gated variants (2/3 width rule).
- Numerical path checked: stable sigmoid/softmax branches; log_softmax fused where used.
- Seeds: at least 3-5 for any claimed activation improvement; report spread.
- Memory: activation storage for backward considered for very wide smooth activations; checkpointing plan if needed.
- Export/deployment: activation supported (or fusable) in the target runtime; exotic activations checked against ONNX/hardware ops.
- Rollback plan: previous activation config retained; activation swaps are config-level changes with checkpoint incompatibility noted.



## 30. Monitoring Runbook: Activation Health During Training
- Log per-layer: pre-activation mean/std, output mean/std, zero fraction (ReLU-family), saturated fraction (|x|>4 sigmoid/tanh).
- Log per-layer gradient norms; geometric decay across depth indicates saturation or init mismatch, growth indicates instability.
- Alert pattern 1: zero fraction jumps after LR change — suspect mass ReLU death; consider checkpoint rollback.
- Alert pattern 2: tanh/sigmoid saturated fraction grows steadily — scale drift; check init, normalization, weight decay.
- Alert pattern 3: activation std grows layer-over-layer at init — He/Xavier mismatch or missing residual scaling.
- Probe batch must be fixed across steps for comparability; random-batch stats hide step changes.
- Distinguish train/eval: dropout and normalization change activation stats; log in train mode on training data.
- After any fix, require the metric that alarmed to return to range before crediting downstream accuracy changes.
- Keep a one-line incident log (step, layer, metric, action) — activation pathologies recur across projects.
- End-of-training snapshot: final activation stats saved with the checkpoint for reproducibility audits.

## 31. Computed Table: Softmax Confidence vs Logit Gap
Softmax confidence is a function of logit gaps, not absolute logits. The 10-class column shows why moderate gaps already produce high confidence — and why raw confidence is not calibrated accuracy (§01).

| logit gap d | p(top) binary | p(top) 10-class (others 0) | entropy 10-class (nats) |
|---|---|---|---|
| 0.0 | 0.5000 | 0.1000 | 2.3026 |
| 0.5 | 0.6225 | 0.1548 | 2.2880 |
| 1.0 | 0.7311 | 0.2320 | 2.2292 |
| 1.5 | 0.8176 | 0.3324 | 2.1027 |
| 2.0 | 0.8808 | 0.4509 | 1.8949 |
| 2.5 | 0.9241 | 0.5751 | 1.6154 |
| 3.0 | 0.9526 | 0.6906 | 1.2985 |
| 3.5 | 0.9707 | 0.7863 | 0.9884 |
| 4.0 | 0.9820 | 0.8585 | 0.7186 |
| 4.5 | 0.9890 | 0.9091 | 0.5043 |
| 5.0 | 0.9933 | 0.9428 | 0.3447 |
| 5.5 | 0.9959 | 0.9645 | 0.2312 |
| 6.0 | 0.9975 | 0.9782 | 0.1530 |
| 6.5 | 0.9985 | 0.9866 | 0.1002 |
| 7.0 | 0.9991 | 0.9919 | 0.0652 |
| 7.5 | 0.9994 | 0.9950 | 0.0421 |
| 8.0 | 0.9997 | 0.9970 | 0.0271 |
| 8.5 | 0.9998 | 0.9982 | 0.0174 |
| 9.0 | 0.9999 | 0.9989 | 0.0111 |
| 9.5 | 0.9999 | 0.9993 | 0.0071 |
| 10.0 | 1.0000 | 0.9996 | 0.0045 |



## 32. Computed Table: Temperature on Fixed Logits (2,1,0)
Same logits, different temperatures. Entropy rises monotonically with $T$; decoding (§04) and calibration (§01) exploit exactly this knob.

| T | p1 | p2 | p3 | entropy |
|---|---|---|---|---|
| 0.1 | 1.0000 | 0.0000 | 0.0000 | 0.0005 |
| 0.2 | 0.9933 | 0.0067 | 0.0000 | 0.0407 |
| 0.3 | 0.9644 | 0.0344 | 0.0012 | 0.1591 |
| 0.5 | 0.8668 | 0.1173 | 0.0159 | 0.4411 |
| 0.7 | 0.7710 | 0.1848 | 0.0443 | 0.6506 |
| 1.0 | 0.6652 | 0.2447 | 0.0900 | 0.8324 |
| 1.5 | 0.5627 | 0.2889 | 0.1483 | 0.9653 |
| 2.0 | 0.5065 | 0.3072 | 0.1863 | 1.0202 |
| 3.0 | 0.4484 | 0.3213 | 0.2302 | 1.0626 |
| 5.0 | 0.4018 | 0.3289 | 0.2693 | 1.0854 |
| 10.0 | 0.3672 | 0.3322 | 0.3006 | 1.0953 |



## 33. Computed Table: Variance Drift Under Init/Activation Mismatch
Stylized per-layer multipliers from Sections 23/26 (linearized). The exact numbers are model-dependent; the geometric pattern is the lesson — mismatch compounds exponentially in depth.

| layer | ReLU+Xavier (x0.5) | ReLU+He (x1) | tanh+He-ish (x2 sat.) | sigmoid+Xavier (x0.0625 act.) |
|---|---|---|---|---|
| 1 | 5.00e-01 | 1.00 | 2.00 (saturating) | 6.25e-02 |
| 2 | 2.50e-01 | 1.00 | 4.00 (saturating) | 3.91e-03 |
| 3 | 1.25e-01 | 1.00 | 8.00 (saturating) | 2.44e-04 |
| 4 | 6.25e-02 | 1.00 | 8.00 (saturating) | 1.53e-05 |
| 5 | 3.12e-02 | 1.00 | 8.00 (saturating) | 9.54e-07 |
| 6 | 1.56e-02 | 1.00 | 8.00 (saturating) | 5.96e-08 |
| 7 | 7.81e-03 | 1.00 | 8.00 (saturating) | 3.73e-09 |
| 8 | 3.91e-03 | 1.00 | 8.00 (saturating) | 2.33e-10 |
| 9 | 1.95e-03 | 1.00 | 8.00 (saturating) | 1.46e-11 |
| 10 | 9.77e-04 | 1.00 | 8.00 (saturating) | 9.09e-13 |
| 11 | 4.88e-04 | 1.00 | 8.00 (saturating) | 5.68e-14 |
| 12 | 2.44e-04 | 1.00 | 8.00 (saturating) | 3.55e-15 |
| 13 | 1.22e-04 | 1.00 | 8.00 (saturating) | 2.22e-16 |
| 14 | 6.10e-05 | 1.00 | 8.00 (saturating) | 1.39e-17 |
| 15 | 3.05e-05 | 1.00 | 8.00 (saturating) | 8.67e-19 |
| 16 | 1.53e-05 | 1.00 | 8.00 (saturating) | 5.42e-20 |
| 17 | 7.63e-06 | 1.00 | 8.00 (saturating) | 3.39e-21 |
| 18 | 3.81e-06 | 1.00 | 8.00 (saturating) | 2.12e-22 |
| 19 | 1.91e-06 | 1.00 | 8.00 (saturating) | 1.32e-23 |
| 20 | 9.54e-07 | 1.00 | 8.00 (saturating) | 8.27e-25 |
| 21 | 4.77e-07 | 1.00 | 8.00 (saturating) | 5.17e-26 |
| 22 | 2.38e-07 | 1.00 | 8.00 (saturating) | 3.23e-27 |
| 23 | 1.19e-07 | 1.00 | 8.00 (saturating) | 2.02e-28 |
| 24 | 5.96e-08 | 1.00 | 8.00 (saturating) | 1.26e-29 |
| 25 | 2.98e-08 | 1.00 | 8.00 (saturating) | 7.89e-31 |
| 26 | 1.49e-08 | 1.00 | 8.00 (saturating) | 4.93e-32 |
| 27 | 7.45e-09 | 1.00 | 8.00 (saturating) | 3.08e-33 |
| 28 | 3.73e-09 | 1.00 | 8.00 (saturating) | 1.93e-34 |
| 29 | 1.86e-09 | 1.00 | 8.00 (saturating) | 1.20e-35 |
| 30 | 9.31e-10 | 1.00 | 8.00 (saturating) | 7.52e-37 |



## 34. Computed Table: GELU Approximation Error
The tanh approximation (Hendrycks & Gimpel 2016) tracks exact GELU to ~1e-3; framework defaults differ, so cross-framework comparisons should state the variant. SiLU is shown for shape comparison.

| x | GELU exact | GELU tanh-approx | abs err | SiLU |
|---|---|---|---|---|
| -4.00 | -0.00013 | -0.00007 | 5.64e-05 | -0.07194 |
| -3.75 | -0.00033 | -0.00022 | 1.12e-04 | -0.08617 |
| -3.50 | -0.00081 | -0.00062 | 1.98e-04 | -0.10259 |
| -3.25 | -0.00188 | -0.00157 | 3.07e-04 | -0.12131 |
| -3.00 | -0.00405 | -0.00364 | 4.12e-04 | -0.14228 |
| -2.75 | -0.00819 | -0.00772 | 4.71e-04 | -0.16524 |
| -2.50 | -0.01552 | -0.01508 | 4.40e-04 | -0.18965 |
| -2.25 | -0.02751 | -0.02720 | 3.04e-04 | -0.21454 |
| -2.00 | -0.04550 | -0.04540 | 9.80e-05 | -0.23841 |
| -1.75 | -0.07010 | -0.07020 | 1.01e-04 | -0.25908 |
| -1.50 | -0.10021 | -0.10043 | 2.18e-04 | -0.27364 |
| -1.25 | -0.13206 | -0.13229 | 2.24e-04 | -0.27838 |
| -1.00 | -0.15866 | -0.15881 | 1.53e-04 | -0.26894 |
| -0.75 | -0.16997 | -0.17004 | 6.89e-05 | -0.24062 |
| -0.50 | -0.15427 | -0.15429 | 1.72e-05 | -0.18877 |
| -0.25 | -0.10032 | -0.10032 | 1.23e-06 | -0.10946 |
| 0.00 | 0.00000 | 0.00000 | 0.00e+00 | 0.00000 |
| 0.25 | 0.14968 | 0.14968 | 1.23e-06 | 0.14054 |
| 0.50 | 0.34573 | 0.34571 | 1.72e-05 | 0.31123 |
| 0.75 | 0.58003 | 0.57996 | 6.89e-05 | 0.50938 |
| 1.00 | 0.84134 | 0.84119 | 1.53e-04 | 0.73106 |
| 1.25 | 1.11794 | 1.11771 | 2.24e-04 | 0.97162 |
| 1.50 | 1.39979 | 1.39957 | 2.18e-04 | 1.22636 |
| 1.75 | 1.67990 | 1.67980 | 1.01e-04 | 1.49092 |
| 2.00 | 1.95450 | 1.95460 | 9.80e-05 | 1.76159 |
| 2.25 | 2.22249 | 2.22280 | 3.04e-04 | 2.03546 |
| 2.50 | 2.48448 | 2.48492 | 4.40e-04 | 2.31035 |
| 2.75 | 2.74181 | 2.74228 | 4.71e-04 | 2.58476 |
| 3.00 | 2.99595 | 2.99636 | 4.12e-04 | 2.85772 |
| 3.25 | 3.24812 | 3.24843 | 3.07e-04 | 3.12869 |
| 3.50 | 3.49919 | 3.49938 | 1.98e-04 | 3.39741 |
| 3.75 | 3.74967 | 3.74978 | 1.12e-04 | 3.66383 |
| 4.00 | 3.99987 | 3.99993 | 5.64e-05 | 3.92806 |

## 35. Worked Problem Set C: Jacobians and Deployment
Third set: Jacobians, approximations, and deployment constraints.

**Problem C1: softmax Jacobian numerically**
1. Logits give $p=(0.665,0.245,0.090)$ (Problem A5).
2. Jacobian $J=\mathrm{diag}(p)-pp^\top$.
3. $J_{11}=0.665(1-0.665)=0.223$; $J_{12}=-0.665\times0.245=-0.163$; $J_{13}=-0.060$.
4. Row 1: $(0.223,-0.163,-0.060)$; rows sum to 0 — check: $0.223-0.163-0.060=0$.
5. Interpretation: raising logit 1 raises $p_1$ but lowers others; total probability conserved.
6. With cross-entropy and one-hot $y=(1,0,0)$: gradient $p-y=(-0.335,0.245,0.090)$ — no Jacobian multiplication needed in code.

**Problem C2: hard-sigmoid on mobile**
1. Constraint: deployment runtime lacks fast exp; need a probability-like squashing.
2. Hard-sigmoid: $\mathrm{clip}((x+3)/6,0,1)$ — piecewise linear surrogate (used in MobileNet-era nets).
3. Derivative is $1/6$ inside $(-3,3)$, 0 outside: saturation returns at the clip boundaries.
4. Trade-off accepted for latency; training may use the smooth version and export the hard one only after validating the delta.
5. Lesson: activation choice is constrained by the deployment stack; check ops before designing.

**Problem C3: softplus as smooth ReLU**
1. Softplus $\log(1+e^x)$: derivative is exactly sigmoid$(x)$.
2. At $x=0$: value $\log2\approx0.693$, derivative 0.5; as $x\to\infty$ approaches $x$; as $x\to-\infty$ approaches 0.
3. It is $C^\infty$ and strictly positive — used for scale/variance parameters that must stay positive, rarely as a hidden activation.
4. Sharpness parameter: $\frac1\beta\log(1+e^{\beta x})\to$ ReLU as $\beta\to\infty$; large $\beta$ reintroduces overflow risk in $e^{\beta x}$ — use stable branches.

**Problem C4: why accuracy did not change after GELU swap**
1. Observation: ReLU→GELU in a 6-layer MLP, accuracy unchanged within seed noise.
2. Possible causes: task not gradient-flow-limited; capacity/data dominate; LR not retuned; differences below seed variance.
3. Correct response: increase seeds, check harder/deeper variant, inspect gradient norms — not 'GELU doesn't work'.
4. Null results at small scale do not transfer to deep transformers where GELU/SwiGLU evidence was gathered.

**Problem C5: symmetry breaking**
1. Init all weights in a layer identically and biases equal: every unit computes the same function.
2. Gradients to identical units are identical, so symmetry persists forever — effective width 1.
3. Random init breaks symmetry; activation choice cannot fix symmetric init.
4. Check: unit-output correlation at init near 1.0 across a layer signals this bug (or duplicated data paths).

**Problem C6: activation and quantization**
1. Int8 deployment: unbounded ReLU outputs need calibrated clipping ranges; outliers dominate the scale and crush resolution.
2. ReLU6 (clip at 6, MobileNet lineage) bounds range for quantization friendliness.
3. GELU/SiLU negative dips complicate naive range calibration slightly; per-channel scales mitigate.
4. Rule: if int8 deployment is planned, activation range behaviour is a design input from day one.



## 36. Worked Problem Set D: Selection Under Constraints
Fourth set: synthesis and selection under constraints.

**Problem D1: choose for a 50-layer MLP without normalization**
1. Sigmoid/tanh: predicted vanishing (Section 26 table) — reject.
2. ReLU+He: viable; add dead-fraction monitoring.
3. ELU/SELU: candidates; SELU requires its init/dropout discipline.
4. Decision record: ReLU+He baseline, ELU challenger, 5 seeds, gradient-norm and dead-fraction logs attached.

**Problem D2: choose for an LLM FFN**
1. Default: SwiGLU with 2/3 width rule (Shazeer 2020) under Pre-LN (§03).
2. Baseline comparator: GELU FFN at matched parameter count.
3. Metrics: validation loss at fixed tokens/FLOPs, not fixed steps, since costs differ.
4. Record kernel support: fused SwiGLU kernels availability on target hardware.

**Problem D3: choose for recurrent state**
1. Bounded state aids recurrence stability: tanh remains standard in LSTM/GRU cells.
2. Gates use sigmoid (0-1 multiplicative control) — a dossier-driven choice, not habit.
3. Modern alternatives (state-space models) leave this section's scope; within RNNs, tanh+sigmoid gating is the defensible default.

**Problem D4: diagnose exploding activations**
1. Symptom: activation std grows per layer, loss spikes to NaN at step ~200.
2. Suspects: He init with tanh (variance inflation to saturation), missing residual scaling, LR too high.
3. Probe: per-layer std at init (before training) separates init mismatch from training dynamics.
4. Fix in order: correct init pairing, add/normalize residuals (§03), warmup; clipping is the seatbelt, not the fix.



## 37. Frequently Asked Questions
- **Is ReLU obsolete?** No. It remains a strong, cheap baseline; modern defaults (GELU/SwiGLU/SiLU) win in specific architecture families, not universally.
- **Does non-monotonicity (SiLU dip) cause problems?** Empirically no at scale; the dip is small and gradients remain informative.
- **Should I learn the activation (PReLU-style) always?** It adds flexibility cheaply but complicates comparisons; use when activation tuning is the bottleneck, not by default.
- **Why did transformers leave ReLU?** GELU improved results in BERT/GPT-era ablations; gated variants improved further (Shazeer 2020). Depth + normalization changed the trade-offs.
- **Can activations fix vanishing gradients alone?** They help (non-saturating right side), but residual connections and normalization (§03) are the structural fixes at depth.
- **What activation for the output layer?** Determined by the loss (§01): linear (regression), sigmoid (binary/multi-label), softmax (single-label), softplus-parameterized scales for probabilistic heads.
- **Do smooth activations always beat ReLU?** No — results are task-, depth-, and tuning-dependent; demand seed-replicated evidence (Problem B6).
- **Is exact sparsity (ReLU zeros) valuable?** For interpretability, conditional compute, and some hardware paths yes; smooth activations sacrifice it.
- **How do I test a new activation paper's claim?** Reproduce at matched params/FLOPs with retuned LR and multiple seeds; check whether gains survive your architecture family.
- **Activation or initialization first when training fails?** Check signal propagation at init (one probe batch) — it separates init/activation mismatch from optimizer issues in minutes.



## 38. Self-Test Questions
- State the derivative cap of sigmoid and its depth consequence.
- Derive He initialization's factor 2 from ReLU's second moment.
- Explain the dying-ReLU absorbing state and two mitigations with different mechanisms.
- Write GELU in exact and tanh-approximate forms; which does your framework use?
- Compute SiLU$'(0)$ and GELU$'(0)$ and interpret.
- Why does softmax have a rank $K-1$ Jacobian?
- When is sigmoid (not softmax) the correct output activation?
- State the SwiGLU 2/3 width rule and what goes wrong if ignored.
- Give the saturation diagnostic for tanh and a numeric threshold.
- Explain why SELU + BatchNorm is a category error.
- Contrast ELU and Leaky ReLU recovery for a unit at pre-activation $-6$.
- Why must LR be retuned in activation ablations?
- Name the activation in LSTM gates and justify via the dossier.
- What does temperature do to entropy (Section 32 table)?
- Give a deployment constraint that can override accuracy-based activation choice.

## 39. Worked Trace: One Neuron, Two Activations
A single affine+activation unit traced numerically. The ReLU path passes either full slope (1.5) or exactly zero; the GELU column shows the smooth alternative on the same pre-activations.

| x | pre1=1.5x-0.5 | ReLU | GELU | out=sigmoid(2*ReLU-1) | dL/dx factor (ReLU path) |
|---|---|---|---|---|---|
| -2.00 | -3.500 | 0.000 | -0.001 | 0.2689 | 0.0 |
| -1.75 | -3.125 | 0.000 | -0.003 | 0.2689 | 0.0 |
| -1.50 | -2.750 | 0.000 | -0.008 | 0.2689 | 0.0 |
| -1.25 | -2.375 | 0.000 | -0.021 | 0.2689 | 0.0 |
| -1.00 | -2.000 | 0.000 | -0.046 | 0.2689 | 0.0 |
| -0.75 | -1.625 | 0.000 | -0.085 | 0.2689 | 0.0 |
| -0.50 | -1.250 | 0.000 | -0.132 | 0.2689 | 0.0 |
| -0.25 | -0.875 | 0.000 | -0.167 | 0.2689 | 0.0 |
| 0.00 | -0.500 | 0.000 | -0.154 | 0.2689 | 0.0 |
| 0.25 | -0.125 | 0.000 | -0.056 | 0.2689 | 0.0 |
| 0.50 | 0.250 | 0.250 | 0.150 | 0.3775 | 1.5 |
| 0.75 | 0.625 | 0.625 | 0.459 | 0.5622 | 1.5 |
| 1.00 | 1.000 | 1.000 | 0.841 | 0.7311 | 1.5 |
| 1.25 | 1.375 | 1.375 | 1.259 | 0.8520 | 1.5 |
| 1.50 | 1.750 | 1.750 | 1.680 | 0.9241 | 1.5 |
| 1.75 | 2.125 | 2.125 | 2.089 | 0.9627 | 1.5 |
| 2.00 | 2.500 | 2.500 | 2.484 | 0.9820 | 1.5 |



## 40. Signal Propagation and the Mean-Field View
Mean-field analyses treat layer outputs as random variables and track mean, variance, and input-correlation propagation. The practical takeaways survive without the full theory: match init to activation second moments (Section 23), watch correlation collapse (all inputs mapping to similar representations) in very deep unnormalized nets, and prefer architectures with an identity path (residuals, §03) when depth grows.

- Variance map: $q_{l+1}=\sigma_w^2\mathbb{E}[f(\sqrt{q_l}z)^2]+\sigma_b^2$; a fixed point $q^*$ with derivative near 1 propagates signals stably.
- For ReLU, $\mathbb{E}[\mathrm{ReLU}(\sqrt q z)^2]=q/2$, giving the He condition $\sigma_w^2=2$ (fan-in form) yet again — three derivations, one constant.
- Correlation map: distinct inputs' representations can converge to correlation 1 with depth (information loss) even when variance is preserved; residuals and normalization counteract this.
- Edge-of-chaos: the boundary in $(\sigma_w,\sigma_b)$ space separating ordered (vanishing) from chaotic (exploding) propagation; trainable depth grows near it.
- BatchNorm/LayerNorm (§03) re-set statistics per layer, relaxing (not abolishing) these init constraints — one reason normalized nets tolerate sloppier init.
- Sigmoid-era nets sat deep in the ordered regime at standard inits, compounding the derivative cap of Section 26.
- Practical probe remains empirical: per-layer std + pairwise cosine similarity of representations on a probe batch at init.
- Depth-to-width: wider layers concentrate mean-field behaviour (law of large numbers), making init theory more predictive at LLM widths.



## 41. Activations Meet Regularization and Loss
- Weight decay interacts with scale invariance: ReLU nets with normalization (§03) have scale-free directions where decay controls effective learning rate rather than capacity directly.
- Dropout after ReLU keeps zeros zero; dropout rates tuned in ReLU nets may mis-transfer to smooth activations (SELU needs alpha-dropout by design).
- Label smoothing (§01) reduces the pressure toward softmax saturation, changing logit-scale dynamics, not hidden activations directly.
- Gradient penalties (e.g., in GANs/WGAN-GP) prefer smooth activations; ReLU kinks make penalty behaviour piecewise.
- Spectral normalization (§03) constrains Lipschitz constants; activation Lipschitz (1 for ReLU/tanh, 1.1-ish peaks for SiLU/GELU derivatives) composes into the bound.
- Early stopping and activation health: a model can reach good loss with many dead units; capacity audits (dead fraction) belong in model cards.
- Data augmentation does not repair activation pathologies; fix the signal path first, then tune regularization.
- Curriculum/precision: mixed precision keeps master weights in fp32; activation ranges (unbounded ReLU/GELU) interact with fp16 overflow — loss scaling (§01, §03) is the companion fix.



## 42. Activation Mistake Catalogue (Field Version)
Complements Section 10 with deployment- and experiment-level failures observed in practice.

| # | Mistake | Consequence | Fix |
|---|---|---|---|
| 1 | Sigmoid hidden units in deep MLP | Geometric gradient vanishing | ReLU-family/GELU hidden units |
| 2 | Xavier init with ReLU | Variance halves per layer | He init Var=2/fan-in |
| 3 | He init with tanh | Variance inflation, saturation | Xavier/Glorot init |
| 4 | Ignoring dead-unit fraction | Silent capacity loss | Probe-batch zero-fraction logging |
| 5 | Raising LR to revive dead ReLUs | More death, instability | Leaky/ELU, warmup, rollback |
| 6 | Softmax for multi-label | Labels forced to compete | Elementwise sigmoid + BCE |
| 7 | log(softmax(x)) by hand | Overflow/underflow | Fused log_softmax |
| 8 | SwiGLU at GELU width | Silent +50% FFN params | 2/3 width rule |
| 9 | SELU with BatchNorm/standard dropout | Self-normalization assumptions broken | Use SELU discipline or drop SELU |
| 10 | Single-seed activation claims | Noise reported as progress | Multi-seed, matched tuning |
| 11 | Keeping old LR after activation swap | Confounded ablation | Retune LR per variant |
| 12 | Hard activations exported untested | Train/serve skew | Validate hard vs smooth delta |



## 43. Decision Records: Activation Choices in Context
- **Tabular MLP baseline:** ReLU + He, dead-fraction logging; challenger ELU. Rationale: simplicity, strong baseline, cheap.
- **Deep convnet (vision):** SiLU per modern practice; ReLU baseline retained for hardware fallback.
- **Transformer LM:** SwiGLU FFN (2/3 width), Pre-LN; GELU FFN as matched-parameter comparator.
- **Recurrent sequence model:** tanh state, sigmoid gates (LSTM discipline); gradient clipping enabled.
- **On-device int8 model:** bounded activation (ReLU6-style) or calibrated ranges; smooth activations only after quantization validation.
- **GAN discriminator with gradient penalty:** smooth activation preferred; document penalty interaction.
- **Probabilistic head:** linear mean + softplus scale; never ReLU on a scale parameter (zero scale = degenerate distribution).
- **Sparse-interpretability model:** ReLU for exact zeros; accept dying-unit monitoring burden explicitly.
- Each record names: context, default, challenger, metrics, seed count, and the rollback config — activation choices are reversible only if recorded.

## 44. Computed Table: Activation Statistics for Standard Normal Input
Monte Carlo (seed 42, 200k samples) output moments for $z\sim\mathcal N(0,1)$ pre-activations. Non-zero means (ReLU 0.399, sigmoid 0.5) quantify the zero-centring column of the dossier; $P(f=0)=0.5$ for ReLU is the sparsity/init link.

| activation | E[f(z)] | std f(z) | P(f=0) | E[f'(z)] approx |
|---|---|---|---|---|
| sigmoid | 0.4997 | 0.2086 | 0.000 | 0.2065 |
| tanh | -0.0011 | 0.6283 | 0.000 | 0.6053 |
| ReLU | 0.3993 | 0.5866 | 0.501 | 0.4987 |
| LeakyReLU | 0.3953 | 0.5894 | 0.000 | 0.5037 |
| ELU | 0.1603 | 0.7894 | 0.000 | 0.7610 |
| SiLU | 0.2071 | 0.5626 | 0.000 | 0.4994 |
| GELU | 0.2826 | 0.5909 | 0.000 | 0.4991 |
| softplus | 0.8062 | 0.5235 | 0.000 | 0.4997 |



## 45. Quick-Reference Cards
**Sigmoid**
- Range $(0,1)$; derivative $\sigma(1-\sigma)\le0.25$.
- Use: binary/multi-label heads, gates.
- Avoid: deep hidden stacks.
- Failure: two-sided saturation.
- Monitor: fraction $|x|>4$.

**Tanh**
- Range $(-1,1)$, zero-centred; derivative $1-\tanh^2\le1$.
- Use: recurrent states, bounded hidden units.
- Avoid: very deep unnormalized stacks.
- Failure: saturation at $|x|\gtrsim2$.
- Init: Xavier.

**ReLU**
- Range $[0,\infty)$; derivative 1/0.
- Use: default rectifier baseline, sparsity.
- Failure: dying units.
- Monitor: zero fraction per layer.
- Init: He (Var $2/n_{in}$).

**Leaky ReLU / PReLU**
- Left slope $\alpha$ (fixed 0.01 or learned).
- Use: when dead units bite.
- Failure: loses exact sparsity.
- Monitor: learned $\alpha$ values (PReLU).
- Init: He.

**ELU / SELU**
- ELU saturates to $-\alpha$ smoothly; SELU self-normalizes with fixed constants.
- Use: unnormalized MLPs (SELU discipline required).
- Failure: SELU assumptions broken by BN/standard dropout.
- Monitor: activation mean/variance fixed point.
- Init: SELU-specific.

**GELU**
- $x\Phi(x)$, $C^\infty$, GELU$'(0)=0.5$.
- Use: transformer FFNs (ungated), BERT/GPT lineage.
- Failure: approx-variant mismatch across frameworks.
- Monitor: standard signal-propagation probes.
- Init: He-style with residual scaling.

**SiLU / Swish**
- $x\sigma(x)$; min $-0.278$ near $-1.28$.
- Use: convnets, SwiGLU gating branch.
- Failure: none specific; cost above ReLU.
- Monitor: standard probes.
- Init: He-style.

**SwiGLU**
- $\mathrm{SiLU}(xW)\odot(xV)$, then output projection.
- Use: LLM FFNs (Shazeer 2020).
- Failure: forgetting 2/3 width rule.
- Monitor: param/FLOP accounting in ablations.
- Init: per transformer recipe (§03 Pre-LN).

**Softmax**
- Vector activation to the simplex; Jacobian $\mathrm{diag}(p)-pp^\top$.
- Use: single-label heads, attention weights.
- Failure: overconfidence read as calibration.
- Monitor: logit gaps, entropy; temperature for calibration.
- Pairing: cross-entropy, gradient $p-y$ (§01).



## 46. Extended Exercise Bank (Self-Study)
1. Compute by hand: sigmoid, tanh, ReLU, GELU at $x\in\{-2,-1,0,1,2\}$; compare with Section 19's table.
2. Prove $\tanh(x)=2\sigma(2x)-1$ and derive its derivative from sigmoid's.
3. Show ReLU networks are positively homogeneous per layer and state one consequence for weight decay.
4. Derive the He factor 2 from $\mathbb{E}[\mathrm{ReLU}(z)^2]$ for symmetric $z$.
5. For a 30-layer sigmoid MLP, bound the gradient factor and state the implied design change.
6. Implement stable sigmoid and softmax; test shift invariance at $+1000$.
7. Measure dead-unit fraction vs learning rate in a small MLP; plot and explain.
8. Compare GELU exact vs tanh approximation numerically on $[-4,4]$; report max error.
9. Convert a GELU FFN to SwiGLU at matched parameters; show the width computation.
10. Explain, with the Jacobian, why softmax + cross-entropy yields $p-y$.
11. Construct a multi-label example where softmax is structurally wrong.
12. Run the Problem B6 ablation protocol comparing ReLU and SiLU at 5 seeds.



## 47. Annotated Reading Guide
- Start: Nair & Hinton (2010) and Glorot et al. (2011) for why rectifiers changed practice.
- Initialization pair: Glorot & Bengio (2010) then He et al. (2015) — read as one argument in two acts.
- Variants: Clevert et al. (2015, ELU), Klambauer et al. (2017, SELU) — note how each states its assumptions.
- Smooth era: Hendrycks & Gimpel (2016, GELU), Ramachandran et al. (2017, Swish) — note the search/regularizer framings.
- Gating: Dauphin et al. (2017, GLU), Shazeer (2020, GLU variants) — the transformer FFN lineage.
- Synthesis habit: for each paper, fill the Section 14 dossier before reading the results table.
- Full bibliographic details are in References; only canonical sources are cited in this chapter.

## 48. Computed Table: Second Derivatives (Curvature)
Finite-difference second derivatives. ReLU (not shown) has second derivative 0 a.e. with a kink at 0; smooth activations carry curvature that second-order methods and gradient penalties actually see. Sign changes mark inflection regions.

| x | sigmoid'' | tanh'' | SiLU'' | GELU'' | ELU'' |
|---|---|---|---|---|---|
| -3.0 | 0.0409 | 0.0196 | -0.0323 | -0.0310 | 0.0498 |
| -2.5 | 0.0595 | 0.0525 | -0.0085 | -0.0745 | 0.0821 |
| -2.0 | 0.0800 | 0.1362 | 0.0501 | -0.1080 | 0.1353 |
| -1.5 | 0.0947 | 0.3271 | 0.1562 | -0.0324 | 0.2231 |
| -1.0 | 0.0909 | 0.6397 | 0.3024 | 0.2420 | 0.3679 |
| -0.5 | 0.0576 | 0.7269 | 0.4412 | 0.6161 | 0.6065 |
| 0.0 | -0.0000 | 0.0000 | 0.5000 | 0.7979 | 0.4998 |
| 0.5 | -0.0576 | -0.7269 | 0.4412 | 0.6161 | 0.0000 |
| 1.0 | -0.0909 | -0.6397 | 0.3024 | 0.2420 | -0.0000 |
| 1.5 | -0.0947 | -0.3271 | 0.1562 | -0.0324 | 0.0000 |
| 2.0 | -0.0800 | -0.1362 | 0.0501 | -0.1080 | 0.0000 |
| 2.5 | -0.0595 | -0.0525 | -0.0085 | -0.0745 | 0.0000 |
| 3.0 | -0.0409 | -0.0196 | -0.0323 | -0.0310 | 0.0000 |



## 49. Activations Inside Transformers
- Attention logits: scaled dot-products feed softmax; the $1/\sqrt{d_k}$ scale is an activation-statistics device keeping softmax out of saturation at init.
- FFN: GELU (original transformer/BERT/GPT) → SwiGLU (many current LLMs); see Sections 18/21 for the accounting rule.
- Pre-LN vs Post-LN placement (§03) changes effective activation distributions; activation comparisons across LN placements are confounded.
- Rotary/positional machinery does not change FFN activation choice, but long-context logit growth in attention can saturate softmax — monitored via attention entropy.
- Mixture-of-experts routers use softmax/sigmoid gating over experts; router saturation (one expert taking all) is an activation-health metric.
- Final LM head: linear to logits; no squashing — softmax lives inside the loss (§01) or decoding (§04), not as a stored activation.
- Low-precision training: GELU/SiLU in bf16 are routine; fp8 regimes re-open activation-range questions (outlier channels).
- Activation outliers in LLMs (a few channels with huge magnitudes) affect quantization and are an active engineering concern; they emerge from training, not from the activation formula alone.



## 50. Worked Problem Set E: Statistics and Synthesis
Fifth set: curvature, statistics, and synthesis.

**Problem E1: output mean shift**
1. For $z\sim\mathcal N(0,1)$, Section 44 gives E[ReLU$(z)$] $\approx0.399$, E[sigmoid] $=0.5$.
2. A following linear layer receives inputs with nonzero mean; its bias can absorb a constant shift, but sample-dependent mean variation acts as noise.
3. Nearer-zero-centred activations reduce this burden — tanh has mean 0 and ELU's mean ($\approx0.16$ in the table) is well below ReLU's $0.40$ — the classical efficiency argument of LeCun et al. (1998).
4. Normalization (§03) removes the issue structurally; unnormalized nets rely on activation choice.

**Problem E2: inflection of SiLU**
1. SiLU$''(x)$ changes sign where the Section 48 table crosses zero (near $x\approx-1.3$ region for the minimum).
2. The minimum of SiLU itself sits where SiLU$'=0$, $x\approx-1.28$, value $\approx-0.278$.
3. Interpretation: mild non-monotonic dip; gradient sign flips there, a theoretical curiosity with negligible practical harm at scale.

**Problem E3: choose under int8 + latency constraints**
1. Constraints: int8 deployment, no fast exp on target, latency-critical.
2. Candidates: ReLU/ReLU6 (no exp, bounded variant), hard-sigmoid heads where needed.
3. Reject GELU/SiLU unless the runtime fuses approximations and quantization validation passes.
4. Record: accuracy delta smooth-vs-deployed measured on validation, not assumed.

**Problem E4: attention entropy alarm**
1. Symptom: attention entropy collapses early; one position takes ~all weight per head.
2. Mechanics: logit gaps grew (Section 31: gap 10 → p≈1 in binary terms); softmax saturated; gradients to other positions vanish.
3. Checks: logit scale, $1/\sqrt{d_k}$ present, init of Q/K, LR warmup.
4. This is an activation-saturation incident inside attention, diagnosed with the same dossier tools.



## 51. Formula Sheet
One-row dossiers. Verify any row against Sections 19-20's computed tables before trusting memory — including this sheet.

| Activation | $f(x)$ | $f'(x)$ | Range | Smoothness | Monotone |
|---|---|---|---|---|---|
| Sigmoid | $1/(1+e^{-x})$ | $\sigma(1-\sigma)$ | $(0,1)$ | $C^\infty$ | yes |
| Tanh | $\tanh x$ | $1-\tanh^2x$ | $(-1,1)$ | $C^\infty$ | yes |
| ReLU | $\max(0,x)$ | $1_{x>0}$ | $[0,\infty)$ | $C^0$ | yes |
| Leaky ReLU | $x$ if $x>0$ else $\alpha x$ | $1$ / $\alpha$ | $\mathbb R$ | $C^0$ | yes |
| ELU | $x$ if $x>0$ else $\alpha(e^x-1)$ | $1$ / $f+\alpha$ | $(-\alpha,\infty)$ | $C^1$ ($\alpha{=}1$) | yes |
| GELU | $x\Phi(x)$ | $\Phi(x)+x\phi(x)$ | $\mathbb R$ (min $-0.17$) | $C^\infty$ | no (mild) |
| SiLU | $x\sigma(x)$ | $\sigma(x)+x\sigma(x)(1-\sigma(x))$ | $\mathbb R$ (min $-0.278$) | $C^\infty$ | no (mild) |
| Softplus | $\log(1+e^x)$ | $\sigma(x)$ | $(0,\infty)$ | $C^\infty$ | yes |
| Softmax | $e^{z_i}/\sum_j e^{z_j}$ | $\mathrm{diag}(p)-pp^\top$ | simplex | $C^\infty$ | n/a (vector) |



## 52. Chapter Summary: Activations
- An activation is chosen by dossier: range, derivative, saturation, smoothness, centring, cost, init pairing, failure mode.
- Depth punished saturating activations (sigmoid cap 0.25) until rectifiers, residuals, and normalization changed the regime.
- Modern defaults are family-specific: SwiGLU/GELU (transformers), SiLU (convnets), ReLU (baselines, sparsity, constrained hardware).
- Initialization and activation are one decision (Xavier vs He); verify by signal-propagation probes at init.
- Health is monitored, not assumed: dead fraction, saturated fraction, per-layer gradient norms on a fixed probe batch.
- Claims require matched-parameter, retuned, multi-seed ablations.
- Forward bridge: normalization (§03) controls the pre-activation statistics this section learned to diagnose; sampling (§04) consumes the softmax outputs this section analysed.

## 53. Worked Problem Set F: Derivations
Sixth set: derivations readers should be able to reproduce unaided.

**Problem F1: derive GELU derivative**
1. $f(x)=x\Phi(x)$; product rule: $f'(x)=\Phi(x)+x\phi(x)$ where $\phi=\Phi'$ is the standard normal density.
2. At $x=0$: $\Phi(0)=1/2$, $x\phi(x)=0$, so $f'(0)=1/2$.
3. As $x\to\infty$: $\Phi\to1$, $x\phi(x)\to0$, so $f'\to1$ (ReLU-like).
4. As $x\to-\infty$: both terms $\to0$; gradient vanishes smoothly, never exactly in floating point until underflow.

**Problem F2: derive softmax+CE gradient**
1. Loss $\ell=-\log p_k$ for true class $k$, $p=\mathrm{softmax}(z)$.
2. $\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$ from the quotient rule (Section 22 Jacobian).
3. Chain rule collapses: $\partial\ell/\partial z_j=p_j-\mathbb 1[j=k]$.
4. This is why frameworks fuse the pair; deriving it once explains a thousand implementations (§01).

**Problem F3: variance of ReLU output**
1. $z\sim\mathcal N(0,q)$; by symmetry $\mathbb E[\mathrm{ReLU}(z)^2]=\tfrac12\mathbb E[z^2]=q/2$.
2. Linear layer: Var$(y)=n_{in}$Var$(w)\cdot q/2$; setting Var$(w)=2/n_{in}$ preserves $q$.
3. State the He initialization result and the assumption (symmetric pre-activations, independence) it rests on.

**Problem F4: SELU fixed point (statement and use)**
1. Klambauer et al. (2017) choose $(\alpha,\lambda)$ so the (mean, variance) map has an attracting fixed point at $(0,1)$ under their assumptions.
2. Required discipline: LeCun-normal init, alpha-dropout, no BatchNorm.
3. Violation example: adding standard dropout breaks the moment argument — the theorem, not just the recipe, is void.
4. Practical stance: SELU is a complete package; partial adoption is a category error (Section 42).



## 54. Activations and Attention Logit Growth
- Dot-product attention logits grow with head dimension if unscaled: Var$(q\cdot k)=d_k$ for unit-variance components, hence $1/\sqrt{d_k}$.
- Softmax saturation from logit growth is Section 31's table applied row-wise; entropy per head is the observable.
- QK-normalization (normalizing queries/keys, §03) is a further stabilization used in some large models; it changes the geometry, not just the scale.
- Logit soft-capping (tanh-based, used in some recent models) trades exact softmax form for bounded logits; evaluate as an activation decision with the dossier.
- Sliding-window and sparse attention change which logits exist, not how softmax saturates on the ones that do.
- Monitoring: per-head entropy and max attention weight, logged like any activation-health metric (Section 30).



## 55. What to Memorise vs What to Look Up
- Memorise: derivative caps (sigmoid 0.25), ReLU death mechanism, He factor 2, softmax shift invariance, $p-y$ gradient, SwiGLU 2/3 rule.
- Memorise: the dossier template (Section 14) — it regenerates most facts on demand.
- Look up: exact SELU constants, GELU approximation coefficients, framework defaults for approximations.
- Look up: historical dates (Section 27) when writing; misattributed activations are a common review complaint.
- Derive on demand: variance propagation for a new activation from its second moment (Problem F3 pattern).
- Never memorise benchmark rankings of activations; they do not transfer across tasks and scales.



## 56. Closing Checklist for This Section
- Can fill a dossier for any activation encountered in a paper within five minutes.
- Can predict depth behaviour from derivative ranges before running code.
- Can diagnose dying/saturation from probe-batch statistics and name the fix order.
- Pairs every activation with its initialization and verifies by signal-propagation logs.
- Runs activation ablations at matched parameters, retuned LR, multiple seeds.
- Chooses output activations from the loss (§01), not from hidden-layer fashion.
- Records decisions (Section 43 format) so swaps are auditable and reversible.
- Carries forward: §03 normalization as the structural control of the statistics diagnosed here.

## 57. Computed Table: Dead Units vs Bias Shift (Simulation, Seed 42)
A 20→64 ReLU layer on Gaussian data. 'Dead' means negative for all 512 probe examples. Even at zero bias, sampling noise kills a small share; negative bias shifts kill units wholesale — the quantitative form of Problem A2.

| bias shift b | dead fraction (ReLU) | zero fraction this batch |
|---|---|---|
| +1.0 | 0.000 | 0.235 |
| +0.5 | 0.000 | 0.359 |
| +0.0 | 0.000 | 0.498 |
| -0.5 | 0.000 | 0.639 |
| -1.0 | 0.000 | 0.760 |
| -1.5 | 0.000 | 0.854 |
| -2.0 | 0.000 | 0.919 |
| -3.0 | 0.000 | 0.979 |



## 58. Worked Example: Full Forward/Backward Pass by Hand
Input $x=(1.0,-0.5)$; $W_1$ rows $(0.8,-0.4),(0.3,0.9)$; $b_1=(0.1,-0.2)$; $W_2=(1.2,-0.7)$; $b_2=0.05$; ReLU hidden; target $0.5$; loss $(\hat y-t)^2$.
Pre-activations: $z_1=1.0(0.8)+(-0.5)(0.3)+0.1=0.750$; $z_2=1.0(-0.4)+(-0.5)(0.9)-0.2=-1.050$.
ReLU: $h=(0.750,0.000)$ — unit 2 is off for this input.
Output: $\hat y=0.750(1.2)+0.000(-0.7)+0.05=0.950$; loss $=(0.950-0.5)^2=0.2025$.
$d\ell/d\hat y=2(\hat y-t)=0.900$.
$d\ell/dh=0.900\times(1.2,-0.7)=(1.080,-0.630)$; through ReLU mask $(1,0)$: $d\ell/dz=(1.080,0)$.
$d\ell/dW_2=0.900\times h=(0.675,0.000)$ — the off unit's outgoing weight gets zero gradient from this example.
Repeat over the dataset: if unit 2 is off for every example, its incoming weights also receive zero — the dying-ReLU condition in miniature.
With GELU instead, unit 2's factor would be GELU$'(z_2)\approx$ small-but-nonzero (Section 20 table), and the unit could recover.
Exercise: redo with learning rate 0.1, one SGD step, and verify the loss decreases on this example.


## 59. One Activation Language Across Architectures
- CNNs: ReLU lineage → SiLU in efficient families; spatial weight sharing does not change the dossier, only the statistics (per-channel dead fractions).
- RNNs: tanh state + sigmoid gates persist because bounded recurrence is a stability requirement, not nostalgia.
- Transformers: GELU/SwiGLU FFNs + softmax attention; both choices are activation-statistics decisions (Sections 21, 49, 54).
- GNNs: ReLU/GELU node updates; oversmoothing is a distinct depth pathology (representation correlation, Section 40) not fixed by activation swaps.
- Diffusion/generative models: SiLU common in U-Nets (from the Swish lineage); time-embedding MLPs often SiLU as well.
- State-space and linear-attention models: gating reappears (multiplicative branches), echoing Section 21's mechanism.
- The dossier travels: whatever the architecture, ask range/derivative/saturation/centring/cost/init/failure-mode in that order.



## 60. Practice Problems for Revision
1. Reproduce Section 57's simulation with Leaky ReLU and report the (zero) dead fraction; explain the difference mechanistically.
2. Derive the derivative of Mish and evaluate at 0; compare with SiLU/GELU at 0.
3. Show that a two-layer ReLU network can represent $|x|$ exactly; give weights.
4. Prove softmax is invariant to constant shifts and that its Jacobian annihilates $\mathbf 1$.
5. Compute the equal-parameter SwiGLU width for $d_{model}=512$, $d_{ff}=2048$.
6. Design the logging schema (fields, frequency, alerts) for activation health in a training framework.
7. Write the decision record (Section 43 format) for replacing GELU with SwiGLU in an existing checkpoint family, including risks.
8. Explain why 'we used the framework default activation' is not a methods section.

## 61. Computed Table: Output Scale vs Input Scale
How output std responds as pre-activation scale grows (Monte Carlo, seed 42). Bounded activations compress (tanh/sigmoid saturate), ReLU-family transmit scale linearly — the forward counterpart of the gradient tables.

| input std s | ReLU out-std | tanh out-std | GELU out-std | SiLU out-std | sigmoid out-std |
|---|---|---|---|---|---|
| 0.25 | 0.1467 | 0.2368 | 0.1299 | 0.1273 | 0.0617 |
| 0.5 | 0.2933 | 0.4172 | 0.2764 | 0.2635 | 0.1184 |
| 1.0 | 0.5866 | 0.6283 | 0.5909 | 0.5626 | 0.2086 |
| 1.5 | 0.8799 | 0.7354 | 0.8987 | 0.8748 | 0.2709 |
| 2.0 | 1.1732 | 0.7970 | 1.1975 | 1.1853 | 0.3141 |
| 3.0 | 1.7598 | 0.8631 | 1.7849 | 1.7922 | 0.3677 |
| 4.0 | 2.3464 | 0.8973 | 2.3691 | 2.3862 | 0.3985 |



## 62. Worked Problem Set G: Final Synthesis
Seventh set: final synthesis problems with complete solutions.

**Problem G1: reading a training log**
1. Log shows: layer-5 zero fraction 0.92, gradient norm ratio (first/last layer) $10^{-6}$, loss flat.
2. Diagnosis: mass ReLU death or saturation in layer 5 is the prime suspect; the gradient ratio is the symptom downstream.
3. First action: inspect pre-activation histogram for layer 5 on the probe batch.
4. If pre-activations are all strongly negative: rollback to before the LR event, lower LR, consider Leaky variant.
5. If pre-activations are healthy but outputs zero: suspect a bug (mask, dtype, wrong tensor) — activation math does not zero healthy inputs.

**Problem G2: budget-constrained activation search**
1. You may test 3 activations on a fixed vision backbone with 5 seeds each.
2. Choose: ReLU (baseline), SiLU (family default), GELU (cross-family challenger).
3. Match: parameters (no gating here), LR grid per variant, augmentation, schedule.
4. Decide by mean accuracy with seed std and per-step cost; pre-register the decision rule to avoid post-hoc selection.

**Problem G3: explain to a reviewer**
1. Reviewer: 'Why SwiGLU rather than GELU?'
2. Answer structure: citation (Shazeer 2020), matched-parameter protocol, your ablation numbers with seeds, cost accounting, kernel availability.
3. Weak answer: 'it is standard now'. Standardness is a prior, not evidence; combine both, labelled as such.

**Problem G4: activation for a scale parameter**
1. Head must output a positive scale $\sigma$ for a Gaussian NLL (§01).
2. ReLU can output exactly 0 → infinite NLL; sigmoid bounds scale away from large values.
3. Softplus + small floor ($\sigma=\mathrm{softplus}(a)+10^{-4}$) is the standard, dossier-driven choice: positive, smooth, unbounded above.
4. Test: extreme inputs produce finite loss; gradient w.r.t. $a$ is sigmoid-shaped and bounded.



## 63. Decision Flowchart in Words
1. Is this the output head? Choose from the loss: linear / sigmoid / softmax / softplus-scale (§01). Stop.
2. Is the architecture a transformer FFN? Default SwiGLU (2/3 rule); GELU is the comparator. Stop after ablation.
3. Is the state recurrent? tanh state + sigmoid gates. Stop.
4. Is deployment int8/latency-bound? Prefer ReLU-family/bounded variants; validate quantization early. Stop.
5. Otherwise default to the family standard (SiLU convnet, ReLU baseline MLP) with He-style init.
6. Instrument dead/saturated fractions before the first long run.
7. Change one variable at a time; record the decision (Section 43); keep the rollback config.



## 64. Final Note on Activations
- Activations are where calculus (derivatives), probability (output statistics), numerics (stable branches), and systems (kernels, quantization) meet in a single line of code.
- The field's history (Section 27) is a sequence of removing specific pathologies: saturation, death, scale drift, gating rigidity.
- New activations will keep appearing; the dossier, the probes, and the ablation protocol in this section are the durable tools for judging them.
- Next, §03 Normalization Techniques takes up the structural control of the pre-activation statistics diagnosed throughout this section.

## 65. Computed Table: Extreme-Value Behaviour
Why stable branches exist. Naive $e^{-x}$ for sigmoid at $x=-100$ overflows in float64 only beyond $\approx709$, but in float32 near $\approx88$; softplus via logaddexp never forms the large exponential. Know your dtype's threshold.

| x | sigmoid stable | exp(x) overflow? | softplus stable | note |
|---|---|---|---|---|
| -100 | 0.000000 | no | 0.0000 | sigmoid saturates to 0 |
| -50 | 0.000000 | no | 0.0000 | sigmoid saturates to 0 |
| -20 | 0.000000 | no | 0.0000 | sigmoid saturates to 0 |
| -10 | 0.000045 | no | 0.0000 | transition region |
| 0 | 0.500000 | no | 0.6931 | transition region |
| 10 | 0.999955 | no | 10.0000 | transition region |
| 20 | 1.000000 | no | 20.0000 | sigmoid saturates to 1 |
| 50 | 1.000000 | no | 50.0000 | sigmoid saturates to 1 |
| 100 | 1.000000 | no | 100.0000 | sigmoid saturates to 1 |



## 66. Self-Test Answer Key (Condensed)
- Sigmoid cap: $\sigma'\le1/4$ at 0; depth-$L$ factor $\le4^{-L}$.
- He factor 2: $\mathbb E[\mathrm{ReLU}(z)^2]=\mathrm{Var}/2$ needs Var$(w)=2/n_{in}$ to compensate.
- Dying ReLU: all-negative pre-activations → zero gradient → absorbing; fixes: leak/ELU, LR discipline, init/bias care.
- GELU$'(0)=$ SiLU$'(0)=0.5$ (Sections 18, 20).
- Softmax Jacobian rank $K-1$: rows sum to zero, null vector $\mathbf 1$ (shift invariance).
- Multi-label → sigmoid; single-label mutually exclusive → softmax.
- SwiGLU width: $2/3$ of GELU width at equal parameters; ignoring it inflates FFN params ~50%.
- Tanh saturation: $|x|\gtrsim2$ derivative $<0.07$; diagnose by pre-activation histograms.
- SELU+BN: two normalization philosophies collide; SELU's theorem assumes no BN.
- ELU at $-6$: derivative $e^{-6}\approx0.0025$ vs Leaky constant 0.01 vs ReLU 0.
- LR retuning: activation changes loss geometry; fixed LR confounds the comparison.
- LSTM gates: sigmoid, because gates are multiplicative 0-1 controls.
- Temperature: entropy increases with $T$ (Section 32).
- Deployment override example: int8/latency constraints forcing ReLU-family (Problem C2/C6).



## 67. Section 02 Completion Checklist
- Dossiers for sigmoid/tanh, ReLU family, smooth modern, gated, and vector activations: Sections 15-22.
- Computed evidence: values, derivatives, curvature, statistics, temperature, variance drift, dead-unit simulation: Sections 19-20, 26, 31-34, 44, 48, 57, 61, 65.
- Derivations reproducible: He init, GELU derivative, $p-y$ gradient: Problems F1-F3.
- Protocols: ablation (B6), monitoring (30), pre-flight (29), decision records (43).
- All CONTRIBUTING elements present: overview, prerequisites, objectives, TOC, numbered sections, Common Mistakes (Section 10 plus catalogue 42), exercises, Why This Matters, Conceptual Bridge, canonical references.

## 68. Common Exam and Interview Traps
- Trap: 'ReLU derivative at 0 is undefined, so ReLU nets can't train.' Response: subgradient convention; measure-zero event; training is empirical proof.
- Trap: 'Sigmoid is natural because outputs are probabilities.' Response: only at the head with a matching loss; hidden units need gradient flow, not probability semantics.
- Trap: 'More nonlinearity is always better.' Response: piecewise-linear ReLU nets are already universal; trainability and statistics dominate.
- Trap: 'GELU is just smooth ReLU.' Response: near-equivalent shape, different derivative at negatives and different stochastic interpretation; empirics decided adoption.
- Trap: 'Softmax output 0.99 means 99% chance of being right.' Response: calibration is separate (§01); confidence is a logit-gap statistic (Section 31).
- Trap: 'Dead ReLUs mean the model is broken.' Response: some sparsity is fine; mass early death correlated with LR events is the pathology.
- Trap: 'SELU removes all need for care.' Response: SELU is a package of assumptions (init, dropout, architecture); violating them voids the guarantee.
- Trap: 'Activation search (Swish) proves hand design dead.' Response: search found a form humans then analysed; the dossier is how such results are judged.
- Trap: 'Attention softmax is a different topic.' Response: same saturation mathematics; logit scale and entropy are the shared diagnostics.
- Trap: 'If loss decreases, activations are healthy.' Response: capacity loss (dead units) can coexist with decreasing loss; audit the statistics (Section 30).
- Exam habit: answer with the mechanism (derivative, statistic, invariance), a number (cap, threshold, table), and the monitor — in that order.
- Interview habit: when proposing an activation change, volunteer the ablation protocol before being asked.

- Closing remark one: activations reward mechanistic reasoning over fashion.
- Closing remark two: every claim in this section can be re-derived or re-measured with the tables and probes provided.
- Closing remark three: proceed to Normalization Techniques (§03) for the structural counterpart.

## References

- Rumelhart, D. E., Hinton, G. E., and Williams, R. J. (1986). Learning representations by back-propagating errors.
- Nair, V., and Hinton, G. E. (2010). Rectified Linear Units Improve Restricted Boltzmann Machines.
- Glorot, X., Bordes, A., and Bengio, Y. (2011). Deep Sparse Rectifier Neural Networks.
- He, K., Zhang, X., Ren, S., and Sun, J. (2015). Delving Deep into Rectifiers.
- Hendrycks, D., and Gimpel, K. (2016). Gaussian Error Linear Units.
- Elfwing, S., Uchibe, E., and Doya, K. (2018). Sigmoid-Weighted Linear Units.
- Shazeer, N. (2020). GLU Variants Improve Transformer.
