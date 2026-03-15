[← Back to ML-Specific Math](../README.md) | [Next: Normalization Techniques →](../03-Normalization-Techniques/notes.md)

---

# Activation Functions for Neural Networks

> _"Without non-linearity, all the depth in the world is worth nothing — you are still just rotating and scaling."_
> — paraphrasing Yann LeCun, *A Path Towards Autonomous Machine Intelligence* (2022)

## Overview

Activation functions are the gatekeepers of non-linearity in neural networks. Without them, a network of arbitrary depth reduces to a single affine transformation — the composition of linear maps is linear, and no amount of depth can express a non-linear function. The moment a non-linear activation is introduced after each linear layer, something profound happens: the network gains the capacity to approximate any continuous function on a compact domain. This is the content of the universal approximation theorem, one of the foundational results in deep learning theory.

But the *choice* of activation function is far from cosmetic. It determines the gradient flow through the network: whether gradients vanish to zero in early layers (preventing learning), explode toward infinity (causing instability), or flow cleanly through hundreds of layers. It shapes the loss landscape — how smooth, how convex, how easily navigated by gradient descent. It affects the statistics of activations at each layer, determining whether initialisation schemes maintain variance, and whether the network is amenable to techniques like batch normalisation.

This section develops the mathematics of activation functions from first principles. We derive gradient formulas, analyse vanishing gradient phenomena, prove the universal approximation theorem, and study the properties (Lipschitz constants, monotonicity, saturation) that determine an activation's suitability for different tasks. We trace the historical arc from sigmoid (1943) through ReLU (2011) to GELU and SwiGLU (2016–2021), understanding each transition as a solution to a precise mathematical problem. The section culminates in modern transformer FFN variants — SwiGLU in LLaMA, GeGLU in T5 — and their connections to activation-aware quantisation and mechanistic interpretability.

## Prerequisites

- **Loss Functions** (§13-01): gradient of loss w.r.t. pre-activations; chain rule structure
- **Linear algebra** (Chapters 02–03): matrix-vector products, rank, composition of linear maps
- **Calculus and differentiation** (Chapter 04): chain rule, derivative of composed functions, Taylor series
- **Probability** (Chapter 06): Gaussian CDF, expectation, variance
- **Optimisation** (Chapter 08): gradient descent, loss landscape geometry, Hessian
- **Neural network layers** (§14-02): pre-activation $\mathbf{z}^{[l]}$, post-activation $\mathbf{a}^{[l]}$, backpropagation

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive derivations: activation curves, vanishing gradient analysis, GELU/SwiGLU, softmax Jacobian, initialisation gain, temperature scaling |
| [exercises.ipynb](exercises.ipynb) | 10 graded problems covering sigmoid derivatives, ReLU gradients, GELU approximation, vanishing gradients, SELU normalisation, Lipschitz constants, SwiGLU budgets, He init, temperature calibration, and activation engineering |

## Learning Objectives

After completing this section, you will:

- Prove that a network of linear layers with no activation is equivalent to a single linear layer, regardless of depth
- State the universal approximation theorem (Cybenko 1989; Hornik 1991) and explain what it guarantees and what it does not
- Derive the derivative of sigmoid and tanh and quantify vanishing gradients at depth $L$
- Explain the dying ReLU problem and derive the condition under which a neuron permanently outputs zero
- Define GELU, Swish/SiLU, and Mish mathematically and identify the stochastic interpretation of GELU
- Derive the Jacobian of softmax and use it to verify the stable gradient formula for cross-entropy
- Compute the Lipschitz constant of ReLU, sigmoid, tanh, and GELU
- Derive Glorot/Xavier and He initialisation from the variance preservation requirement
- Explain SwiGLU's parameter budget advantage over standard FFN and why LLaMA uses hidden dimension $\frac{8}{3}d_{\text{model}}$
- Connect temperature scaling to calibration and explain the $\tau \to 0$ / $\tau \to \infty$ limits
- Describe how activation patching is used in mechanistic interpretability to locate knowledge in LLMs

---

## Table of Contents

- [Activation Functions for Neural Networks](#activation-functions-for-neural-networks)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition and Motivation](#1-intuition-and-motivation)
    - [1.1 What Activation Functions Do](#11-what-activation-functions-do)
    - [1.2 The Non-Linearity Requirement](#12-the-non-linearity-requirement)
    - [1.3 Activation Functions in Modern AI](#13-activation-functions-in-modern-ai)
    - [1.4 Historical Timeline 1943–2024](#14-historical-timeline-19432024)
  - [2. Formal Definitions and Taxonomy](#2-formal-definitions-and-taxonomy)
    - [2.1 Rigorous Definition](#21-rigorous-definition)
    - [2.2 Taxonomy by Properties](#22-taxonomy-by-properties)
    - [2.3 Key Measurable Properties](#23-key-measurable-properties)
  - [3. Classical Activations: Sigmoid and Tanh](#3-classical-activations-sigmoid-and-tanh)
    - [3.1 Sigmoid](#31-sigmoid)
    - [3.2 Hyperbolic Tangent](#32-hyperbolic-tangent)
    - [3.3 Vanishing Gradient Analysis](#33-vanishing-gradient-analysis)
    - [3.4 When to Use Classical Activations](#34-when-to-use-classical-activations)
  - [4. The ReLU Family](#4-the-relu-family)
    - [4.1 Rectified Linear Unit](#41-rectified-linear-unit)
    - [4.2 The Dying ReLU Problem](#42-the-dying-relu-problem)
    - [4.3 Leaky ReLU and PReLU](#43-leaky-relu-and-prelu)
    - [4.4 ELU and SELU](#44-elu-and-selu)
    - [4.5 Gradient Comparison Across ReLU Variants](#45-gradient-comparison-across-relu-variants)
  - [5. Modern Smooth Activations](#5-modern-smooth-activations)
    - [5.1 GELU](#51-gelu)
    - [5.2 Swish / SiLU](#52-swish--silu)
    - [5.3 Mish](#53-mish)
    - [5.4 Why Smooth Non-Monotone Activations Help](#54-why-smooth-non-monotone-activations-help)
  - [6. Output Layer Activations](#6-output-layer-activations)
    - [6.1 Softmax](#61-softmax)
    - [6.2 Log-Softmax and Numerical Stability](#62-log-softmax-and-numerical-stability)
    - [6.3 Sparsemax](#63-sparsemax)
    - [6.4 Temperature Scaling](#64-temperature-scaling)
  - [7. Gated Linear Units and Transformer FFN Variants](#7-gated-linear-units-and-transformer-ffn-variants)
    - [7.1 Gated Linear Unit](#71-gated-linear-unit)
    - [7.2 SwiGLU](#72-swiglu)
    - [7.3 GeGLU](#73-geglu)
    - [7.4 Parameter Budget in Gated FFN](#74-parameter-budget-in-gated-ffn)
  - [8. Activation Properties: Theory](#8-activation-properties-theory)
    - [8.1 Lipschitz Continuity](#81-lipschitz-continuity)
    - [8.2 Universal Approximation Theorem](#82-universal-approximation-theorem)
    - [8.3 Monotonicity and Fixed Points](#83-monotonicity-and-fixed-points)
    - [8.4 Symmetry and Zero-Centred Outputs](#84-symmetry-and-zero-centred-outputs)
  - [9. Gradient Flow and Initialisation](#9-gradient-flow-and-initialisation)
    - [9.1 Vanishing and Exploding Gradients](#91-vanishing-and-exploding-gradients)
    - [9.2 Glorot / Xavier Initialisation](#92-glorot--xavier-initialisation)
    - [9.3 He Initialisation](#93-he-initialisation)
    - [9.4 Gain Factors for Non-Standard Activations](#94-gain-factors-for-non-standard-activations)
  - [10. Computational Considerations and Approximations](#10-computational-considerations-and-approximations)
    - [10.1 Computational Cost Comparison](#101-computational-cost-comparison)
    - [10.2 Hard Approximations](#102-hard-approximations)
    - [10.3 Activation-Aware Quantisation](#103-activation-aware-quantisation)
  - [11. Applications in Modern AI](#11-applications-in-modern-ai)
    - [11.1 BERT and GPT: GELU as Standard](#111-bert-and-gpt-gelu-as-standard)
    - [11.2 LLaMA and PaLM: SwiGLU Dominance](#112-llama-and-palm-swiglu-dominance)
    - [11.3 Attention Patterns and Temperature](#113-attention-patterns-and-temperature)
    - [11.4 Activation Engineering and Mechanistic Interpretability](#114-activation-engineering-and-mechanistic-interpretability)
  - [12. Common Mistakes](#12-common-mistakes)
  - [13. Exercises](#13-exercises)
  - [14. Why This Matters for AI (2026 Perspective)](#14-why-this-matters-for-ai-2026-perspective)
  - [15. Conceptual Bridge](#15-conceptual-bridge)
    - [Looking Backward: Loss Functions](#looking-backward-loss-functions)
    - [The Central Theme: Non-Linearity and Expressiveness](#the-central-theme-non-linearity-and-expressiveness)
    - [Looking Forward: Normalisation Techniques](#looking-forward-normalisation-techniques)

---

## 1. Intuition and Motivation

### 1.1 What Activation Functions Do

Consider a neural network with $L$ layers. At each layer $l$, the network computes a pre-activation:

$$\mathbf{z}^{[l]} = W^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$

and a post-activation:

$$\mathbf{a}^{[l]} = \sigma(\mathbf{z}^{[l]})$$

where $\sigma$ is applied element-wise. Without $\sigma$, the composition of these layers is:

$$\mathbf{a}^{[L]} = W^{[L]}(W^{[L-1]}(\cdots(W^{[1]}\mathbf{x} + \mathbf{b}^{[1]}) \cdots) + \mathbf{b}^{[L-1]}) + \mathbf{b}^{[L]}$$

This simplifies to $W_{\text{eff}} \mathbf{x} + \mathbf{b}_{\text{eff}}$ for some effective weight matrix $W_{\text{eff}} = W^{[L]} W^{[L-1]} \cdots W^{[1]}$ and bias $\mathbf{b}_{\text{eff}}$. **No matter how many layers, a linear network is equivalent to a single linear layer.** A 100-layer linear network has no more expressive power than a 1-layer linear network.

The activation function breaks this collapse. By introducing a non-linear function after each linear transformation, the network can represent curved decision boundaries, non-linear feature interactions, and ultimately arbitrary continuous functions. Crucially, the non-linearity must be non-polynomial — a polynomial activation (e.g., $\sigma(z) = z^2$) still allows composition into a polynomial, which has limited expressive power per the Stone-Weierstrass theorem applied to polynomial spaces.

**For AI:** The FFN block in every transformer consists of two linear layers with an activation in between:

$$\text{FFN}(\mathbf{x}) = W_2 \, \sigma(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

Without $\sigma$, this reduces to a single linear layer $W_2 W_1 \mathbf{x} + \text{const}$, and the entire transformer degenerates to a linear function of the input embeddings. The activation function is the source of the FFN's capacity to act as a "key-value memory" (Geva et al., 2021), storing factual knowledge that can be retrieved by pattern-matching in the first layer.

### 1.2 The Non-Linearity Requirement

Why must the activation be non-polynomial? The Weierstrass approximation theorem guarantees that any continuous function on a compact interval can be approximated by polynomials. However, in multiple dimensions, polynomial representations suffer from the **curse of dimensionality**: approximating an arbitrary function in $\mathbb{R}^d$ with degree-$k$ polynomials requires $\binom{d+k}{k}$ basis functions, which grows exponentially with $d$.

The universal approximation theorem avoids this by using **superposition** of shifted, scaled copies of a single non-linear function. Cybenko (1989) showed that for any continuous $\sigma$ that is not a polynomial, a network with one hidden layer of sufficient width can approximate any continuous function on a compact subset of $\mathbb{R}^d$ to arbitrary accuracy in $L^\infty$ norm. The key property required is that $\sigma$ is **non-polynomial**, not that it has any particular shape.

A more operational characterisation: the activation introduces **branching** in the computation graph. Consider ReLU: $\operatorname{ReLU}(z) = \max(0, z)$. This defines two regimes — $z \le 0$ (output zero, no gradient) and $z > 0$ (output $z$, gradient 1). Different inputs activate different subsets of neurons, effectively implementing a different linear function for each input region. A network with $n$ ReLU neurons in one hidden layer can implement at most $\sum_{i=0}^{d} \binom{n}{i}$ distinct linear regions (Pascanu et al., 2013), a number that grows super-polynomially with $n$ and $d$.

**Non-examples (activations that fail universality):**
- Identity $\sigma(z) = z$: composition of linear layers is linear
- Polynomial $\sigma(z) = z^2$: composition of polynomial layers is polynomial
- Constant $\sigma(z) = 1$: destroys all information
- Step function approximations via piecewise-linear with finitely many pieces: approximates but polynomial growth in pieces needed

### 1.3 Activation Functions in Modern AI

The evolution of activation functions traces the history of deep learning itself. Each transition solved a specific mathematical problem:

| Era | Dominant Activation | Problem Solved | Problem Introduced |
|---|---|---|---|
| 1943–1986 | Step / hard sigmoid | Biological plausibility | Non-differentiable |
| 1986–2011 | Sigmoid, tanh | Differentiable; bounded | Vanishing gradients in deep nets |
| 2011–2015 | ReLU | Non-saturating; sparse; fast | Dying neurons; not zero-centred |
| 2015–2016 | ELU, PReLU, SELU | Negative outputs; self-normalising | Slower to compute |
| 2016–2020 | GELU | Smooth; non-monotone; stochastic interp. | Slight compute overhead vs ReLU |
| 2020–present | SwiGLU, GeGLU | Gating; better scaling; LLM quality | Double-projection parameter cost |

The current frontier (2026) is dominated by **gated** activations in transformer FFN blocks. SwiGLU (Shazeer, 2020) is used in LLaMA 2/3, PaLM, Gemma, Mistral, Qwen, and Falcon. GeGLU (Noam Shazeer, unpublished; widely adopted) is used in T5 v1.1 and GPT-NeoX. The empirical finding is consistent: gated activations outperform standard FFN on perplexity benchmarks, with gains of 0.3–1.0 perplexity points on language modelling (Su et al., 2024).

### 1.4 Historical Timeline 1943–2024

```
ACTIVATION FUNCTION HISTORY
════════════════════════════════════════════════════════════════════════

  1943  McCulloch & Pitts — binary threshold neuron: σ(z) = H(z)
  1958  Rosenblatt Perceptron — same threshold; first learning rule
  1986  Rumelhart, Hinton, Williams — backprop with sigmoid σ(z)=1/(1+e^{-z})
  1989  Cybenko — universal approximation theorem (sigmoid)
  1991  Hornik — UAT generalised to any non-polynomial activation
  1991  LeCun et al. — tanh preferred over sigmoid; zero-centred analysis
  1998  LeCun et al. — sigmoid saturation slows learning; weight init study
  2010  Glorot & Bengio — vanishing gradients in sigmoid; Xavier init
  2011  Glorot, Bordes, Bengio — ReLU for deep networks; sparse activations
  2013  Maas et al. — Leaky ReLU; fixed slope for negative inputs
  2013  Pascanu et al. — number of linear regions in deep ReLU nets
  2015  He et al. — PReLU; kaiming/He initialisation for ReLU
  2015  Clevert et al. — ELU; negative regime; continuous derivative
  2016  Klambauer et al. — SELU; self-normalising NNs; fixed point proof
  2016  Hendrycks & Gimpel — GELU; stochastic interpretation; used in BERT
  2017  Ramachandran et al. — Swish (x·σ(βx)) via NAS; better than ReLU
  2018  Devlin et al. (BERT) — GELU standardised for transformers
  2019  Misra — Mish activation; smooth and non-monotone
  2020  Shazeer — SwiGLU; gated FFN; parameter efficiency
  2020  Lepikhin et al. (GShard) — GeGLU in production LLMs
  2022  Touvron et al. (LLaMA) — SwiGLU as default; hidden dim 8d/3
  2023  LLaMA 2, Gemma, Mistral — SwiGLU universally adopted
  2024  Activation-aware quantisation (SmoothQuant, AWQ) — activation
         outlier analysis for INT8/INT4 deployment

════════════════════════════════════════════════════════════════════════
```


---

## 2. Formal Definitions and Taxonomy

### 2.1 Rigorous Definition

**Definition (Activation Function).** An activation function is a measurable function $\sigma: \mathbb{R} \to \mathbb{R}$ applied element-wise to a vector. When applied to a vector $\mathbf{z} \in \mathbb{R}^n$, we write $\sigma(\mathbf{z})$ to mean the vector $(\sigma(z_1), \ldots, \sigma(z_n))^\top$.

The pre-activation at layer $l$ is $\mathbf{z}^{[l]} = W^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]} \in \mathbb{R}^{n_l}$, and the activation is $\mathbf{a}^{[l]} = \sigma(\mathbf{z}^{[l]}) \in \mathbb{R}^{n_l}$.

**Differentiability classes:**
- $C^0$ (continuous): required for backpropagation to be well-defined almost everywhere. Non-smooth points (e.g., $z=0$ in ReLU) are handled via subgradients.
- $C^1$ (continuously differentiable): gradient is continuous; preferred for second-order methods.
- $C^\infty$ (smooth): infinitely differentiable; e.g., sigmoid, tanh, GELU, Swish.
  
**Saturation:** An activation saturates when $\lim_{z \to \pm\infty} \sigma'(z) = 0$. Saturation is the proximate cause of vanishing gradients — once a neuron is in the saturated regime, its gradient is near zero and it contributes almost nothing to learning. Sigmoid and tanh both saturate in both tails. ReLU saturates only in the negative tail (gradient zero for $z < 0$, constant 1 for $z > 0$). GELU and Swish saturate mildly in the negative tail (approaching $-0.17$ and $-\infty$ respectively as $z \to -\infty$, but with non-zero gradient).

### 2.2 Taxonomy by Properties

The following taxonomy organises all major activation functions along the dimensions that matter most for practical network design:

| Activation | Monotone | Bounded above | Bounded below | Smooth ($C^\infty$) | Zero-centred output | Lipschitz $K$ |
|---|---|---|---|---|---|---|
| Sigmoid $\sigma$ | Yes | Yes (1) | Yes (0) | Yes | No (range $(0,1)$) | 0.25 |
| Tanh | Yes | Yes (1) | Yes (−1) | Yes | Yes | 1.00 |
| ReLU | Yes | No | Yes (0) | No ($C^0$) | No | 1.00 |
| Leaky ReLU | Yes | No | No | No ($C^0$) | No | 1.00 |
| ELU | Yes | No | Yes ($-\alpha$) | No ($C^1$) | No | max(1,$\alpha$) |
| SELU | Yes | No | Yes ($-\lambda\alpha$) | No ($C^1$) | Yes (self-norm.) | $\lambda \approx 1.05$ |
| GELU | No | No | Yes ($\approx -0.17$) | Yes | Approx. | ≈1.13 |
| Swish | No | No | Yes ($\approx -0.28$) | Yes | Approx. | ≈1.10 |
| Mish | No | No | Yes ($\approx -0.31$) | Yes | Approx. | ≈1.07 |
| Softmax | — | (vector) | (vector) | Yes | — | 1 (output) |

**For AI:** The "zero-centred output" column connects directly to gradient dynamics. If $\sigma$ outputs values in $(0,1)$ (sigmoid), then the gradient $\partial \mathcal{L}/\partial W^{[l]}$ has all components with the same sign as the upstream gradient (LeCun et al., 1998). This creates a **zig-zag gradient problem** where weight updates must all go in the same direction, slowing convergence. Tanh and the modern smooth activations (GELU, Swish) are approximately zero-centred, avoiding this issue.

### 2.3 Key Measurable Properties

Given an activation $\sigma$, the following quantitative properties determine its behaviour in deep networks:

**1. Lipschitz constant.** The smallest $K \ge 0$ such that $\lvert \sigma(a) - \sigma(b) \rvert \le K \lvert a - b \rvert$ for all $a, b$. For differentiable activations, $K = \sup_z \lvert \sigma'(z) \rvert$. A small Lipschitz constant prevents gradient explosion; too small (saturation) causes gradient vanishing.

**2. Derivative range.** The range of $\sigma'(z)$ as $z$ varies over $\mathbb{R}$. For gradient flow, we want $\sigma'(z) \approx 1$ for most inputs — this ensures gradients neither explode nor vanish through each layer.

**3. Fixed points.** A fixed point satisfies $\sigma(z^*) = z^*$. SELU has a fixed point at $(0, 0)$ which is central to its self-normalising property. Understanding fixed points helps analyse the equilibrium distribution of activations.

**4. Computational cost.** Relative to a multiply-accumulate (MAC) operation:
- ReLU: ~0 extra cost (comparison only)
- Leaky ReLU: ~0 extra cost
- Sigmoid/tanh: ~4–8 MACs (exp computation)
- GELU (exact): ~8–12 MACs (Gaussian CDF)
- GELU (approx.): ~6 MACs (tanh approximation)
- Swish/SiLU: ~4–6 MACs (sigmoid sub-computation)

**5. Output range.** Bounded outputs (sigmoid, tanh) can aid stability in recurrent networks; unbounded outputs (ReLU, GELU, Swish) are generally preferred for feedforward networks as they do not suppress large activations.


---

## 3. Classical Activations: Sigmoid and Tanh

### 3.1 Sigmoid

The sigmoid function is:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Range:** $(0, 1)$. **Derivative:**

$$\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2} = \sigma(z)(1 - \sigma(z))$$

This derivative has a maximum of $1/4$ at $z = 0$, attained because $\sigma(0) = 1/2$ and $(1/2)(1/2) = 1/4$. The derivative approaches zero in both tails: $\sigma'(z) \to 0$ as $z \to \pm\infty$.

**Probabilistic interpretation:** $\sigma(z)$ is the probability that a logistic random variable with location $0$ and scale $1$ is less than $z$. In logistic regression, $P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b)$ is the natural parameterisation of a Bernoulli distribution. This makes sigmoid the canonical output activation for binary classification.

**Numerical stability:** Computing $\sigma(z)$ directly for very negative $z$ leads to $0/1 = 0$ (stable), but for very large $z$, computing $e^z$ overflows. The numerically stable form is:
$$\sigma(z) = \begin{cases} \frac{1}{1+e^{-z}} & z \ge 0 \\ \frac{e^z}{1+e^z} & z < 0 \end{cases}$$

**Properties at a glance:**
- Bounded: $\sigma(z) \in (0, 1)$
- Lipschitz constant: $K = 1/4$ (derivative bounded by $1/4$)
- Not zero-centred: all outputs positive
- Smooth: $C^\infty$
- Inverse: $\sigma^{-1}(p) = \log(p/(1-p))$ (logit function)

**Non-examples (common misuses):**
- Using sigmoid as hidden activation in deep networks: saturates and kills gradients
- Interpreting $\sigma(\mathbf{W}\mathbf{x})$ as multiclass probabilities: not normalised to sum to 1 (use softmax instead)
- Using sigmoid output for regression: artificially constrains output to $(0,1)$

### 3.2 Hyperbolic Tangent

The hyperbolic tangent is:

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \frac{e^{2z} - 1}{e^{2z} + 1}$$

**Relation to sigmoid:**

$$\tanh(z) = 2\sigma(2z) - 1$$

This is immediate: $2\sigma(2z) - 1 = 2/(1+e^{-2z}) - 1 = (2 - 1 - e^{-2z})/(1+e^{-2z}) = (1 - e^{-2z})/(1 + e^{-2z}) = \tanh(z)$.

**Derivative:**

$$\tanh'(z) = 1 - \tanh^2(z) = \operatorname{sech}^2(z)$$

Maximum value of 1 at $z = 0$. Bounded in $[0, 1]$. Still approaches 0 in both tails.

**Zero-centred advantage:** Unlike sigmoid, tanh has range $(-1, 1)$ and outputs zero at $z = 0$. This means the gradient $\partial \mathcal{L}/\partial W = \delta \cdot \mathbf{a}^\top$ can have both positive and negative components, avoiding the zig-zag gradient problem of sigmoid. LeCun et al. (1998) demonstrated experimentally that tanh converges significantly faster than sigmoid as a hidden-layer activation.

**For AI:** Tanh is the standard activation in LSTM gates and GRU hidden state transformations. The LSTM cell update uses $\tanh$ to maintain outputs in $(-1, 1)$, which bounds the hidden state and helps with gradient flow through many time steps. However, even tanh saturates for large inputs, which is why LSTMs still struggle with very long sequences (motivating transformers with attention).

**Higher derivatives:**
$$\tanh''(z) = -2\tanh(z)\operatorname{sech}^2(z)$$
$$\tanh'''(z) = -2\operatorname{sech}^2(z)(1 - 3\tanh^2(z))$$

The second derivative being zero at $z = 0$ makes tanh locally quadratic around the origin, which is exploited by some second-order optimisation methods.

### 3.3 Vanishing Gradient Analysis

Consider a deep network with $L$ layers, each using the same activation $\sigma$. The gradient of the loss with respect to the pre-activation at layer $l$ (the "error signal" $\delta^{[l]}$) is:

$$\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} = \left(\delta^{[l+1]} \cdot W^{[l+1]}\right) \odot \sigma'(\mathbf{z}^{[l]})$$

Unrolling through all $L$ layers:

$$\delta^{[1]} = \delta^{[L]} \cdot \prod_{l=2}^{L} W^{[l]} \cdot \prod_{l=1}^{L-1} \operatorname{diag}(\sigma'(\mathbf{z}^{[l]}))$$

The scalar gradient at layer 1 is bounded by:

$$\lVert \delta^{[1]} \rVert \le \lVert \delta^{[L]} \rVert \cdot \prod_{l=2}^{L} \lVert W^{[l]} \rVert_2 \cdot \prod_{l=1}^{L-1} \max_j \lvert \sigma'(z_j^{[l]}) \rvert$$

**For sigmoid:** $\max_j \lvert \sigma'(z_j^{[l]}) \rvert \le 1/4$. If the weights are initialised with $\lVert W^{[l]} \rVert_2 \approx 1$ (e.g., orthogonal initialisation), the gradient magnitude is bounded by $(1/4)^{L-1}$.

**Numerical impact:**
- $L = 5$: gradient $\le (1/4)^4 \approx 0.0039$ — 250× attenuation
- $L = 10$: gradient $\le (1/4)^9 \approx 3.8 \times 10^{-6}$ — 260,000× attenuation
- $L = 20$: gradient $\le (1/4)^{19} \approx 3.6 \times 10^{-12}$ — early layers receive essentially zero gradient

This quantifies why sigmoid networks beyond ~5 layers were intractable before ReLU and residual connections. The vanishing gradient problem is not a numerical precision issue — it is a fundamental mathematical consequence of repeated multiplication by numbers less than 1.

**For tanh:** $\max_j \lvert \tanh'(z) \rvert = 1$ at $z = 0$, but quickly drops: $\tanh'(\pm 1) \approx 0.42$, $\tanh'(\pm 2) \approx 0.07$. For zero-initialised networks, tanh avoids vanishing in the first epoch, but saturated neurons quickly push the effective max derivative far below 1.

**Mitigation strategies discovered historically:**
1. **Unsupervised pre-training** (Hinton & Salakhutdinov, 2006): layer-wise RBM pre-training put weights in the non-saturating regime
2. **Careful initialisation** (Glorot & Bengio, 2010): variance formula to maintain gradient variance = 1
3. **ReLU activation** (Glorot et al., 2011): gradient is either 0 or 1, not squeezed by saturation
4. **Batch normalisation** (Ioffe & Szegedy, 2015): normalises pre-activations to prevent saturation
5. **Residual connections** (He et al., 2016): additive skip connections bypass the multiplicative chain

### 3.4 When to Use Classical Activations

Despite their limitations for hidden layers, sigmoid and tanh remain essential in specific contexts:

**Sigmoid (appropriate uses):**
- Binary classification output layer: $P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{z})$
- LSTM/GRU gates: forget gate $f_t = \sigma(W_f \mathbf{h}_{t-1} + U_f \mathbf{x}_t + b_f)$; bounded output in $(0,1)$ is semantically meaningful as a "proportion to forget"
- Attention gates in older architectures
- Calibration: Platt scaling $P(y=1 \mid s) = \sigma(as + b)$ where $s$ is a raw score

**Tanh (appropriate uses):**
- LSTM cell state update: $\tilde{C}_t = \tanh(W_c \mathbf{h}_{t-1} + U_c \mathbf{x}_t + b_c)$; zero-centred dynamics
- Actor network output in RL when action space is $[-1, 1]$
- Feature normalisation in learned embeddings (bounding extreme values)
- Physics-informed NNs where the true function is bounded


---

## 4. The ReLU Family

### 4.1 Rectified Linear Unit

The Rectified Linear Unit (ReLU), introduced by Glorot, Bordes & Bengio (2011), is:

$$\operatorname{ReLU}(z) = \max(0, z) = \begin{cases} z & z > 0 \\ 0 & z \le 0 \end{cases}$$

**Derivative:**

$$\operatorname{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}$$

At $z = 0$, the derivative is undefined in the classical sense, but in practice any value in $[0, 1]$ is used (subgradient). Most frameworks use $\operatorname{ReLU}'(0) = 0$.

**Why ReLU transformed deep learning:**

1. **Non-saturating positive regime:** for $z > 0$, the gradient is exactly 1 — no attenuation regardless of the magnitude of $z$. A chain of $L$ ReLU layers contributes gradient factors of either 0 or 1, so the product $\prod_{l=1}^{L-1} \sigma'(z^{[l]})$ is either 0 (if any neuron is in the zero regime) or 1 (if all neurons are active).

2. **Sparsity induction:** approximately 50% of neurons output zero (for inputs centred near 0), creating sparse representations. Sparse activations have information-theoretic advantages (efficient coding) and computational benefits (sparse matrix-vector products).

3. **Computational efficiency:** $\max(0, z)$ is a single comparison operation, far cheaper than exponentials in sigmoid/tanh. For inference on CPU/GPU, this matters significantly at scale.

4. **Biological plausibility:** firing-rate models of neurons show a roughly linear response above threshold, consistent with ReLU shape. This inspired the connection to the biological motivation, though modern deep learning has largely moved beyond biological plausibility as a design criterion.

**Properties:**
- Lipschitz constant: $K = 1$ (derivative bounded by 1)
- Range: $[0, \infty)$ — not zero-centred
- Not bounded above — outputs grow without bound
- Not smooth at $z = 0$ (piecewise linear, $C^0$ only)
- Positive homogeneous: $\operatorname{ReLU}(\lambda z) = \lambda \operatorname{ReLU}(z)$ for $\lambda \ge 0$

### 4.2 The Dying ReLU Problem

A neuron "dies" when it becomes permanently inactive — outputting zero for every input in the training set. Once dead, its gradient is zero for all inputs, and the gradient descent update for its incoming weights is zero. The neuron can never recover through gradient-based learning.

**Formal condition for a dead neuron:** Neuron $j$ at layer $l$ is dead if:

$$\forall \mathbf{x} \in \mathcal{D}: \quad W^{[l]}_{j,:} \, \mathbf{a}^{[l-1]}(\mathbf{x}) + b^{[l]}_j \le 0$$

where $\mathcal{D}$ is the training distribution.

**When does it happen?**

1. **Large learning rate:** a large gradient step pushes the weight vector $W^{[l]}_{j,:}$ such that the pre-activation becomes negative for all training inputs in one step
2. **Negative bias initialisation:** if $b^{[l]}_j$ is initialised too negatively, the neuron starts dead
3. **Large L2 regularisation:** weight decay shrinks weights toward zero, which can push pre-activations negative when the data has non-zero mean
4. **Gradient clipping with wrong threshold:** clipping can distort the direction of updates

**Empirical prevalence:** In networks without batch normalisation, dead neuron fractions of 10–40% are common after training with standard SGD. In networks with batch normalisation, the pre-activation distribution is normalised near zero, reducing the dead fraction significantly.

**Mitigation:**
- Leaky ReLU / PReLU: non-zero gradient for $z < 0$
- He initialisation: variance $2/n_{\text{in}}$ reduces probability of negative initial pre-activations
- Batch normalisation: normalises pre-activations
- Careful learning rate scheduling
- ELU/SELU: smooth, non-zero negative regime

### 4.3 Leaky ReLU and PReLU

**Leaky ReLU** (Maas et al., 2013) introduces a small fixed slope for negative inputs:

$$\operatorname{LReLU}(z; \alpha) = \begin{cases} z & z > 0 \\ \alpha z & z \le 0 \end{cases}$$

where $\alpha \in (0, 1)$, typically $\alpha = 0.01$. The derivative is:

$$\operatorname{LReLU}'(z; \alpha) = \begin{cases} 1 & z > 0 \\ \alpha & z \le 0 \end{cases}$$

Gradient is never zero, so neurons cannot die. Lipschitz constant is 1. Still not zero-centred.

**Parametric ReLU (PReLU)** (He et al., 2015) treats $\alpha$ as a learned parameter:

$$\operatorname{PReLU}(z; \alpha) = \max(\alpha z, z)$$

The gradient with respect to $\alpha$ is:

$$\frac{\partial \operatorname{PReLU}}{\partial \alpha} = \begin{cases} 0 & z > 0 \\ z & z \le 0 \end{cases}$$

During training, $\alpha$ is learned via backpropagation like any other parameter, with a separate learning rate and no weight decay. The initialisation $\alpha = 0.25$ is typical. PReLU converges to approximately $\alpha \approx 0.01$–$0.25$ depending on the architecture and task.

**For AI:** PReLU was used in He et al.'s ImageNet classification networks (2015). Modern LLMs do not typically use PReLU due to the extra parameter and the dominance of GELU/SwiGLU, but it remains relevant in computer vision architectures like ResNet variants.

### 4.4 ELU and SELU

**Exponential Linear Unit (ELU)** (Clevert, Unterthiner & Hochreiter, 2015):

$$\operatorname{ELU}(z; \alpha) = \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \le 0 \end{cases}$$

where $\alpha > 0$ (typically $\alpha = 1$). The derivative is:

$$\operatorname{ELU}'(z; \alpha) = \begin{cases} 1 & z > 0 \\ \alpha e^z & z \le 0 \end{cases}$$

Note that $\operatorname{ELU}'(0^-) = \alpha$ and $\operatorname{ELU}'(0^+) = 1$. For $\alpha = 1$ the derivative is discontinuous at 0 (left limit = 1, right limit = 1, but the function itself is $C^1$ since $\operatorname{ELU}(0^-) = \alpha(e^0 - 1) = 0$ and $\operatorname{ELU}(0^+) = 0$). ELU is $C^1$ but not $C^2$ at $z = 0$ when $\alpha = 1$.

**Key property:** ELU pushes mean activations towards zero from the negative regime. As $z \to -\infty$, $\operatorname{ELU}(z) \to -\alpha$. This soft lower bound means the mean output of a layer can be close to zero without requiring centred inputs.

**Scaled ELU (SELU)** (Klambauer et al., 2016) is remarkable: it is the unique scaling of ELU that makes the $\mathcal{N}(0,1)$ distribution a fixed point under the activation.

$$\operatorname{SELU}(z) = \lambda \begin{cases} z & z > 0 \\ \alpha(e^z - 1) & z \le 0 \end{cases}$$

The constants $\lambda$ and $\alpha$ are determined by requiring that if the pre-activation $z \sim \mathcal{N}(0,1)$, then the output $\operatorname{SELU}(z)$ also has mean 0 and variance 1. Solving the fixed-point equations:

$$\mathbb{E}[\operatorname{SELU}(z)] = 0 \quad \text{and} \quad \operatorname{Var}[\operatorname{SELU}(z)] = 1 \quad \text{when } z \sim \mathcal{N}(0,1)$$

yields the values:
$$\alpha \approx 1.6732632423543772$$
$$\lambda \approx 1.0507009873554805$$

With SELU activations and lecun-normal initialisation (weights from $\mathcal{N}(0, 1/n_{\text{in}})$), the activations in each layer converge to a standard normal distribution, making explicit batch normalisation unnecessary. This "self-normalising" property was proved rigorously using fixed-point theory of iterated function systems.

### 4.5 Gradient Comparison Across ReLU Variants

The fundamental difference across the ReLU family is behaviour in the negative input regime:

| Activation | Gradient ($z < 0$) | Gradient ($z = 0$) | Gradient ($z > 0$) |
|---|---|---|---|
| ReLU | 0 | undefined (0 by convention) | 1 |
| Leaky ReLU | $\alpha$ (~0.01) | $\alpha$ | 1 |
| PReLU | $\alpha$ (learned) | $\alpha$ | 1 |
| ELU ($\alpha=1$) | $e^z \in (0,1)$ | 1 | 1 |
| SELU | $\lambda \alpha e^z$ | $\lambda$ | $\lambda \approx 1.05$ |

For gradient flow analysis, the key observation is that ELU and SELU have **exponentially decaying** gradients in the negative regime ($e^z \to 0$ as $z \to -\infty$), while Leaky ReLU has a constant positive gradient. Both avoid dying neurons, but with different gradient magnitudes.


---

## 5. Modern Smooth Activations

### 5.1 GELU

The **Gaussian Error Linear Unit (GELU)** was introduced by Hendrycks & Gimpel (2016) and became the default hidden-layer activation in BERT (2018), GPT-2, GPT-3, and most large language models.

**Definition:**

$$\operatorname{GELU}(z) = z \cdot \Phi(z)$$

where $\Phi(z) = P(X \le z) = \frac{1}{2}\left[1 + \operatorname{erf}\!\left(\frac{z}{\sqrt{2}}\right)\right]$ is the CDF of the standard normal distribution $\mathcal{N}(0,1)$.

**Stochastic interpretation:** GELU has a remarkable probabilistic interpretation. Consider a random variable $X \sim \mathcal{N}(0,1)$. Then:

$$\operatorname{GELU}(z) = z \cdot P(X \le z) = \mathbb{E}_{X \sim \mathcal{N}(0,1)}[z \cdot \mathbf{1}[X \le z]]$$

This is the expected value of the deterministic activation $z \cdot \mathbf{1}[X \le z]$ — a stochastic ReLU that keeps the input $z$ with probability $\Phi(z)$ and zeros it out with probability $1 - \Phi(z)$. Higher-value inputs are more likely to be retained, connecting GELU to dropout-style regularisation.

**Derivative:**

$$\operatorname{GELU}'(z) = \Phi(z) + z \cdot \phi(z)$$

where $\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$ is the standard normal PDF. Note that $\operatorname{GELU}'(z) > 0$ for most $z$ (non-monotone: has a small negative region near $z \approx -0.4$ where $\operatorname{GELU}'(z) < 0$).

**Practical approximation (used in BERT, GPT-2):**

$$\operatorname{GELU}(z) \approx 0.5z \left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\left(z + 0.044715 z^3\right)\right)\right)$$

This approximation uses only tanh (which is faster than computing $\operatorname{erf}$) and matches the exact GELU to within $10^{-3}$ absolute error across $z \in [-5, 5]$.

**Key numerical properties:**
- Minimum value: $\operatorname{GELU}(z) \ge \operatorname{GELU}(-0.751) \approx -0.170$ — bounded below
- Not bounded above: $\operatorname{GELU}(z) \approx z$ for large $z$
- Lipschitz constant: $K \approx 1.13$ (gradient slightly exceeds 1 near $z = 1$)
- $\operatorname{GELU}(0) = 0$, $\operatorname{GELU}'(0) = 0.5$

### 5.2 Swish / SiLU

**Swish** (Ramachandran, Zoph & Le, 2017) was discovered by automated neural architecture search (NAS) across 1,000+ activation functions. It consistently outperformed ReLU and GELU on ResNet and Inception architectures.

$$\operatorname{Swish}(z; \beta) = z \cdot \sigma(\beta z) = \frac{z}{1 + e^{-\beta z}}$$

where $\beta > 0$ is either fixed (typically $\beta = 1$) or learned. The $\beta = 1$ case is called **SiLU** (Sigmoid Linear Unit), which is the form used in most modern LLMs.

**Derivative:**

$$\operatorname{Swish}'(z; \beta) = \beta \operatorname{Swish}(z; \beta) + \sigma(\beta z)(1 - \beta \operatorname{Swish}(z;\beta))$$

For $\beta = 1$:
$$\operatorname{SiLU}'(z) = \sigma(z) + z \sigma(z)(1 - \sigma(z)) = \sigma(z)(1 + z(1 - \sigma(z)))$$

**Non-monotone behaviour:** For $\beta = 1$, $\operatorname{SiLU}'(z) = 0$ at $z \approx -1.28$. For $z < -1.28$, the gradient is negative — the function decreases as $z$ increases. This non-monotone "bump" in the negative regime allows the activation to distinguish between "strongly negative" and "moderately negative" inputs, a capacity that ReLU completely lacks.

**Limiting behaviour:**
- As $\beta \to 0$: $\operatorname{Swish}(z; \beta) \to z/2$ (linear)
- As $\beta \to \infty$: $\operatorname{Swish}(z; \beta) \to \operatorname{ReLU}(z)$
- $\beta = 1$ (SiLU): smooth interpolation between linear and ReLU

**Connection to GELU:** Both GELU and SiLU are of the form $z \cdot g(z)$ where $g$ is a "gating" function increasing from 0 to 1. For GELU, $g(z) = \Phi(z)$ (normal CDF); for SiLU, $g(z) = \sigma(z)$ (sigmoid). These are both cumulative distributions — normal and logistic, respectively.

### 5.3 Mish

**Mish** (Misra, 2019) is:

$$\operatorname{Mish}(z) = z \cdot \tanh(\operatorname{softplus}(z)) = z \cdot \tanh(\log(1 + e^z))$$

**Derivative:**

$$\operatorname{Mish}'(z) = \tanh(\operatorname{softplus}(z)) + z \cdot \operatorname{sech}^2(\operatorname{softplus}(z)) \cdot \sigma(z)$$

where $\sigma(z) = e^z/(1+e^z)$ is the sigmoid (derivative of softplus).

**Properties:**
- Smooth ($C^\infty$): both tanh and softplus are smooth
- Bounded below: $\operatorname{Mish}(z) \ge \operatorname{Mish}(-0.621) \approx -0.308$
- Not bounded above: $\operatorname{Mish}(z) \approx z$ for large $z$
- Non-monotone: decreasing for $z \in (-\infty, -0.621)$
- Lipschitz constant: $K \approx 1.07$

Mish was found to outperform ReLU, Swish, and GELU on certain computer vision benchmarks (YOLOv4 uses Mish), but has seen limited adoption in LLMs due to higher computational cost than SiLU.

### 5.4 Why Smooth Non-Monotone Activations Help

Three mechanisms explain the empirical superiority of GELU/Swish over ReLU:

**1. Smoothness and higher-order optimisation.** ReLU is piecewise linear ($C^0$) — its second derivative is zero everywhere except at the kink. This means Newton-type methods that use curvature information (Hessian) see very little signal from the activation function itself. Smooth activations have non-trivial Hessians, enabling better curvature-based updates. In practice, Adam (which approximates the diagonal Hessian) benefits from smooth gradients.

**2. Non-monotone negative regime — implicit regularisation.** ReLU's negative regime is flat zero: any input below zero contributes identically (nothing). GELU and Swish preserve some information from the negative regime — the output of $\operatorname{GELU}(-1) \approx -0.159$ is different from $\operatorname{GELU}(-3) \approx -0.003$. This allows the network to distinguish "slightly negative" (possibly relevant) from "strongly negative" (likely noise), implementing a form of soft filtering.

**3. Gradient identity at large positive values.** For large positive $z$, $\operatorname{GELU}(z) \approx z$ and $\operatorname{SiLU}(z) \approx z$, so the gradient approaches 1. This means large, confident activations propagate gradients with near-unit gain — unlike sigmoid/tanh which saturate to zero gradient.

**For AI:** The consistent advantage of GELU/SwiGLU over ReLU in LLM training (documented in PaLM, LLaMA, and BLOOM ablations) suggests that the qualitative differences above compound over billions of training steps. The softmax attention already provides a form of sparsity (attention weights sum to 1), and the FFN activation provides complementary smoothness — together they give transformers their characteristic inductive biases.


---

## 6. Output Layer Activations

### 6.1 Softmax

Softmax is the canonical output activation for multi-class classification and the attention mechanism. Given a vector $\mathbf{z} \in \mathbb{R}^K$, the softmax output is:

$$\operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

**Properties:**
- Output is a valid probability vector: $\operatorname{softmax}(\mathbf{z})_i \ge 0$ and $\sum_i \operatorname{softmax}(\mathbf{z})_i = 1$
- Invariant to constant shifts: $\operatorname{softmax}(\mathbf{z} + c\mathbf{1}) = \operatorname{softmax}(\mathbf{z})$ for any scalar $c$ — this is the numerical stability trick
- Equivariant to permutations: permuting $\mathbf{z}$ permutes the output in the same way
- Not equivariant to scaling: $\operatorname{softmax}(\lambda \mathbf{z}) \ne \lambda \operatorname{softmax}(\mathbf{z})$ — scaling sharpens (large $\lambda$) or flattens (small $\lambda$)

**Jacobian derivation.** The Jacobian $J \in \mathbb{R}^{K \times K}$ with $(J)_{ij} = \partial s_i / \partial z_j$ where $\mathbf{s} = \operatorname{softmax}(\mathbf{z})$:

For $i = j$:
$$\frac{\partial s_i}{\partial z_i} = \frac{e^{z_i} \sum_k e^{z_k} - e^{z_i} \cdot e^{z_i}}{(\sum_k e^{z_k})^2} = s_i - s_i^2 = s_i(1 - s_i)$$

For $i \ne j$:
$$\frac{\partial s_i}{\partial z_j} = \frac{0 - e^{z_i} e^{z_j}}{(\sum_k e^{z_k})^2} = -s_i s_j$$

Compactly: $J = \operatorname{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$. This can be written as $J = S(I - S)$ where $S = \operatorname{diag}(\mathbf{s})$.

**Gradient of cross-entropy through softmax.** In classification with target $y$ (one-hot vector $\mathbf{e}_y$), the loss is:

$$\mathcal{L} = -\log \operatorname{softmax}(\mathbf{z})_y = -z_y + \log\sum_j e^{z_j}$$

The gradient with respect to $\mathbf{z}$ is:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \operatorname{softmax}(\mathbf{z}) - \mathbf{e}_y = \hat{\mathbf{p}} - \mathbf{e}_y$$

This beautiful result — the gradient is just the predicted probability vector minus the one-hot target — is why cross-entropy + softmax is the universal classification loss. The gradient at position $i$ is:
- $s_i - 1 < 0$ for the true class: gradient pushes the logit up
- $s_j - 0 = s_j > 0$ for wrong classes: gradient pushes logits down, proportional to their current probability

**Multinomial logistic regression.** In the two-class case ($K = 2$), $\operatorname{softmax}(\mathbf{z})_1 = e^{z_1}/(e^{z_1} + e^{z_2}) = \sigma(z_1 - z_2)$. The softmax reduces to sigmoid on the difference of logits, confirming that binary classification with softmax is equivalent to logistic regression.

### 6.2 Log-Softmax and Numerical Stability

Computing $\log \operatorname{softmax}(\mathbf{z})$ naively as $\log(e^{z_i}/\sum_j e^{z_j})$ is numerically catastrophic for large $\lvert z_i \rvert$:
- If $z_i = 1000$: $e^{1000}$ overflows IEEE 754 float64 ($\approx 10^{308}$ limit)
- If $z_i = -1000$: $e^{-1000} \approx 5 \times 10^{-435}$, underflows to 0

The **log-sum-exp trick** resolves both issues. Let $m = \max_j z_j$. Then:

$$\log \sum_j e^{z_j} = m + \log \sum_j e^{z_j - m}$$

Since $z_j - m \le 0$, all exponentials are at most 1, preventing overflow. The shift by $m$ is cancelled in the log, giving the exact result. The numerically stable log-softmax is:

$$\log \operatorname{softmax}(\mathbf{z})_i = z_i - m - \log \sum_j e^{z_j - m}$$

**Cross-entropy via log-softmax.** In PyTorch, `nn.CrossEntropyLoss` fuses softmax and NLL loss, computing log-softmax internally with the numerical stability trick. Calling `softmax()` then `log()` then `NLLLoss` separately invites numerical instability and is always wrong in production code.

**For AI:** The log-sum-exp trick is ubiquitous in transformer inference:
- Attention computation: $\operatorname{softmax}(QK^\top/\sqrt{d_k})$ uses max-subtraction for stability
- FlashAttention (Dao et al., 2022): tile-wise computation requires maintaining a running log-sum-exp to correctly accumulate attention scores without materialising the full $N \times N$ attention matrix
- Language model perplexity: $\operatorname{PPL} = \exp(-\frac{1}{T}\sum_{t=1}^T \log p(x_t \mid x_{<t}))$ uses log-domain accumulation

### 6.3 Sparsemax

**Sparsemax** (Martins & Astudillo, 2016) is an alternative to softmax that produces sparse probability distributions — many outputs are exactly zero, not just close to zero.

$$\operatorname{sparsemax}(\mathbf{z}) = \arg\min_{\mathbf{p} \in \Delta^{K-1}} \lVert \mathbf{p} - \mathbf{z} \rVert_2^2$$

where $\Delta^{K-1} = \{\mathbf{p} \in \mathbb{R}^K : p_i \ge 0, \sum_i p_i = 1\}$ is the probability simplex. Sparsemax is the Euclidean projection of $\mathbf{z}$ onto the simplex, whereas softmax is the entropic projection.

**Computation:** Sort $z_1 \ge z_2 \ge \cdots \ge z_K$. Find $k(\mathbf{z}) = \max\{k : 1 + kz_k > \sum_{j \le k} z_j\}$. Set $\tau(\mathbf{z}) = (\sum_{j \le k} z_j - 1)/k$. Then $\operatorname{sparsemax}(\mathbf{z})_i = \max(z_i - \tau, 0)$.

**Gradient:** When $p_i > 0$ (active support): $\partial p_i / \partial z_j = (\mathbf{I} - \mathbf{1}\mathbf{1}^\top/|S|)_{ij}$ where $S$ is the support set; when $p_i = 0$: gradient is zero. This is a projection onto the active simplex face.

**Applications:** Sparse attention in NLP allows exact zero attention weights (not just near-zero as with softmax), enabling interpretable attention patterns. Used in Transformers with structured predictions.

### 6.4 Temperature Scaling

Temperature scaling modifies softmax by dividing logits by a temperature $\tau > 0$:

$$\operatorname{softmax}(\mathbf{z}/\tau)_i = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}$$

**Limiting behaviour:**
- $\tau \to 0$: $\operatorname{softmax}(\mathbf{z}/\tau) \to \operatorname{one\_hot}(\arg\max_i z_i)$ — hard maximum
- $\tau \to \infty$: $\operatorname{softmax}(\mathbf{z}/\tau) \to \mathbf{1}/K$ — uniform distribution
- $\tau = 1$: standard softmax

**Entropy as a function of $\tau$:** The entropy $H[\operatorname{softmax}(\mathbf{z}/\tau)]$ is monotonically increasing in $\tau$. High temperature = high entropy = more uniform = more exploratory. Low temperature = low entropy = more peaked = more deterministic.

**Calibration:** Neural networks trained with cross-entropy tend to be overconfident — their predicted probabilities are too extreme (Guo et al., 2017). **Temperature scaling** post-training fixes this: find the $\tau > 1$ that minimises the negative log-likelihood on a held-out validation set, then use this $\tau$ at test time. For large models like GPT-4, $\tau$ is typically tuned between 0.5 and 2.0 depending on the generation task.

**For AI:** Temperature is a key generation hyperparameter:
- Code generation: $\tau = 0.2$–$0.5$ (low temperature for determinism)
- Creative writing: $\tau = 0.8$–$1.2$ (higher temperature for diversity)
- Nucleus sampling (top-$p$): samples from the minimal set of tokens whose probabilities sum to $p$ under $\operatorname{softmax}(\mathbf{z}/\tau)$
- Attention: $\operatorname{Attention}(Q,K,V) = \operatorname{softmax}(QK^\top/\sqrt{d_k})V$ — the $1/\sqrt{d_k}$ factor is a temperature, preventing the dot products from growing large and making softmax too peaked


---

## 7. Gated Linear Units and Transformer FFN Variants

### 7.1 Gated Linear Unit

**Gated Linear Units (GLU)** were introduced by Dauphin et al. (2017) for language modelling with convolutional networks. The basic form is:

$$\operatorname{GLU}(\mathbf{x}, W, V, \mathbf{b}, \mathbf{c}) = (\mathbf{x}W + \mathbf{b}) \odot \sigma(\mathbf{x}V + \mathbf{c})$$

where $W, V \in \mathbb{R}^{d \times m}$, $\mathbf{b}, \mathbf{c} \in \mathbb{R}^m$, and $\odot$ denotes element-wise (Hadamard) product. The first component $\mathbf{x}W + \mathbf{b}$ is the "content" path; the second $\sigma(\mathbf{x}V + \mathbf{c})$ is the "gate" — a vector in $(0,1)^m$ that selectively passes or suppresses each dimension of the content.

**Gradient via product rule.** The gradient of the GLU output with respect to the content $\mathbf{u} = \mathbf{x}W + \mathbf{b}$ and gate $\mathbf{g} = \sigma(\mathbf{x}V + \mathbf{c})$ is:

$$\frac{\partial \operatorname{GLU}}{\partial \mathbf{u}} = \mathbf{g}, \qquad \frac{\partial \operatorname{GLU}}{\partial \mathbf{g}} = \mathbf{u}$$

The gradient flows back through both paths. Crucially, when the gate $g_j \approx 1$, the gradient of the content path is unattenuated. When $g_j \approx 0$, the gradient is suppressed — the gate acts as a learned gradient mask.

**Motivation:** In language modelling, the gate learns which input features are relevant for the current token. The sigmoid gate approximates a binary decision: "is this dimension of the intermediate representation informative?" This is a form of feature selection learned end-to-end.

### 7.2 SwiGLU

**SwiGLU** (Shazeer, 2020) replaces the sigmoid gate in GLU with the Swish/SiLU activation:

$$\operatorname{SwiGLU}(\mathbf{x}, W_1, W_2, W_3) = \operatorname{Swish}(\mathbf{x}W_1) \odot (\mathbf{x}W_2)$$

In practice, without separate biases and using $\operatorname{Swish}(z) = z\sigma(z) = \operatorname{SiLU}(z)$:

$$\operatorname{SwiGLU}(\mathbf{x}) = (\mathbf{x}W_1 \odot \sigma(\mathbf{x}W_1)) \odot (\mathbf{x}W_2)$$

where the first factor is $\operatorname{SiLU}(\mathbf{x}W_1)$ and the second is the linear gate $\mathbf{x}W_2$.

**Why SwiGLU outperforms standard FFN:**

1. **Multiplicative interaction.** The element-wise product creates multiplicative interactions between the two projections of the input. This allows the FFN to represent multiplicative feature combinations, which pure additive FFNs cannot.

2. **Smooth gating.** Swish (unlike sigmoid) has a non-zero gradient everywhere, including the negative regime. This allows the gate to be learned more smoothly and avoids the gradient vanishing problem at gate=0.

3. **Implicit normalisation.** The SiLU gate outputs values in roughly $[-0.3, \infty)$, with a distribution that is approximately zero-centred for inputs near zero. This provides a form of implicit activation normalisation.

**Empirical evidence:** Shazeer's original paper showed that SwiGLU outperformed ReLU, GeLU, and ReGLU (with ReLU gating) on T5 and GPT-style models, achieving lower perplexity with the same parameter count. The improvement was consistent across scales from 100M to 1B parameters.

### 7.3 GeGLU

**GeGLU** replaces the Swish gate with GELU:

$$\operatorname{GeGLU}(\mathbf{x}) = \operatorname{GELU}(\mathbf{x}W_1) \odot (\mathbf{x}W_2)$$

Used in T5 v1.1 (Google, 2020), GPT-NeoX (EleutherAI, 2022), and several BLOOM variants. Empirically, GeGLU and SwiGLU perform comparably, with SwiGLU having a slight edge due to lower computational cost (SiLU is faster than GELU because it avoids the erf computation).

**ReGLU (ReLU GLU) and BiGLU (Bilinear)** are the simplest variants:
$$\operatorname{ReGLU}(\mathbf{x}) = \operatorname{ReLU}(\mathbf{x}W_1) \odot (\mathbf{x}W_2)$$
$$\operatorname{BiGLU}(\mathbf{x}) = (\mathbf{x}W_1) \odot (\mathbf{x}W_2)$$

BiGLU (fully bilinear, no activation) still outperforms standard FFN due to multiplicative interactions, but underperforms gated variants. ReGLU sits between BiGLU and SwiGLU/GeGLU.

### 7.4 Parameter Budget in Gated FFN

A standard transformer FFN with hidden dimension $d$ and intermediate dimension $4d$ uses:

$$\text{Params}_{\text{standard}} = 2 \times (d \times 4d) = 8d^2 \quad \text{(two weight matrices)}$$

A gated FFN (SwiGLU/GeGLU) with the same hidden dimension $d$ but two projection matrices to the intermediate dimension $m$ uses:

$$\text{Params}_{\text{gated}} = 3 \times (d \times m) \quad \text{(three matrices: } W_1, W_2, W_3\text{)}$$

To match the parameter count of the standard FFN:
$$3dm = 8d^2 \implies m = \frac{8d}{3} \approx 2.667d$$

**LLaMA hidden dimension formula.** LLaMA sets the intermediate dimension to:

$$m = \frac{2}{3} \times 4d = \frac{8d}{3}$$

rounded up to the nearest multiple of 256 for hardware efficiency. For LLaMA-7B with $d = 4096$:

$$m = \lceil (8 \times 4096/3) / 256 \rceil \times 256 = \lceil 10922.7 / 256 \rceil \times 256 = 43 \times 256 = 11008$$

This explains why LLaMA's FFN has intermediate dimension 11008 (not 16384 as in standard BERT-like models).

**For AI:** Understanding the parameter budget is critical for model analysis and compression. Gated FFN blocks account for roughly 2/3 of all parameters in LLaMA (the remaining 1/3 is in attention projection matrices). When fine-tuning with LoRA (Hu et al., 2022), the FFN weight matrices $W_1, W_2, W_3$ are the primary targets because they contain the model's stored "knowledge" (Geva et al., 2021).


---

## 8. Activation Properties: Theory

### 8.1 Lipschitz Continuity

**Definition.** A function $f: \mathbb{R} \to \mathbb{R}$ is Lipschitz continuous with constant $K$ if:

$$\lvert f(a) - f(b) \rvert \le K \lvert a - b \rvert \quad \forall a, b \in \mathbb{R}$$

For differentiable functions, the Lipschitz constant equals the supremum of the absolute derivative: $K = \sup_z \lvert f'(z) \rvert$.

**Lipschitz constants for standard activations:**

| Activation | $\sup_z \lvert \sigma'(z) \rvert$ | Achieved at |
|---|---|---|
| Sigmoid $\sigma$ | $1/4$ | $z = 0$ |
| Tanh | $1$ | $z = 0$ |
| ReLU | $1$ | $z > 0$ (entire positive half) |
| Leaky ReLU ($\alpha=0.01$) | $1$ | $z > 0$ |
| ELU ($\alpha=1$) | $1$ | $z \ge 0$ |
| SELU | $\lambda \approx 1.051$ | $z \ge 0$ |
| GELU | $\approx 1.13$ | $z \approx 0.6$ |
| SiLU | $\approx 1.10$ | $z \approx 0.5$ |
| Softmax | $1$ | (per output dimension) |

**Why Lipschitz constant matters:**

1. **Spectral norm and stability.** For a layer $\mathbf{a} = \sigma(W\mathbf{x})$, the operator norm of the layer satisfies $\lVert \mathbf{a}(\mathbf{x}_1) - \mathbf{a}(\mathbf{x}_2) \rVert \le K_\sigma \lVert W \rVert_2 \lVert \mathbf{x}_1 - \mathbf{x}_2 \rVert$. The product of Lipschitz constants across layers bounds the Lipschitz constant of the whole network.

2. **Adversarial robustness.** A network with bounded Lipschitz constant is robust to input perturbations: a perturbation $\delta$ in input causes at most $K^L \lVert \delta \rVert$ change in output (where $K$ includes both weight and activation Lipschitz constants). Spectral normalisation (Miyato et al., 2018) constrains $\lVert W \rVert_2 = 1$ per layer to control this.

3. **GAN discriminator stability.** WGAN-GP (Gulrajani et al., 2017) requires the discriminator to be 1-Lipschitz. The activation's Lipschitz constant contributes to the total discriminator Lipschitz bound.

4. **Gradient explosion.** If $K > 1$ (GELU, SELU), gradients can grow slightly at each layer. For very deep networks, this can cause instability if not controlled by careful initialisation.

### 8.2 Universal Approximation Theorem

**Theorem (Cybenko, 1989).** Let $\sigma$ be a continuous, bounded, non-constant function (e.g., sigmoid). Then the set of functions:

$$\left\{ \mathbf{x} \mapsto \sum_{j=1}^N \alpha_j \, \sigma(\mathbf{w}_j^\top \mathbf{x} + b_j) : N \in \mathbb{N},\, \alpha_j, b_j \in \mathbb{R},\, \mathbf{w}_j \in \mathbb{R}^d \right\}$$

is dense in $C(K)$ (continuous functions on compact $K \subset \mathbb{R}^d$) with respect to the uniform norm.

**Theorem (Hornik, 1991).** The result holds for any bounded, non-polynomial $\sigma$.

**Theorem (Leshno et al., 1993).** The result holds for any locally bounded, piecewise continuous $\sigma$ that is not a polynomial — this explicitly includes ReLU.

**What the UAT guarantees:**
- Any continuous function on a compact domain can be approximated by a shallow (one-hidden-layer) network with sufficient width
- The approximation is uniform — the error is bounded everywhere on the domain
- No constraint on the structure of the function (it need not be smooth or convex)

**What the UAT does NOT guarantee:**
- How many neurons $N$ are needed (could be exponential in $d$)
- That gradient descent will find the approximating network
- Anything about the sample complexity (how much data is needed)
- Anything about approximation with bounded depth vs. width

**Depth vs. width.** For ReLU networks, Telgarsky (2016) showed that there exist functions that can be approximated with depth $L$ but require exponentially many neurons with depth $L-1$. This is the mathematical basis for preferring deep over wide networks — depth gives exponential expressive power growth, while width gives polynomial growth.

### 8.3 Monotonicity and Fixed Points

**Definition.** $\sigma$ is monotone if $z_1 \le z_2 \implies \sigma(z_1) \le \sigma(z_2)$.

All classical activations (sigmoid, tanh, ReLU, ELU, SELU) are monotone. Modern activations (GELU, Swish, Mish) are non-monotone — they decrease in a small negative region.

**Impact of non-monotonicity:**
- Non-monotone activations can represent functions that are not monotone in any input dimension without requiring opposing sign weights — this increases expressive power
- The non-monotone "bump" in the negative regime allows the activation to implement a form of winner-take-all suppression: strongly positive neurons amplify their signals, slightly negative neurons transmit with reduced gain, strongly negative neurons are suppressed
- For gradient flow: a non-monotone activation has regions where the gradient is negative — this can cause oscillation in gradient descent if the network is not carefully initialised

**Fixed points.** A fixed point of $\sigma$ is a value $z^*$ satisfying $\sigma(z^*) = z^*$:
- Sigmoid: $\sigma(z^*) = z^*$ at $z^* \approx 0.652$ (unique non-trivial fixed point — verify numerically)
- Tanh: $\tanh(0) = 0$ — origin is a fixed point; no other real fixed points
- ReLU: all $z^* \ge 0$ are fixed points (identity on the positive half-line)
- SELU: $\operatorname{SELU}(0) = 0$ — origin is a fixed point; this is the self-normalising fixed point

The SELU fixed point at the origin is the mathematical core of self-normalisation: if the distribution of pre-activations is zero-mean, SELU outputs zero-mean activations, which again produce zero-mean pre-activations in the next layer (under appropriate weight initialisation).

### 8.4 Symmetry and Zero-Centred Outputs

**Definition.** An activation $\sigma$ is antisymmetric if $\sigma(-z) = -\sigma(z)$ (odd function). Antisymmetric activations have zero-centred outputs when the input distribution is symmetric around zero.

- Tanh: $\tanh(-z) = -\tanh(z)$ — antisymmetric, zero-centred
- ReLU: $\operatorname{ReLU}(-z) = 0 \ne -\operatorname{ReLU}(z) = -\max(0,z)$ — not antisymmetric, not zero-centred
- GELU: $\operatorname{GELU}(-z) = -z\Phi(-z) = -z(1-\Phi(z)) \ne -\operatorname{GELU}(z)$ — not antisymmetric, approximately zero-centred for $z \sim \mathcal{N}(0,1)$

**The zig-zag gradient problem in detail** (LeCun et al., 1998). For a one-layer network $f(\mathbf{x}) = \sum_j w_j \sigma(z_j)$, the gradient of the loss w.r.t. the weights is:

$$\frac{\partial \mathcal{L}}{\partial w_j} = \frac{\partial \mathcal{L}}{\partial f} \cdot a_j$$

where $a_j = \sigma(z_j)$. If $\sigma(z) \ge 0$ for all $z$ (like sigmoid), then all $a_j \ge 0$. This means all $\partial \mathcal{L}/\partial w_j$ have the same sign as $\partial \mathcal{L}/\partial f$. Consequently, all weights must move in the same direction simultaneously — the optimiser can only navigate the parameter space along "cone" directions. Tanh avoids this because $a_j$ can be negative, allowing individual weights to move in opposite directions.


---

## 9. Gradient Flow and Initialisation

### 9.1 Vanishing and Exploding Gradients

The backpropagation algorithm computes gradients via the chain rule. For a network with $L$ layers, pre-activations $\mathbf{z}^{[l]}$, and activations $\mathbf{a}^{[l]} = \sigma(\mathbf{z}^{[l]})$:

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \delta^{[l]} (\mathbf{a}^{[l-1]})^\top$$

where the error signal $\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}}$ satisfies the recurrence:

$$\delta^{[l]} = (W^{[l+1]})^\top \delta^{[l+1]} \odot \sigma'(\mathbf{z}^{[l]})$$

The full product across all layers:

$$\delta^{[1]} = \left(\prod_{l=1}^{L-1} \operatorname{diag}(\sigma'(\mathbf{z}^{[l]})) (W^{[l+1]})^\top \right) \delta^{[L]}$$

**Spectral norm analysis.** Taking norms and applying submultiplicativity:

$$\lVert \delta^{[1]} \rVert_2 \le \lVert \delta^{[L]} \rVert_2 \cdot \prod_{l=1}^{L-1} \max_j \lvert \sigma'(z_j^{[l]}) \rvert \cdot \lVert W^{[l+1]} \rVert_2$$

For gradient **vanishing**: if $\max_j \lvert \sigma'(z_j^{[l]}) \rvert \cdot \lVert W^{[l+1]} \rVert_2 < 1$ on average, the product shrinks exponentially in $L$.

For gradient **exploding**: if the product exceeds 1 on average, gradients grow exponentially.

**The "edge of chaos" regime** (Poole et al., 2016; Schoenholz et al., 2017): for random initialisation with variance $\sigma_w^2 / n$ per weight, the signal propagation in a random ReLU network satisfies:
$$q^{[l]} = \frac{\sigma_w^2}{2} q^{[l-1]} + \sigma_b^2$$

The fixed point $q^* = \sigma_b^2/(1 - \sigma_w^2/2)$ is stable for $\sigma_w^2 < 2$. At $\sigma_w^2 = 2$, the network is at the "edge of chaos" where signals propagate without amplification or damping, and gradients flow most cleanly. He initialisation ($\sigma_w^2 = 2/n_{\text{in}}$, so $\sigma_w^2 n_{\text{in}} = 2$) is precisely designed to hit this critical point.

### 9.2 Glorot / Xavier Initialisation

**Glorot & Bengio (2010)** derived the optimal weight variance for tanh and sigmoid activations. The key assumption: at initialisation, each activation is approximately linear (small weights → small pre-activations → activation in the approximately linear regime near the origin).

**Variance preservation requirement.** For a layer $z_j = \sum_{i=1}^{n_{\text{in}}} w_{ji} a_i$ with IID weights $w_{ji} \sim \mathcal{N}(0, \sigma_w^2)$ and IID activations $a_i$ with variance $V$:

$$\operatorname{Var}(z_j) = n_{\text{in}} \cdot \sigma_w^2 \cdot V$$

For the variance to be preserved ($\operatorname{Var}(z_j) = V$): $\sigma_w^2 = 1/n_{\text{in}}$.

**Symmetry argument.** The same analysis applies in the backward pass: the variance of gradients $\partial \mathcal{L}/\partial a_i$ satisfies $\operatorname{Var}(\partial \mathcal{L}/\partial a_i) = n_{\text{out}} \cdot \sigma_w^2 \cdot \operatorname{Var}(\partial \mathcal{L}/\partial z_j)$, requiring $\sigma_w^2 = 1/n_{\text{out}}$.

Since $1/n_{\text{in}}$ and $1/n_{\text{out}}$ are generally incompatible, Glorot & Bengio use the harmonic mean:

$$\sigma_w^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

This is the **Xavier / Glorot initialisation**. In uniform form: $w \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, +\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$.

**Activation gain.** For a non-linear activation $\sigma$ that is not exactly linear even at initialisation, multiply by the activation's gain $g_\sigma$:

$$\sigma_w^2 = g_\sigma^2 \cdot \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

Gain values: sigmoid $g = 1$, tanh $g = 5/3 \approx 1.67$, ReLU $g = \sqrt{2}$, GELU $\approx 1.7$, Swish $\approx 1.7$.

### 9.3 He Initialisation

**He et al. (2015)** derived initialisation specifically for ReLU networks. The key insight: ReLU kills half its inputs (outputs zero for $z < 0$), so the effective fan-in for variance computation is halved.

For a layer with ReLU activation and $n_{\text{in}}$ inputs, the variance of the post-activation is:

$$\operatorname{Var}(\operatorname{ReLU}(z)) = \frac{1}{2} \operatorname{Var}(z) = \frac{n_{\text{in}} \sigma_w^2}{2} \cdot \operatorname{Var}(a_{\text{in}})$$

For variance preservation:

$$\sigma_w^2 = \frac{2}{n_{\text{in}}}$$

This is **He initialisation** (also called Kaiming initialisation): $w \sim \mathcal{N}(0, 2/n_{\text{in}})$, or in uniform form $w \sim \mathcal{U}(-\sqrt{6/n_{\text{in}}}, +\sqrt{6/n_{\text{in}}})$.

**For bias vectors:** He initialisation typically uses $\mathbf{b} = \mathbf{0}$ — zero initialisation. Starting with zero biases ensures the pre-activations are initially zero-centred.

**Fan-out variant.** There is also a fan-out variant: $\sigma_w^2 = 2/n_{\text{out}}$, appropriate when considering the backward pass. In practice, the fan-in variant is more commonly used and performs slightly better empirically.

**Comparison table:**

| Initialisation | Formula | Best for |
|---|---|---|
| Xavier uniform | $\mathcal{U}(-\sqrt{6/(n_{\text{in}}+n_{\text{out}})}, +)$ | Tanh, sigmoid, linear |
| Xavier normal | $\mathcal{N}(0, 2/(n_{\text{in}}+n_{\text{out}}))$ | Tanh, sigmoid |
| He / Kaiming | $\mathcal{N}(0, 2/n_{\text{in}})$ | ReLU, Leaky ReLU |
| LeCun | $\mathcal{N}(0, 1/n_{\text{in}})$ | SELU (self-normalising) |
| Orthogonal | $W = Q\Sigma^{1/2}$, $Q$ random orthogonal | RNNs, gradient flow |

### 9.4 Gain Factors for Non-Standard Activations

For any activation $\sigma$, the gain $g_\sigma$ is defined so that the recommended initialisation is $\sigma_w^2 = g_\sigma^2 / n_{\text{in}}$:

$$g_\sigma^2 = \frac{1}{\mathbb{E}_{z \sim \mathcal{N}(0,1)}[\sigma'(z)^2]}$$

This is the reciprocal of the expected squared derivative under the initialisation distribution, measuring how much variance the activation preserves.

**Computing gains:**

For ReLU: $\mathbb{E}[\operatorname{ReLU}'(z)^2] = P(z > 0) = 1/2$, so $g_{\text{ReLU}} = \sqrt{2} \approx 1.41$.

For tanh: $\mathbb{E}[\operatorname{sech}^4(z)]$ under $z \sim \mathcal{N}(0,1)$ evaluates to approximately $0.36$, giving $g_{\tanh} \approx 1/\sqrt{0.36} \approx 5/3$.

For GELU: numerically, $\mathbb{E}_{z \sim \mathcal{N}(0,1)}[\operatorname{GELU}'(z)^2] \approx 0.35$, giving $g_{\text{GELU}} \approx \sqrt{1/0.35} \approx 1.70$.

For SiLU: similarly, $g_{\text{SiLU}} \approx 1.67$.

**For AI:** PyTorch's `torch.nn.init.calculate_gain('relu')` returns $\sqrt{2}$, `calculate_gain('tanh')` returns $5/3$, and `calculate_gain('leaky_relu', 0.01)` returns $\sqrt{2/(1+0.01^2)}$. GELU and Swish are not natively supported in PyTorch's gain table — practitioners typically use the He gain ($\sqrt{2}$) as an approximation, which works well empirically.


---

## 10. Computational Considerations and Approximations

### 10.1 Computational Cost Comparison

Activation functions vary significantly in computational cost, which matters at the scale of LLM inference. For a model with $T$ tokens, $L$ layers, and hidden dimension $d$, the FFN activation is applied $T \times L \times 2d/3 \times m$ times (where $m \approx 8d/3$ is the intermediate dimension). For LLaMA-7B with $T=2048$, $L=32$, $d=4096$, $m=11008$:

$$\text{Activation calls} = 2048 \times 32 \times 11008 \approx 720 \text{ million}$$

at each forward pass. The cost difference between activations translates directly to throughput:

| Activation | Relative FLOPS | Latency (CPU, relative) | Notes |
|---|---|---|---|
| ReLU | 1× | 1× | Single comparison |
| Leaky ReLU | 1× | 1× | Comparison + multiply |
| HardSwish | 2× | 1.5× | Clamp + multiply |
| Sigmoid | 5–8× | 4× | exp + add + div |
| Tanh | 5–8× | 4× | exp + exp + div |
| GELU (approx.) | 6–8× | 5× | tanh + polynomial |
| GELU (exact) | 8–12× | 8× | erf function |
| SiLU | 4–6× | 3× | sigmoid (shared) |
| Mish | 10–15× | 10× | tanh + softplus |

In practice, the FFN computation is dominated by the matrix multiplications, and the activation computation is a small fraction of total time. The performance difference between GELU and SiLU (~20–30% in activation-only benchmarks) translates to <5% in end-to-end transformer inference. This is why SwiGLU's advantages in model quality outweigh the slight overhead vs. ReLU.

### 10.2 Hard Approximations

**HardSigmoid.** Piecewise linear approximation to sigmoid:

$$\operatorname{HardSigmoid}(z) = \operatorname{clamp}\!\left(\frac{z + 3}{6}, 0, 1\right) = \begin{cases} 0 & z \le -3 \\ \frac{z+3}{6} & -3 \le z \le 3 \\ 1 & z \ge 3 \end{cases}$$

Used in quantisation-aware training (QAT) as a drop-in for sigmoid. The slope $1/6$ approximates the maximum slope of sigmoid ($1/4$) — close enough for many tasks.

**HardSwish** (Howard et al., MobileNetV3, 2019):

$$\operatorname{HardSwish}(z) = z \cdot \frac{\operatorname{ReLU6}(z + 3)}{6} = \begin{cases} 0 & z \le -3 \\ \frac{z(z+3)}{6} & -3 \le z \le 3 \\ z & z \ge 3 \end{cases}$$

where $\operatorname{ReLU6}(z) = \min(\max(0,z), 6)$. HardSwish approximates SiLU with piecewise-linear operations that are friendly to fixed-point arithmetic on mobile hardware. MobileNetV3 achieved a 15% speedup on mobile devices by replacing Swish with HardSwish at minimal accuracy cost.

**HardTanh:**

$$\operatorname{HardTanh}(z; \text{min\_val}, \text{max\_val}) = \operatorname{clamp}(z, \text{min\_val}, \text{max\_val})$$

Used as a bounding operation, commonly in quantisation-aware training to simulate the effect of clipping activations to the representable range.

**Quantisation-aware training (QAT).** During QAT, gradients flow through hard activations via the straight-through estimator (STE): the forward pass uses the hard (quantised) activation, but the backward pass treats it as the identity. This works because the gradient through the hard activation is 0 (gradient of clamp outside the saturation zone is 1, inside is 0), and STE replaces it with 1 everywhere.

### 10.3 Activation-Aware Quantisation

Large activation values (outliers) are the primary challenge for LLM quantisation. In LLMs, a small fraction of activation dimensions (~0.1%) have values 10–100× larger than the typical activation, making naive INT8 quantisation catastrophic (all small values round to zero to accommodate the range needed by outliers).

**SmoothQuant** (Xiao et al., 2022) migrates the quantisation difficulty from activations to weights:

$$\mathbf{Y} = (\mathbf{X} \operatorname{diag}(\mathbf{s})^{-1}) \cdot (\operatorname{diag}(\mathbf{s}) W)$$

where $\mathbf{s} \in \mathbb{R}^d$ is a per-channel smoothing factor: $s_j = \max_i \lvert X_{ij} \rvert^\alpha / \max_k \lvert W_{jk} \rvert^{1-\alpha}$ with $\alpha \in [0.5, 1]$. This divides large activation values by $s_j$ (reducing their range) and multiplies the corresponding weight row by $s_j$ (maintaining the computation while shifting the range difficulty to weights, which tolerate quantisation better).

**AWQ (Activation-aware Weight Quantisation)** (Lin et al., 2023) searches for per-channel scaling factors that minimise the quantisation error on a small calibration set, achieving near-FP16 accuracy at INT4 precision for models like LLaMA-7B and OPT-66B.

**For AI:** Understanding activation distributions is essential for deployment. In practice, LayerNorm before the FFN reduces activation outliers significantly (the normalisation step bounds activation variance), which is one reason that pre-norm transformers (LLaMA, GPT-NeoX) quantise more cleanly than post-norm transformers (original BERT).


---

## 11. Applications in Modern AI

### 11.1 BERT and GPT: GELU as Standard

**BERT (Devlin et al., 2018)** was the first major transformer to adopt GELU as its hidden-layer activation, replacing the ReLU used in the original "Attention is All You Need" FFN. The rationale: GELU's stochastic interpretation connects naturally to BERT's masked language modelling objective, where tokens are randomly masked and the model must predict them from context. The soft gating of GELU (probabilistically zeroing neurons) is analogous to dropout, providing implicit regularisation during the noisy pre-training process.

**GPT-2 (Radford et al., 2019)** and **GPT-3 (Brown et al., 2020)** both use GELU in the FFN, establishing it as the OpenAI standard. The fastGELU approximation (tanh-based, described in §5.1) was adopted for speed in GPT-2.

**Empirical comparison on language modelling (from Hendrycks & Gimpel, 2016):**
- On Penn Treebank with a 2-layer transformer: GELU reduced perplexity by 2–5 points vs. ReLU
- The improvement was larger for deeper models and smaller for shallower models
- On MNIST and CIFAR-10, GELU matched or exceeded ReLU with fewer training steps

### 11.2 LLaMA and PaLM: SwiGLU Dominance

**PaLM (Chowdhery et al., 2022)** established SwiGLU as the FFN activation for large-scale language models, using it in a 540B parameter model. The rationale was Shazeer's (2020) ablation showing SwiGLU outperforming all other FFN variants at the 220M parameter scale, with gains consistent across scale.

**LLaMA (Touvron et al., 2022)** made SwiGLU standard for open-weight models. LLaMA's architecture deviations from Llama's predecessors:
1. Pre-norm with RMSNorm (instead of post-norm with LayerNorm)
2. SwiGLU FFN (instead of standard ReLU or GELU FFN)
3. RoPE positional embeddings (instead of absolute or ALiBi)
4. No bias terms in linear layers

The combination of these choices (particularly pre-norm + SwiGLU) produces models that train stably at large scale with minimal numerical issues.

**LLaMA 2 and LLaMA 3** retain the same SwiGLU architecture, changing primarily the data, context length, and scale. The activation function's stability has made it the de facto standard for open-weight LLMs: Mistral, Falcon, Qwen, Gemma, Phi, and most 2023–2026 models use SwiGLU.

**The SwiGLU advantage at scale (mechanistic hypothesis).** The multiplicative structure of SwiGLU — $\operatorname{SiLU}(\mathbf{x}W_1) \odot (\mathbf{x}W_2)$ — enables the FFN to implement conditional computation: $W_1$ identifies "which concept" (via the gating function), and $W_2$ provides "what value" for that concept. This mirrors the key-value memory interpretation of Geva et al. (2021), where the first FFN layer acts as a key memory and the second as a value memory.

### 11.3 Attention Patterns and Temperature

The attention mechanism in transformers uses softmax to convert raw attention scores to attention weights:

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

The division by $\sqrt{d_k}$ is a temperature scaling: without it, for $d_k = 64$, the dot products $q_i \cdot k_j$ have variance 64 (for unit-variance queries and keys), leading to a softmax so peaked that gradients vanish.

**Attention entropy and overconfidence.** In very deep transformers, the attention entropy $H[A_l] = -\sum_j A_{lj} \log A_{lj}$ can drop to near zero (attention becomes essentially a hard lookup). This "attention collapse" is associated with overfitting and is addressed by:
- Attention dropout: randomly zeroing attention weights during training
- Talking-Heads attention (Shazeer et al., 2020): linear mixing across attention heads before and after softmax
- $\alpha$-entmax (Peters et al., 2019): a generalised softmax with controllable sparsity

**Temperature in decoding.** During autoregressive generation, temperature $\tau$ controls the diversity of sampled tokens. High temperature diversifies outputs but introduces hallucinations; low temperature makes outputs deterministic but repetitive. Beam search with $\tau = 0$ (argmax at each step) often produces degenerate text due to the "beam search curse" (Holtzman et al., 2020) — high probability sequences often lack diversity.

### 11.4 Activation Engineering and Mechanistic Interpretability

**Activation patching** (Meng et al., 2022) is a technique for localising knowledge in LLMs:
1. Run the model on a clean prompt; record all activations $\{a_l^{(i)}\}$ (layer $l$, token $i$)
2. Run the model on a corrupted prompt (e.g., with a key entity changed)
3. For each (layer, token) position, patch the corrupted run's activation with the clean activation
4. Measure the restoration of the correct output

If patching layer $l$ at token $i$ restores the correct answer, that (layer, token) pair causally mediates the knowledge. In the ROME paper, knowledge about specific facts (e.g., "The Eiffel Tower is in...") was localised to specific MLP (FFN) layers at the subject token position.

**Superposition theory** (Elhage et al., 2022). Neural networks can store more features than they have dimensions by superposing feature directions. The key enabling factor: non-linear activations allow the network to represent many "near-orthogonal" feature directions simultaneously. A linear network cannot do this (superposed linear features interfere destructively). The ReLU activation's sparsity (roughly 50% zero activations) is mathematically central to superposition — sparse activations reduce interference between superposed features.

**Activation steering** (Zou et al., 2023; Arditi et al., 2024): adding a "steering vector" to a layer's activations during the forward pass can induce specific behaviours (e.g., outputting content in a different language, displaying certain emotions, refusing refusals). The steering vector is a direction in activation space associated with a specific concept, computed by averaging the difference in activations between prompts with and without the concept. This works because LLM activations have a linear structure for many high-level concepts, despite the non-linear activation functions.


---

## 12. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Using sigmoid as hidden-layer activation in a deep network | Maximum derivative 1/4 causes gradient vanishing exponentially in depth; 10-layer sigmoid network has gradient attenuation of $(1/4)^9 \approx 4 \times 10^{-6}$ | Use ReLU, GELU, or Swish for hidden layers; reserve sigmoid for output (binary classification) or gating |
| 2 | Not using He initialisation with ReLU | Xavier init gives $\sigma_w^2 = 2/(n_{\text{in}}+n_{\text{out}})$, but ReLU kills half its inputs — this halves the effective fan-in and leads to variance decay | Use $\sigma_w^2 = 2/n_{\text{in}}$ (He/Kaiming) for all ReLU-family activations |
| 3 | Computing `softmax(z).log()` then `NLLLoss` | Separate softmax → log introduces numerical instability: $\log(e^z / \sum e^z)$ overflows for large $z$ | Use `log_softmax(z)` directly (numerically stable via log-sum-exp) or PyTorch's fused `CrossEntropyLoss` |
| 4 | Using `log_softmax` then applying cross-entropy over it | Double-applying log: `CrossEntropyLoss` expects raw logits, not log-probabilities; `NLLLoss` expects log-probabilities | Use `CrossEntropyLoss` with raw logits OR `NLLLoss` with `log_softmax` output — never both |
| 5 | Applying ReLU to a regression output layer | ReLU clips negative predictions to zero, unable to predict negative values | Use linear output for regression; apply positivity constraints only if the target is known to be positive (use `softplus` or `exp` instead) |
| 6 | Forgetting temperature scaling for calibration at inference | Models trained with cross-entropy are systematically overconfident; raw softmax outputs are not calibrated probabilities | Apply temperature scaling: fit $\tau$ on a validation set post-training; all confidence-sensitive applications require this |
| 7 | Using sigmoid for multi-class classification output | Sigmoid applied element-wise gives independent probabilities per class — they don't sum to 1 | Use softmax for mutually exclusive classes; sigmoid is correct for multi-label classification where classes are independent |
| 8 | Not accounting for dead neurons after training | Large learning rates or negative biases can permanently deactivate ReLU neurons; dead neurons contribute zero gradient | Monitor fraction of zero activations per layer; use Leaky ReLU, He init, and careful LR scheduling |
| 9 | Treating non-monotone activations like ReLU in gradient checks | GELU and Swish have regions of negative gradient — gradient checks near the non-monotone region require smaller perturbations | Use perturbation $h = 10^{-5}$ or smaller; verify gradient direction, not just magnitude |
| 10 | Assuming activation-wise computation dominates FFN time | The linear projections $(W_1 \mathbf{x})$ are 10–100× more expensive than the activation itself; optimising the activation without addressing the matrix multiply is premature | Profile wall-clock time; focus on matrix multiply efficiency (quantisation, flash attention, kernel fusion) before tuning activation |

---

## 13. Exercises

**Exercise 1 (★).** Prove that a network of $L$ linear layers with no activation is equivalent to a single linear layer. Specifically, show that $f(\mathbf{x}) = W^{[L]} W^{[L-1]} \cdots W^{[1]} \mathbf{x} + \mathbf{b}_{\text{eff}}$ is affine and derive the expression for $\mathbf{b}_{\text{eff}}$ in terms of $\mathbf{b}^{[1]}, \ldots, \mathbf{b}^{[L]}$.

**Exercise 2 (★).** (a) Derive the derivative of sigmoid: show $\sigma'(z) = \sigma(z)(1-\sigma(z))$. (b) Prove that $\tanh(z) = 2\sigma(2z) - 1$. (c) Derive the derivative of tanh from this relation. (d) What is the maximum value of $\sigma'(z)$? What is the maximum of $\tanh'(z)$?

**Exercise 3 (★).** Compute the gradient of $\operatorname{ReLU}(z)$ at $z = 1, 0, -1$ (using the convention $\operatorname{ReLU}'(0) = 0$). Then compute the gradient of GELU at the same points using the formula $\operatorname{GELU}'(z) = \Phi(z) + z\phi(z)$. Compare the values numerically.

**Exercise 4 (★★).** Vanishing gradient analysis. Consider a network with $L$ sigmoid hidden layers, each with weight matrices initialised with $\lVert W^{[l]} \rVert_2 = 1$. (a) Show that the gradient $\lVert \delta^{[1]} \rVert \le (1/4)^{L-1} \lVert \delta^{[L]} \rVert$. (b) Compute the attenuation factor for $L = 5, 10, 20$. (c) Repeat the analysis for tanh. (d) Explain why ReLU networks avoid this problem.

**Exercise 5 (★★).** SELU self-normalising property. Let $z \sim \mathcal{N}(0,1)$. (a) Compute $\mathbb{E}[\operatorname{ELU}(z; \alpha)]$ as a function of $\alpha$ (use $\mathbb{E}[e^z \mathbf{1}[z \le 0]] = \Phi(1) - 1$ where $\Phi$ is the standard normal CDF — verify numerically). (b) Set up the equations $\mathbb{E}[\operatorname{SELU}(z)] = 0$ and $\operatorname{Var}[\operatorname{SELU}(z)] = 1$. (c) Verify numerically that $\lambda \approx 1.0507$ and $\alpha \approx 1.6733$ satisfy both equations using Monte Carlo with $N = 10^6$ samples.

**Exercise 6 (★★).** Lipschitz constants. (a) Verify that the Lipschitz constant of sigmoid is $K = 1/4$ by finding the global maximum of $\lvert \sigma'(z) \rvert$. (b) Compute the Lipschitz constant of GELU numerically by evaluating $\sup_z \lvert \operatorname{GELU}'(z) \rvert$ over a fine grid. (c) For a 3-layer network with activations $f_3 = \sigma_3 \circ W_3 \circ \sigma_2 \circ W_2 \circ \sigma_1 \circ W_1$, bound the network's Lipschitz constant in terms of $K_1, K_2, K_3$ and $\lVert W_1 \rVert_2, \lVert W_2 \rVert_2, \lVert W_3 \rVert_2$.

**Exercise 7 (★★).** Softmax Jacobian. (a) Derive the Jacobian $J_{ij} = \partial s_i / \partial z_j$ where $\mathbf{s} = \operatorname{softmax}(\mathbf{z})$. (b) Show that $J = \operatorname{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$. (c) Verify that $J\mathbf{1} = \mathbf{0}$ (gradient of constant input is zero — softmax is not invertible). (d) Show that the cross-entropy gradient $\partial \mathcal{L}/\partial \mathbf{z} = \mathbf{s} - \mathbf{e}_y$ using this Jacobian.

**Exercise 8 (★★★).** SwiGLU parameter budget. (a) For a standard FFN with input/output dimension $d$ and intermediate dimension $4d$, count the number of parameters. (b) For a SwiGLU FFN with three weight matrices $W_1, W_2, W_3 \in \mathbb{R}^{d \times m}$, express the parameter count as a function of $m$. (c) Solve for $m$ that equates the two counts. (d) For LLaMA-7B with $d = 4096$, compute $m$ exactly and round up to the nearest multiple of 256. Verify your answer is 11008.

**Exercise 9 (★★★).** He initialisation derivation. (a) Let $z = \sum_{i=1}^n w_i a_i$ where $w_i \sim \mathcal{N}(0, \sigma_w^2)$ IID and $a_i \sim p_a$ IID with $\mathbb{E}[a_i] = 0$, $\operatorname{Var}(a_i) = V$. Show that $\operatorname{Var}(z) = n\sigma_w^2 V$. (b) For $a_i = \operatorname{ReLU}(z_i^{\text{prev}})$ with $z_i^{\text{prev}} \sim \mathcal{N}(0, V_{\text{prev}})$, compute $\operatorname{Var}(a_i)$ in terms of $V_{\text{prev}}$ (hint: $\mathbb{E}[\operatorname{ReLU}(X)^2] = V/2$ for $X \sim \mathcal{N}(0,V)$). (c) Derive the He initialisation $\sigma_w^2 = 2/n$ from the variance preservation requirement.

**Exercise 10 (★★★).** Temperature scaling and calibration. (a) Simulate a binary classifier outputting logits from $\mathcal{N}(1.5, 1)$ for class 1 and $\mathcal{N}(-1.5, 1)$ for class 0, with 10,000 test samples. Compute the expected calibration error (ECE) using 10 equal-width bins. (b) Apply temperature scaling: find the $\tau > 0$ that minimises the NLL $-\sum_i y_i \log \sigma(z_i/\tau) + (1-y_i)\log(1-\sigma(z_i/\tau))$ on a held-out validation set. (c) Compute the ECE after temperature scaling and verify it decreases. (d) Plot a reliability diagram (confidence on x-axis, accuracy on y-axis) before and after calibration.

---

## 14. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
|---|---|
| GELU (Hendrycks 2016) | Default activation in BERT, GPT-2/3/4, PaLM, T5; the mathematical justification (stochastic interpretation) connects to masked language modelling's random token masking |
| SwiGLU (Shazeer 2020) | Used in LLaMA 2/3, PaLM, Gemma, Mistral, Qwen, Falcon, Phi-2/3; the $8d/3$ hidden dimension is now a standard LLM architectural fact |
| Softmax in attention | The $1/\sqrt{d_k}$ temperature prevents attention collapse; FlashAttention's numerical stability relies on the log-sum-exp trick; attention entropy connects to generation quality |
| Temperature scaling | Calibration post-training is standard for safety-critical LLM applications; $\tau$ hyperparameter controls generation diversity (top-$p$, top-$k$ sampling) |
| Vanishing gradient → ReLU | The ReLU revolution (2011) enabled the ImageNet era (AlexNet 2012, VGG 2014, ResNet 2016); without ReLU, deep CNNs were impossible |
| He initialisation | Enables training of networks with 50–100+ layers; essential for ResNets, pre-norm transformers, and stable large-scale training |
| Lipschitz continuity | Spectral normalisation (WGAN, Miyato 2018) controls discriminator stability; Lipschitz bounds are used in certified robustness for adversarial defence |
| Activation engineering | Activation patching (ROME, Meng 2022) localises factual knowledge to FFN layers; steering vectors (Zou 2023) enable safe and unsafe behaviour control |
| Superposition (Elhage 2022) | ReLU sparsity enables polysemantic neurons; understanding superposition is central to mechanistic interpretability; sparse autoencoders (Cunningham et al. 2023) decompose superposed representations |
| Hard activations | HardSwish/HardSigmoid in MobileNetV3, EfficientNet-lite; activation-aware quantisation (SmoothQuant, AWQ) enables INT4 LLM inference |

---

## 15. Conceptual Bridge

### Looking Backward: Loss Functions

The previous section (§13-01, Loss Functions) established the chain $\mathcal{L}(\boldsymbol{\theta}) \to \nabla_{\boldsymbol{\theta}} \mathcal{L} \to$ parameter update. The gradient $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ is computed by backpropagation, which applies the chain rule through every layer. Activation functions appear in this chain as the factor $\sigma'(\mathbf{z}^{[l]})$ at each layer. The choice of loss function determines the form of $\delta^{[L]}$ (the output gradient), and the choice of activation determines how this gradient propagates backward. Cross-entropy + softmax produces the particularly clean gradient $\hat{\mathbf{p}} - \mathbf{e}_y$ because the log in cross-entropy cancels the exp in softmax. This cancellation is not coincidental — it is the mathematical consequence of the softmax being the canonical link function for the categorical distribution, and cross-entropy being the negative log-likelihood of that distribution.

### The Central Theme: Non-Linearity and Expressiveness

Activation functions are the mechanism by which depth creates expressive power. The universal approximation theorem guarantees that non-linear activations are sufficient for arbitrary function approximation; the vanishing gradient problem and its solutions (ReLU, He initialisation, residual connections) are the engineering that makes deep non-linear networks actually trainable. The history of deep learning — from the sigmoid era through ReLU to GELU and SwiGLU — is the history of finding activations that are simultaneously expressive, differentiable, non-saturating, and computationally efficient.

### Looking Forward: Normalisation Techniques

The next section (§13-03, Normalisation Techniques) extends the gradient flow analysis from this section. Activation functions control the non-linearity; normalisation techniques (BatchNorm, LayerNorm, RMSNorm) control the scale of activations between layers. The two mechanisms are deeply coupled: LayerNorm before the FFN activation (pre-norm) ensures that the FFN receives approximately $\mathcal{N}(0,1)$ inputs, putting activations in their non-saturating regime and making the Glorot/He initialisation assumptions approximately valid throughout training. Without normalisation, GELU and Swish can still saturate (for very large inputs), and the benefits of smooth activations over ReLU are partially lost.

```
POSITION IN CURRICULUM
════════════════════════════════════════════════════════════════════════

  Chapter 13: ML-Specific Mathematics
  ─────────────────────────────────────────────────────────────────

  [§13-01 Loss Functions]
       │  Gradient of loss determines δ^{[L]}; choice of loss ↔
       │  choice of output activation (sigmoid ↔ BCE, softmax ↔ CE)
       ▼
  [§13-02 Activation Functions]  ← YOU ARE HERE
       │  σ'(z) determines gradient flow through each layer;
       │  activation choice ↔ initialisation strategy (He, Glorot);
       │  SwiGLU/GELU are architectural decisions, not hyperparameters
       ▼
  [§13-03 Normalisation Techniques]
       │  BatchNorm, LayerNorm, RMSNorm stabilise activation scale;
       │  pre-norm vs. post-norm placement changes gradient dynamics
       ▼
  [§13-04 Sampling Methods]
       │  Temperature parameter in softmax; top-p/top-k sampling;
       │  stochastic activation interpretation of GELU

  ─────────────────────────────────────────────────────────────────

  Key dependencies:
  ┌─────────────────────────────────────────────────────────────┐
  │  Differentiability (§04) → Chain rule → Backprop gradients  │
  │  Linear algebra (§02-03) → Layer composition = matrix prod   │
  │  Probability (§06) → GELU stochastic interpretation         │
  │  UAT → Expressive power → Why depth matters                 │
  └─────────────────────────────────────────────────────────────┘

════════════════════════════════════════════════════════════════════════
```


---

## Appendix A: Activation Function Reference Table

Complete reference for all major activation functions with formulas, derivatives, and key properties.

### A.1 Classical Activations

**Sigmoid:**
$$\sigma(z) = \frac{1}{1+e^{-z}}, \qquad \sigma'(z) = \sigma(z)(1-\sigma(z)), \qquad \sigma'_{\max} = \frac{1}{4}$$

Range: $(0,1)$; monotone; $C^\infty$; not zero-centred; Lipschitz $K = 1/4$.

**Tanh:**
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1, \qquad \tanh'(z) = 1 - \tanh^2(z) = \operatorname{sech}^2(z)$$

Range: $(-1,1)$; monotone; $C^\infty$; zero-centred; antisymmetric; Lipschitz $K = 1$.

**Softplus (smooth ReLU approximation):**
$$\operatorname{softplus}(z) = \log(1 + e^z), \qquad \frac{d}{dz}\operatorname{softplus}(z) = \sigma(z)$$

Range: $(0, \infty)$; monotone; $C^\infty$; Lipschitz $K = 1$. Note: $\operatorname{softplus}(z) \approx z$ for large $z$, $\operatorname{softplus}(z) \approx 0$ for $z \ll 0$.

### A.2 ReLU Family

**ReLU:**
$$\operatorname{ReLU}(z) = \max(0,z), \qquad \operatorname{ReLU}'(z) = \mathbf{1}[z > 0]$$

Range: $[0,\infty)$; monotone; $C^0$; positive homogeneous; Lipschitz $K=1$.

**Leaky ReLU ($\alpha = 0.01$):**
$$\operatorname{LReLU}(z;\alpha) = \max(\alpha z, z), \qquad \operatorname{LReLU}'(z;\alpha) = \alpha + (1-\alpha)\mathbf{1}[z>0]$$

Range: $(-\infty,\infty)$; monotone; $C^0$; Lipschitz $K=1$.

**ELU ($\alpha = 1$):**
$$\operatorname{ELU}(z;\alpha) = \begin{cases} z & z \ge 0 \\ \alpha(e^z-1) & z < 0 \end{cases}, \qquad \operatorname{ELU}'(z;\alpha) = \begin{cases} 1 & z \ge 0 \\ \alpha e^z & z < 0 \end{cases}$$

Range: $(-\alpha, \infty)$; monotone; $C^1$; lower-bounded; Lipschitz $K = \max(1, \alpha)$.

**SELU:**
$$\operatorname{SELU}(z) = \lambda \cdot \operatorname{ELU}(z; \alpha)$$
$$\lambda \approx 1.0507, \quad \alpha \approx 1.6733, \quad \text{so } \lambda\alpha \approx 1.7581$$

Range: $(-\lambda\alpha, \infty)$; monotone; $C^1$; Lipschitz $K = \lambda \approx 1.0507$; self-normalising.

### A.3 Smooth Activations

**GELU:**
$$\operatorname{GELU}(z) = z\Phi(z) = \frac{z}{2}\left(1 + \operatorname{erf}\!\left(\frac{z}{\sqrt{2}}\right)\right)$$
$$\operatorname{GELU}'(z) = \Phi(z) + z\phi(z), \qquad \phi(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$$

Minimum at $z \approx -0.751$: $\operatorname{GELU}(-0.751) \approx -0.170$; Lipschitz $K \approx 1.13$.

**Fast GELU approximation:**
$$\widetilde{\operatorname{GELU}}(z) = \frac{z}{2}\left(1 + \tanh\!\left(\sqrt{\frac{2}{\pi}}\left(z + 0.044715z^3\right)\right)\right)$$

**Swish / SiLU ($\beta = 1$):**
$$\operatorname{SiLU}(z) = z\sigma(z) = \frac{z}{1+e^{-z}}$$
$$\operatorname{SiLU}'(z) = \sigma(z)(1 + z(1 - \sigma(z))) = \sigma(z) + z\sigma(z)(1-\sigma(z))$$

Minimum at $z \approx -1.278$: $\operatorname{SiLU}(-1.278) \approx -0.278$; Lipschitz $K \approx 1.10$.

**Mish:**
$$\operatorname{Mish}(z) = z\tanh(\operatorname{softplus}(z)) = z\tanh(\log(1+e^z))$$
$$\operatorname{Mish}'(z) = \tanh(\operatorname{softplus}(z)) + z\operatorname{sech}^2(\operatorname{softplus}(z))\sigma(z)$$

Minimum at $z \approx -0.621$: $\operatorname{Mish}(-0.621) \approx -0.308$; Lipschitz $K \approx 1.07$.

### A.4 Output Activations

**Softmax ($K$ classes):**
$$\operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \qquad J = \operatorname{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$$

**Log-softmax (numerically stable):**
$$\operatorname{log\_softmax}(\mathbf{z})_i = z_i - m - \log\sum_{j=1}^K e^{z_j - m}, \quad m = \max_j z_j$$

**Temperature-scaled softmax:**
$$\operatorname{softmax}(\mathbf{z}/\tau)_i = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}$$

### A.5 Gated Variants

**SwiGLU:**
$$\operatorname{SwiGLU}(\mathbf{x}) = \operatorname{SiLU}(\mathbf{x}W_1) \odot (\mathbf{x}W_2), \quad W_1, W_2 \in \mathbb{R}^{d \times m}, \; m = 8d/3$$

**GeGLU:**
$$\operatorname{GeGLU}(\mathbf{x}) = \operatorname{GELU}(\mathbf{x}W_1) \odot (\mathbf{x}W_2)$$

**GLU (sigmoid gate):**
$$\operatorname{GLU}(\mathbf{x}) = (\mathbf{x}W_1) \odot \sigma(\mathbf{x}W_2)$$

**ReGLU:**
$$\operatorname{ReGLU}(\mathbf{x}) = \operatorname{ReLU}(\mathbf{x}W_1) \odot (\mathbf{x}W_2)$$


---

## Appendix B: Detailed Gradient Derivations

### B.1 Sigmoid Derivative (Step by Step)

$$\frac{d}{dz}\sigma(z) = \frac{d}{dz}\left(\frac{1}{1+e^{-z}}\right) = \frac{-(-e^{-z})}{(1+e^{-z})^2} = \frac{e^{-z}}{(1+e^{-z})^2}$$

Factoring:
$$= \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \sigma(z) \cdot \frac{1+e^{-z}-1}{1+e^{-z}} = \sigma(z)(1-\sigma(z))$$

Maximum: $\sigma(z)(1-\sigma(z)) \le 1/4$ by AM-GM, with equality at $\sigma(z) = 1/2$, i.e., $z = 0$.

### B.2 GELU Derivative (Step by Step)

$$\operatorname{GELU}(z) = z\Phi(z)$$

By the product rule:
$$\operatorname{GELU}'(z) = \Phi(z) + z\Phi'(z) = \Phi(z) + z\phi(z)$$

where $\phi(z) = \Phi'(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}$ is the standard normal PDF.

Evaluating at $z = 0$: $\operatorname{GELU}'(0) = \Phi(0) + 0 = 1/2$. Note that the derivative is not 1 at the origin — unlike ReLU (derivative 1 for $z > 0$) and tanh (derivative 1 at $z=0$), GELU has a moderate slope.

Finding the maximum of $\operatorname{GELU}'(z)$: set $\Phi''(z) + \phi(z) + z\phi'(z) = 0$. Since $\phi'(z) = -z\phi(z)$:
$$\phi'(z) + \phi(z) - z^2\phi(z) = 0 \implies (1 - z^2)\phi(z) = -\phi'(z)$$

The maximum is achieved numerically around $z \approx 0.6$ where $\operatorname{GELU}'(0.6) \approx 1.13$.

### B.3 Softmax Jacobian (Step by Step)

Let $\mathbf{s} = \operatorname{softmax}(\mathbf{z})$ with $s_i = e^{z_i}/S$ where $S = \sum_k e^{z_k}$.

For $i = j$ (diagonal):
$$\frac{\partial s_i}{\partial z_i} = \frac{e^{z_i} \cdot S - e^{z_i} \cdot e^{z_i}}{S^2} = \frac{e^{z_i}}{S} - \frac{e^{z_i}}{S} \cdot \frac{e^{z_i}}{S} = s_i - s_i^2 = s_i(1-s_i)$$

For $i \ne j$ (off-diagonal):
$$\frac{\partial s_i}{\partial z_j} = \frac{0 \cdot S - e^{z_i} \cdot e^{z_j}}{S^2} = -\frac{e^{z_i}}{S} \cdot \frac{e^{z_j}}{S} = -s_i s_j$$

Compact form: $J_{ij} = s_i(\delta_{ij} - s_j)$, or in matrix form $J = \operatorname{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$.

Verification: $J\mathbf{1} = \mathbf{s} - \mathbf{s}(\mathbf{s}^\top \mathbf{1}) = \mathbf{s} - \mathbf{s}\cdot 1 = \mathbf{0}$ since $\sum_i s_i = 1$. This confirms that shifting all logits by a constant does not change the gradient — as expected from the shift-invariance of softmax.

### B.4 Cross-Entropy Gradient Through Softmax

Loss: $\mathcal{L} = -\log s_y = -z_y + \log S$ where $S = \sum_k e^{z_k}$.

Direct differentiation:
$$\frac{\partial \mathcal{L}}{\partial z_i} = -\mathbf{1}[i=y] + \frac{e^{z_i}}{S} = s_i - \mathbf{1}[i=y]$$

In vector form: $\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{s} - \mathbf{e}_y$.

This result can also be obtained via the chain rule: $\nabla_{\mathbf{z}} \mathcal{L} = J^\top \nabla_{\mathbf{s}} \mathcal{L}$ where $\nabla_{\mathbf{s}} \mathcal{L} = -\mathbf{e}_y / s_y$ (component-wise). Then:

$$(\nabla_{\mathbf{z}} \mathcal{L})_i = \sum_j J_{ji} \cdot (-\delta_{jy}/s_y) = -J_{yi}/s_y = -(s_y(\delta_{iy} - s_i)/s_y) = -(δ_{iy} - s_i) = s_i - \delta_{iy}$$

which matches the direct computation.

### B.5 SELU Fixed-Point Computation

We need $(\lambda, \alpha)$ such that if $z \sim \mathcal{N}(0,1)$, then $\mathbb{E}[\operatorname{SELU}(z)] = 0$ and $\operatorname{Var}[\operatorname{SELU}(z)] = 1$.

**Mean condition:**
$$\mathbb{E}[\operatorname{SELU}(z)] = \lambda\left(\int_0^\infty z \phi(z)\,dz + \int_{-\infty}^0 \alpha(e^z - 1)\phi(z)\,dz\right) = 0$$

where $\phi(z)$ is the standard normal PDF.

Computing $\int_0^\infty z\phi(z)\,dz = 1/\sqrt{2\pi}$ (since $\int_0^\infty z e^{-z^2/2}\,dz = 1$).

Computing $\int_{-\infty}^0 (e^z - 1)\phi(z)\,dz = e^{1/2}\Phi(-1) - \Phi(0) = e^{1/2}(1-\Phi(1)) - 1/2$.

Setting the sum to zero and solving gives $\alpha$ in terms of $\Phi(1) \approx 0.8413$.

**Variance condition:** with the mean-zero constraint, $\operatorname{Var}[\operatorname{SELU}(z)] = \lambda^2 \mathbb{E}[\operatorname{ELU}(z;\alpha)^2] = 1$, which determines $\lambda$ given $\alpha$.

The numerical values $\lambda = 1.0507009873554805$ and $\alpha = 1.6732632423543772$ are the unique positive solutions (verified in Klambauer et al. 2016, Supplementary Material).


---

## Appendix C: Initialisation Reference

### C.1 Complete Initialisation Table

| Scheme | Formula | Recommended for | Rationale |
|---|---|---|---|
| Glorot/Xavier uniform | $\mathcal{U}(-\sqrt{6/(n_{\text{in}}+n_{\text{out}})}, +)$ | Sigmoid, tanh, linear | Preserves variance in both forward and backward passes |
| Glorot/Xavier normal | $\mathcal{N}(0, 2/(n_{\text{in}}+n_{\text{out}}))$ | Sigmoid, tanh | Gaussian version of above |
| He/Kaiming normal | $\mathcal{N}(0, 2/n_{\text{in}})$ | ReLU, PReLU | ReLU kills half inputs; doubles variance to compensate |
| He/Kaiming uniform | $\mathcal{U}(-\sqrt{6/n_{\text{in}}}, +)$ | ReLU | Uniform version with matching variance |
| LeCun normal | $\mathcal{N}(0, 1/n_{\text{in}})$ | SELU | Required for self-normalisation property |
| Orthogonal | $W = Q$ from QR of random normal | RNNs, deep networks | Isometry: $\lVert W\mathbf{x} \rVert = \lVert \mathbf{x} \rVert$ |
| Identity + small | $W = I + \epsilon\mathcal{N}(0, 0.01)$ | ResNet residuals | Near-identity at init; relies on skip connections |

### C.2 Gain Factors by Activation

The initialisation formula with gain is: $\sigma_w^2 = g^2 \cdot 2/(n_{\text{in}} + n_{\text{out}})$ (Glorot) or $g^2 \cdot 2/n_{\text{in}}$ (He).

| Activation | Gain $g$ | Source |
|---|---|---|
| Linear | 1.000 | Identity gain |
| Sigmoid | 1.000 | Approx. linear near zero |
| Tanh | 1.667 ($= 5/3$) | LeCun et al. 1998 |
| ReLU | 1.414 ($= \sqrt{2}$) | He et al. 2015 |
| Leaky ReLU ($\alpha$) | $\sqrt{2/(1+\alpha^2)}$ | Generalised He |
| GELU | $\approx 1.703$ | Numerical computation |
| SiLU | $\approx 1.670$ | Numerical computation |
| SELU | 1.000 | LeCun normal (not gain-scaled) |

### C.3 Variance Flow Through a Network

For a network initialised with He ($\sigma_w^2 = 2/n$) and ReLU activations, the variance of activations at each layer satisfies:

$$\operatorname{Var}(\mathbf{a}^{[l]}) = \operatorname{Var}(\mathbf{a}^{[0]}) \cdot \prod_{k=1}^l \frac{\sigma_{w_k}^2 \cdot n_k}{2} = \operatorname{Var}(\mathbf{a}^{[0]}) \cdot 1^l = \operatorname{Var}(\mathbf{a}^{[0]})$$

Perfect variance preservation! This is why He initialisation enables training of very deep networks. In contrast, with Glorot initialisation and ReLU:

$$\operatorname{Var}(\mathbf{a}^{[l]}) = \operatorname{Var}(\mathbf{a}^{[0]}) \cdot \left(\frac{n_{\text{in}}}{n_{\text{in}} + n_{\text{out}}}\right)^l \approx (1/2)^l$$

exponential decay in variance, returning to the vanishing gradient regime.


---

## Appendix D: Modern LLM Architecture Activation Summary

### D.1 FFN Architecture Comparison

| Model | Activation | FFN intermediate dim | Notes |
|---|---|---|---|
| GPT-2 (2019) | GELU (approx.) | $4d_{\text{model}}$ | FastGELU |
| BERT-base (2018) | GELU | $4d_{\text{model}} = 3072$ | Exact GELU |
| T5 v1.0 (2020) | ReLU | $4d_{\text{model}}$ | Standard FFN |
| T5 v1.1 (2020) | GeGLU | $\frac{8}{3}d_{\text{model}}$ | Gated; 3 matrices |
| GPT-3 (2020) | GELU | $4d_{\text{model}}$ | FastGELU |
| PaLM (2022) | SwiGLU | $\frac{8}{3}d_{\text{model}}$ | First large SwiGLU |
| LLaMA-7B (2023) | SwiGLU | $11008$ ($\frac{8}{3} \times 4096$) | Rounded to mult of 256 |
| LLaMA-2-70B (2023) | SwiGLU | $28672$ ($\frac{8}{3} \times d$) | Same architecture |
| Mistral-7B (2023) | SwiGLU | $14336$ | Window attention + SwiGLU |
| Gemma-7B (2024) | GeGLU | $\frac{8}{3}d_{\text{model}}$ | Google Gemma |
| Phi-2 (2023) | GELU | $4d_{\text{model}}$ | Microsoft, kept GELU |
| Qwen-7B (2023) | SwiGLU | $\frac{8}{3}d_{\text{model}}$ | Alibaba |
| Falcon-7B (2023) | GELU | $4d_{\text{model}}$ | TII, kept GELU |
| LLaMA-3 (2024) | SwiGLU | $\frac{8}{3}d_{\text{model}}$ | Meta standard |

### D.2 Why Not MLP with ReLU for LLMs?

Three empirical observations consistently support GELU/SwiGLU over ReLU for large language models:

**1. Perplexity gap.** In Shazeer (2020) and internal ablations at Google/Meta, SwiGLU achieves 0.3–0.8 lower perplexity on standard language modelling benchmarks (WikiText-103, C4) at matched parameter count. This gap compounds over training steps.

**2. Training stability.** ReLU networks occasionally exhibit "activation death" at scale — large gradients kill neurons early in training. GELU/SwiGLU avoid this through smooth gradients and the positive bias of the gating function.

**3. Representation quality.** Probing classifiers (Tenney et al., 2019; Hewitt & Manning, 2019) find that GELU-based models encode syntactic and semantic structure more linearly in their activation space, suggesting that smooth activations produce representations more amenable to linear readout — which is precisely what downstream task heads require.

### D.3 Vision Transformer Activations

Vision Transformers (ViT, Dosovitskiy et al., 2020) use GELU in FFN blocks, following BERT's convention. Later vision models:
- **Swin Transformer (2021):** GELU
- **DeiT (2021):** GELU
- **ConvNeXt (2022):** GELU (deliberately matching transformer design principles)
- **ViT-22B (2023):** SwiGLU

The convergence on SwiGLU for very large models spans both vision and language domains.


---

## Appendix E: Numerical Implementation Patterns

### E.1 Numerically Stable Sigmoid

The naive computation of $\sigma(z) = 1/(1+e^{-z})$ overflows for $z > 709$ (IEEE float64). The stable implementation:

```python
def sigmoid(z):
    # Stable computation: avoid overflow in exp
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))
```

This avoids computing $e^{-z}$ for $z \ll 0$ (would underflow) and $e^z$ for $z \gg 0$ (would overflow).

### E.2 Numerically Stable Log-Softmax

```python
def log_softmax(z):
    # z: shape (..., K), compute log-softmax along last axis
    m = z.max(axis=-1, keepdims=True)      # max subtraction trick
    log_sum_exp = np.log(np.sum(np.exp(z - m), axis=-1, keepdims=True))
    return z - m - log_sum_exp

def cross_entropy(logits, targets):
    # targets: integer class indices
    log_probs = log_softmax(logits)
    return -log_probs[np.arange(len(targets)), targets].mean()
```

### E.3 Fast GELU Approximation

```python
import numpy as np

def gelu_exact(z):
    from scipy.special import erf
    return 0.5 * z * (1 + erf(z / np.sqrt(2)))

def gelu_approx(z):
    # Hendrycks & Gimpel (2016) tanh approximation
    c = np.sqrt(2 / np.pi)
    return 0.5 * z * (1 + np.tanh(c * (z + 0.044715 * z**3)))

# Maximum absolute error
z = np.linspace(-5, 5, 10000)
max_err = np.max(np.abs(gelu_exact(z) - gelu_approx(z)))
# max_err ≈ 0.00005 — well within numerical precision for 32-bit training
```

### E.4 SwiGLU Forward Pass

```python
def swiglu_ffn(x, W1, W2, W3):
    """
    SwiGLU FFN block.
    x:  (batch, seq, d)
    W1: (d, m)  -- gate weight
    W2: (d, m)  -- value weight
    W3: (m, d)  -- output projection
    """
    gate = x @ W1          # (batch, seq, m)
    value = x @ W2         # (batch, seq, m)
    hidden = silu(gate) * value    # element-wise product
    return hidden @ W3     # (batch, seq, d)

def silu(z):
    return z * sigmoid(z)
```

### E.5 Temperature Scaling Calibration

```python
from scipy.optimize import minimize_scalar

def ece(probs, labels, n_bins=10):
    """Expected calibration error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_sum = 0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (confidences >= low) & (confidences < high)
        if mask.sum() == 0:
            continue
        acc = (predictions[mask] == labels[mask]).mean()
        conf = confidences[mask].mean()
        ece_sum += mask.sum() * abs(acc - conf)
    return ece_sum / len(labels)

def calibrate_temperature(logits_val, labels_val):
    """Find optimal temperature on validation set."""
    def nll(log_tau):
        tau = np.exp(log_tau)
        log_probs = log_softmax(logits_val / tau)
        return -log_probs[np.arange(len(labels_val)), labels_val].mean()
    
    result = minimize_scalar(nll, bounds=(-2, 2), method='bounded')
    return np.exp(result.x)   # optimal temperature
```

### E.6 Activation Gain Computation

```python
import numpy as np

def compute_gain(activation_fn, n_samples=1_000_000):
    """Compute initialisation gain for any activation function.
    gain = 1 / sqrt(E[sigma'(z)^2]) for z ~ N(0,1).
    """
    z = np.random.normal(0, 1, n_samples)
    h = 1e-5
    # Numerical derivative via central difference
    deriv = (activation_fn(z + h) - activation_fn(z - h)) / (2 * h)
    return 1.0 / np.sqrt(np.mean(deriv**2))

np.random.seed(42)
print(f"ReLU gain:  {compute_gain(lambda z: np.maximum(0, z)):.4f}")  # ≈ 1.4142
print(f"Tanh gain:  {compute_gain(np.tanh):.4f}")                     # ≈ 1.6667
print(f"GELU gain:  {compute_gain(gelu_approx):.4f}")                 # ≈ 1.7028
print(f"SiLU gain:  {compute_gain(silu):.4f}")                        # ≈ 1.6704
```


---

## Appendix F: Key Papers Quick Reference

| Year | Paper | Contribution | Key Result |
|---|---|---|---|
| 1989 | Cybenko | Universal Approximation Theorem | One hidden layer + sigmoid approximates any continuous function |
| 1991 | Hornik | UAT generalised | Any non-polynomial activation works |
| 1998 | LeCun et al. | Efficient Backprop | Xavier-like init; tanh preferred over sigmoid for hidden layers |
| 2010 | Glorot & Bengio | Understanding vanishing gradients | Xavier initialisation formula; sigmoid kills gradients |
| 2011 | Glorot, Bordes & Bengio | Deep sparse rectifier networks | ReLU for hidden layers; better than sigmoid/tanh |
| 2013 | Maas et al. | Rectifier nonlinearities | Leaky ReLU; prevents dying neurons |
| 2013 | Pascanu et al. | Difficulty of training RNNs | Formal gradient explosion/vanishing analysis |
| 2015 | He et al. | Delving deep into rectifiers | PReLU; He/Kaiming initialisation; ImageNet SOTA |
| 2015 | Clevert et al. | ELU | Negative outputs; smooth derivative; faster convergence |
| 2016 | Klambauer et al. | SELU / Self-normalising NNs | Self-normalising fixed point; lecun-normal init |
| 2016 | Hendrycks & Gimpel | GELU | Gaussian CDF gating; stochastic interpretation |
| 2016 | Poole et al. | Exponential expressivity | Edge of chaos; depth-width expressivity tradeoff |
| 2017 | Dauphin et al. | GLU | Gated linear units for language modelling |
| 2017 | Ramachandran et al. | Swish/SiLU | NAS-discovered activation; outperforms ReLU |
| 2017 | Miyato et al. | Spectral normalisation | Lipschitz constraint for GAN discriminator |
| 2018 | Devlin et al. | BERT | GELU standardised for transformer hidden layers |
| 2019 | Misra | Mish | Smooth non-monotone activation |
| 2020 | Shazeer | GLU variants | SwiGLU, GeGLU, ReGLU ablation |
| 2021 | Geva et al. | FFN as key-value memory | FFN layers retrieve factual knowledge via activations |
| 2021 | Elhage et al. | Superposition hypothesis | ReLU sparsity enables superposition of features |
| 2022 | Meng et al. | ROME | Activation patching for knowledge localisation |
| 2022 | Touvron et al. | LLaMA | SwiGLU + RMSNorm + RoPE as LLM standard |
| 2022 | Xiao et al. | SmoothQuant | Activation-aware quantisation; migrates difficulty to weights |
| 2023 | Lin et al. | AWQ | Activation-aware weight quantisation; INT4 inference |
| 2023 | Cunningham et al. | Sparse autoencoders | Decompose superposed LLM activations |
| 2023 | Zou et al. | Activation steering | Linear representations enable concept steering |

---

## Appendix G: Self-Assessment Checklist

Use this checklist to verify mastery before moving to Normalisation Techniques.

**Section 1 — Intuition:**
- [ ] Can prove that a $L$-layer linear network is equivalent to a single linear layer
- [ ] Can explain why polynomial activations are insufficient for universal approximation
- [ ] Can name the 6 major activation "eras" and the mathematical problem each era solved

**Section 3 — Classical:**
- [ ] Can derive $\sigma'(z) = \sigma(z)(1-\sigma(z))$ from scratch
- [ ] Can prove $\tanh(z) = 2\sigma(2z) - 1$
- [ ] Can compute the gradient attenuation for a $L$-layer sigmoid network
- [ ] Can identify when sigmoid/tanh are appropriate (output layer, gates)

**Section 4 — ReLU:**
- [ ] Can state the dying ReLU condition formally
- [ ] Can explain why He initialisation uses $\sigma_w^2 = 2/n_{\text{in}}$ (not $1/n_{\text{in}}$)
- [ ] Can describe the self-normalising property of SELU (fixed-point argument)

**Section 5 — Modern:**
- [ ] Can state the stochastic interpretation of GELU
- [ ] Can write the SiLU formula and its derivative
- [ ] Can explain the $\tau \to 0$ and $\tau \to \infty$ limits of temperature-scaled softmax

**Section 7 — GLU:**
- [ ] Can write the SwiGLU formula with the correct matrices
- [ ] Can derive the intermediate dimension $m = 8d/3$ from the parameter budget constraint
- [ ] Can explain why LLaMA uses 11008 as the intermediate dimension for $d=4096$

**Section 8–9 — Theory:**
- [ ] Can prove that $\operatorname{GELU}(z) \ge -0.17$ (bounded below)
- [ ] Can derive the Glorot initialisation formula from variance preservation
- [ ] Can compute the He initialisation formula for ReLU specifically
- [ ] Can compute the gain factor for any activation numerically

**Section 11 — Applications:**
- [ ] Can explain what activation patching reveals about LLM knowledge storage
- [ ] Can connect the superposition hypothesis to ReLU sparsity
- [ ] Can describe SmoothQuant's approach to activation quantisation


---

## Appendix H: Activation Functions in Special Architectures

### H.1 Recurrent Networks

**LSTMs (Hochreiter & Schmidhuber, 1997).** The LSTM cell uses four activation functions serving distinct roles:

$$\mathbf{f}_t = \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \quad \text{(forget gate — sigmoid, range (0,1))}$$
$$\mathbf{i}_t = \sigma(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \quad \text{(input gate — sigmoid, range (0,1))}$$
$$\tilde{\mathbf{C}}_t = \tanh(W_C [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_C) \quad \text{(cell candidate — tanh, range (-1,1))}$$
$$\mathbf{o}_t = \sigma(W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \quad \text{(output gate — sigmoid, range (0,1))}$$
$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{C}}_t$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t) \quad \text{(hidden state — tanh applied to cell)}$$

The sigmoid gates have range $(0,1)$, making them natural "valves" — 0 closes the gate, 1 opens it. The cell state update uses tanh to keep the cell state bounded in $(-1,1)$, preventing explosion over long sequences. The final hidden state is bounded in $(-1,1)$ by the outer tanh.

**GRUs** use two sigmoid gates (reset and update) and one tanh for the candidate hidden state. The simpler structure has fewer parameters but comparable performance on many tasks.

**Key property for gradient flow:** the LSTM cell state update $\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \ldots$ creates an **additive** path for gradients through time (the $\mathbf{C}_{t-1}$ term has gradient $\mathbf{f}_t$, which is in $(0,1)$ but not necessarily small). This is conceptually analogous to ResNet skip connections — additive paths preserve gradient magnitude.

### H.2 Graph Neural Networks

In graph attention networks (GATs), the attention coefficient between nodes $i$ and $j$ uses:

$$e_{ij} = \operatorname{LeakyReLU}(\mathbf{a}^\top [W\mathbf{h}_i \| W\mathbf{h}_j])$$

$$\alpha_{ij} = \operatorname{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

The LeakyReLU is used (rather than ReLU) to ensure all attention coefficients have non-zero gradient, even for low-scoring edges. This prevents gradient vanishing through the attention mechanism for graph nodes with many neighbours.

### H.3 Normalising Flows

Normalising flows require **invertible** activations to enable exact likelihood computation. Common choices:

**LeakyReLU** is invertible: $\operatorname{LReLU}^{-1}(y;\alpha) = y$ if $y > 0$, else $y/\alpha$. The Jacobian determinant is simply $1$ if $y > 0$, else $\alpha$.

**Sigmoid** is invertible (logit function): $\sigma^{-1}(y) = \log(y/(1-y))$.

**Non-invertible activations (ReLU, GELU, Swish) cannot be used** in normalising flows because they map multiple pre-activation values to the same post-activation value (ReLU maps all $z \le 0$ to 0), making the flow non-injective and the likelihood computation undefined.

### H.4 Attention with Sparse Activations

In sparse attention mechanisms, the standard softmax is replaced by activations that produce exact zeros:

**ReLU attention** (Peng et al., 2021, "Random Feature Attention"):
$$\operatorname{Attention}_{\text{ReLU}}(Q,K,V)_i = \frac{\sum_j \operatorname{ReLU}(\mathbf{q}_i \cdot \mathbf{k}_j / \sqrt{d_k}) \mathbf{v}_j}{\sum_j \operatorname{ReLU}(\mathbf{q}_i \cdot \mathbf{k}_j / \sqrt{d_k})}$$

ReLU attention is sparse (many zero weights) and enables efficient linear-complexity attention via kernel trick. However, it lacks the normalisation guarantee of softmax and can produce all-zero rows if $\mathbf{q}_i \cdot \mathbf{k}_j < 0$ for all $j$.

**Sigmoid attention** (Ramapuram et al., 2024, "FLASH") replaces softmax with sigmoid applied element-wise to attention logits, divided by the sequence length $N$. This removes the competition between attention weights (no "sum-to-1" constraint), allowing the model to attend to all tokens simultaneously.


---

## Appendix I: Activation Function Visualisations (ASCII)

### I.1 Sigmoid vs. Tanh Shape Comparison

```
SIGMOID AND TANH COMPARISON
════════════════════════════════════════════════════════════════════════

  Value
  1.0 |         ................   ← tanh asymptote (1)
  0.5 |      ../  sigmoid(0)=0.5
  0.0 |·······*·················· ← tanh(0)=0, origin
 -0.5 |           \..
 -1.0 |................           ← tanh asymptote (-1)
       ───────────────────────────
       -4   -2   0    2    4   z

  Key difference: tanh has range (-1,1) and passes through origin;
  sigmoid has range (0,1) and passes through (0, 0.5).
  tanh = 2*sigmoid(2z) - 1 (scale-shift relationship).

════════════════════════════════════════════════════════════════════════
```

### I.2 ReLU Family Shape Comparison

```
RELU FAMILY ACTIVATION COMPARISON
════════════════════════════════════════════════════════════════════════

   value
   3.0 |          /  ← ReLU = Leaky ReLU for z > 0 (identical)
   2.0 |         /
   1.0 |        /
   0.0 *───────*    ← ReLU dead zone (flat at 0 for z ≤ 0)
  -0.3 |     /   ← Leaky ReLU slope α=0.01 (barely visible)
  -1.0 |  ./     ← ELU: -α(1-e^z) approaches -α=-1
        ──────────────────────────────
        -3  -2  -1   0   1   2   3   z

  ReLU:  flat at 0 for z ≤ 0 → dead neurons possible
  Leaky: small slope α for z ≤ 0 → no dead neurons, barely negative
  ELU:   smooth curve, approaches -α → soft negative regime

════════════════════════════════════════════════════════════════════════
```

### I.3 GELU vs. Swish Comparison

```
GELU AND SWISH/SiLU SHAPE COMPARISON
════════════════════════════════════════════════════════════════════════

   value
   3.0 |          /  ← both → z for large z (near linear)
   2.0 |         /
   1.0 |        /
   0.0 *──────./   ← both cross zero near 0
  -0.1 |   _./    ← SiLU minimum ≈ -0.278 at z ≈ -1.28
  -0.2 |  /       ← GELU minimum ≈ -0.170 at z ≈ -0.751
        ──────────────────────────────
        -3  -2  -1   0   1   2   3   z

  Both are non-monotone (decreasing in small negative region).
  Both approach identity for large positive z.
  SiLU has deeper minimum; GELU is more conservative in negative regime.

════════════════════════════════════════════════════════════════════════
```

### I.4 Softmax Temperature Visualisation

```
SOFTMAX TEMPERATURE EFFECT
════════════════════════════════════════════════════════════════════════

  Example logits: z = [3, 1, 0.5, -1, -2]

  τ = 0.2  (low): [0.985, 0.013, 0.002, ~0, ~0]  → near one-hot
  τ = 0.5  :      [0.876, 0.083, 0.034, 0.005, 0.001]
  τ = 1.0  :      [0.717, 0.131, 0.081, 0.040, 0.030]  → standard
  τ = 2.0  :      [0.523, 0.168, 0.143, 0.090, 0.075]
  τ = 5.0  :      [0.280, 0.217, 0.206, 0.168, 0.129]
  τ → ∞   :      [0.200, 0.200, 0.200, 0.200, 0.200]  → uniform

  ↑ higher τ = softer distribution = more exploration
  ↓ lower  τ = sharper distribution = more exploitation

════════════════════════════════════════════════════════════════════════
```

---

## Appendix J: Connections to Other Areas of Mathematics

### J.1 Connection to Functional Analysis

Activation functions are elements of function spaces. The space of continuous functions $C(\mathbb{R})$ with the topology of uniform convergence on compact sets contains all standard activations. The key objects are:

- **Translation**: $\sigma(\cdot - a)$ shifts the activation horizontally
- **Dilation**: $\sigma(b\cdot)$ scales the input
- **Superposition**: $\sum_j c_j \sigma(b_j \cdot - a_j)$ — the building block of shallow networks

The density of such superpositions (the UAT) is equivalent to showing that the **linear span** of all translated and dilated copies of $\sigma$ is dense in $C(K)$. This is a statement about the richness of the orbit of $\sigma$ under the group of affine transformations of $\mathbb{R}$.

### J.2 Connection to Measure Theory

The stochastic interpretation of GELU ($\operatorname{GELU}(z) = z \cdot P(X \le z)$ for $X \sim \mathcal{N}(0,1)$) is a statement about the expectation of a random activation under a Gaussian measure. Formally:

$$\operatorname{GELU}(z) = \mathbb{E}_{X \sim \mathcal{N}(0,1)}[z \cdot \mathbf{1}_{(-\infty, z]}(X)] = \int_{-\infty}^z z \cdot \phi(x)\,dx$$

This integral representation connects GELU to the theory of Gaussian processes: the kernel $K(z, z') = \mathbb{E}[\operatorname{GELU}(z) \cdot \operatorname{GELU}(z')]$ defines an inner product in the feature space of GELU activations, related to the arc-cosine kernels studied in Cho & Saul (2009).

### J.3 Connection to Information Theory

The softmax function appears in the derivation of the maximum entropy distribution: given a constraint on the expected value $\mathbb{E}[z_i] = \mu_i$, the distribution over a finite set $\{1, \ldots, K\}$ that maximises entropy $H = -\sum_i p_i \log p_i$ subject to $\sum_i p_i = 1$ and $\sum_i \lambda_i p_i = \text{const}$ is the Gibbs distribution:

$$p_i^* = \frac{e^{\lambda_i}}{\sum_j e^{\lambda_j}} = \operatorname{softmax}(\boldsymbol{\lambda})_i$$

This is why softmax is the "natural" output for classification: it is the maximum entropy distribution consistent with the linear constraints encoded by the logits. Lower temperature (smaller $\tau$) imposes stronger constraints (lower entropy); higher temperature relaxes the constraints.


---

## Appendix K: Advanced Topics and Open Problems

### K.1 Learnable Activations

Beyond PReLU (learned slope), several approaches learn the activation function end-to-end:

**KANs (Kolmogorov-Arnold Networks)** (Liu et al., 2024): replace fixed non-linear activations with learnable 1D spline functions on every edge of the network graph. Rather than $\mathbf{z} = \sigma(W\mathbf{x})$ (activation on nodes), KANs use $z_{ij} = \phi_{ij}(x_i)$ (activation on edges). This is inspired by the Kolmogorov-Arnold representation theorem: any continuous multivariate function can be decomposed as a sum of univariate functions. KANs demonstrate improved expressiveness for certain scientific computing tasks but remain computationally expensive at scale.

**Activation function search** (Ramachandran et al., 2017; Real et al., 2020): using evolutionary algorithms or reinforcement learning to search over the space of unary and binary operations (e.g., $z \cdot \sigma(z)$, $z^2$, $\max(z, \sigma(z))$) to find activations that outperform ReLU on specific tasks. Swish was discovered this way. The space is combinatorially large, and discovered activations do not always transfer across domains.

### K.2 Activation Functions and Overparameterisation

In overparameterised networks (more parameters than training examples), the implicit bias of gradient descent interacts with the choice of activation:

- For linear networks: gradient descent converges to the minimum-norm solution
- For ReLU networks: gradient descent exhibits a "kernel regime" (gradient flow approximated by NTK) for small learning rates and a "feature learning regime" for larger learning rates
- The transition between regimes depends on the activation's non-linearity: smooth activations (GELU) stay longer in the kernel regime, while hard activations (ReLU) switch to feature learning earlier

Understanding this tradeoff is an active research area (Yang et al., 2023; Bordelon & Pehlevan, 2022).

### K.3 Activation Functions in Physics-Informed Networks

For physics-informed neural networks (PINNs) solving PDEs, the choice of activation affects:
1. **Spectral bias**: networks preferentially learn low-frequency components. $\sin(z)$ activations (used in SIRENs, Sitzmann et al., 2020) overcome this bias by matching the frequency of the target function
2. **Smoothness**: PDEs may require solutions with specific differentiability; GELU and tanh (infinitely differentiable) are preferred over ReLU
3. **Period boundary conditions**: $\sin(z)$ activations naturally implement periodic boundary conditions

### K.4 Open Problems

1. **Theoretical characterisation of SwiGLU superiority**: Why exactly does multiplicative gating improve language modelling? The empirical evidence is clear, but a principled theoretical explanation connecting the activation's properties to the statistical structure of language is lacking.

2. **Optimal temperature for LLMs**: The optimal temperature for generation is task-dependent and often tuned manually. A theoretical framework connecting temperature to the entropy of the target distribution would enable principled temperature selection.

3. **Activation-aware architecture search at scale**: The interaction between activation choice, normalisation placement, and initialisation is complex. Systematic ablations at the 70B+ parameter scale are expensive; principled methods for predicting which activation will perform best for a given architecture would be valuable.

4. **Mechanistic understanding of superposition**: While the superposition hypothesis (Elhage et al., 2022) is compelling, it remains unclear how different activation functions affect the capacity and structure of superposed representations. Do smooth activations allow more features to be superposed, or does sparsity (ReLU) enable cleaner decomposition?


---

## Appendix L: Worked Examples — Activation Function Design

### L.1 Designing an Activation with Specific Properties

Suppose we want an activation $\sigma$ with:
1. $\sigma(0) = 0$ (zero-centred)
2. $\sigma'(0) = 1$ (unit gradient at origin)
3. $\sigma(z) \approx z$ for large $z$ (approximately linear)
4. $\sigma'(z) \ge 0$ for all $z$ (monotone)
5. $\sigma(z) \ge -c$ for some small $c > 0$ (bounded below)

ReLU satisfies 1, 3, 4, 5 (with $c = 0$) but violates 2 ($\operatorname{ReLU}'(0^+) = 1$ but not defined at 0; using $\operatorname{ReLU}'(0) = 0$ violates 2).

ELU with $\alpha = 1$ satisfies all five: $\operatorname{ELU}(0) = 0$, $\operatorname{ELU}'(0) = 1$ (by continuity: $\lim_{z\to 0^-} e^z = 1$), approximately linear for large $z$, monotone (derivative is $e^z > 0$ for $z < 0$ and 1 for $z \ge 0$), bounded below by $-1$.

SELU satisfies all five as well, with the additional property of self-normalisation.

### L.2 Comparing Activations on a Concrete Network

Consider a 5-layer network $f : \mathbb{R}^{100} \to \mathbb{R}$ with 100 neurons per hidden layer, trained on a synthetic regression task where $y = \sum_i x_i^2 + \text{noise}$.

**Expected behaviour by activation:**
- **Sigmoid**: likely to struggle due to vanishing gradients at depth 5; weights in early layers will barely update
- **Tanh**: slightly better (larger max gradient = 1), but still saturates for large pre-activations
- **ReLU**: fast convergence; risk of dead neurons if learning rate is too high or data has unusual distribution
- **GELU**: smooth gradients; non-monotone allows expressing the quadratic structure without explicit squared terms
- **SwiGLU** (if using width 64 intermediate): the gating structure allows each neuron to selectively compute over the input, potentially capturing the pairwise interaction structure more efficiently

Empirically, ReLU and GELU perform similarly for shallow tasks like this; the advantage of GELU/SwiGLU emerges more clearly at scale and on tasks requiring complex compositional reasoning.

### L.3 Understanding GELU via Taylor Expansion

Near $z = 0$, we can expand GELU:

$$\operatorname{GELU}(z) = z\Phi(z) = z \cdot \frac{1}{2}\left(1 + \operatorname{erf}\!\left(\frac{z}{\sqrt{2}}\right)\right)$$

Using $\operatorname{erf}(u) = \frac{2}{\sqrt{\pi}}\left(u - \frac{u^3}{3} + \frac{u^5}{10} - \cdots\right)$:

$$\operatorname{GELU}(z) = z \cdot \frac{1}{2}\left(1 + \frac{2}{\sqrt{\pi}} \cdot \frac{z/\sqrt{2} - (z/\sqrt{2})^3/3 + \cdots}{1}\right)$$

$$= \frac{z}{2} + \frac{z}{\sqrt{2\pi}}\left(z - \frac{z^3}{6} + \cdots\right) = \frac{z}{2} + \frac{z^2}{\sqrt{2\pi}} - \frac{z^4}{6\sqrt{2\pi}} + O(z^6)$$

Near $z = 0$:
- $\operatorname{GELU}(z) \approx z/2 + z^2/\sqrt{2\pi}$ (leading terms)
- $\operatorname{GELU}(0) = 0$ ✓
- $\operatorname{GELU}'(0) = 1/2$ ✓ (matches $\Phi(0) + 0 \cdot \phi(0) = 1/2$)
- The $z^2$ term (positive coefficient) means GELU is **convex near the origin** — unlike ReLU which is linear, GELU curves upward

This curvature is what makes GELU amenable to second-order methods and explains why it behaves differently from ReLU near zero.


---

## Appendix M: Quick-Start Implementation Guide

This guide provides minimal working code for the most common activation function tasks.

### M.1 All Activations in 30 Lines

```python
import numpy as np
from scipy.special import erf

# Classical
sigmoid = lambda z: 1 / (1 + np.exp(-np.clip(z, -500, 500)))
tanh    = np.tanh

# ReLU family
relu        = lambda z: np.maximum(0, z)
leaky_relu  = lambda z, a=0.01: np.where(z > 0, z, a * z)
elu         = lambda z, a=1.0: np.where(z >= 0, z, a * (np.exp(z) - 1))
selu_lam, selu_alp = 1.0507009873554805, 1.6732632423543772
selu        = lambda z: selu_lam * elu(z, selu_alp)

# Modern smooth
gelu        = lambda z: 0.5 * z * (1 + erf(z / np.sqrt(2)))
gelu_approx = lambda z: 0.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+0.044715*z**3)))
silu        = lambda z: z * sigmoid(z)
softplus    = lambda z: np.log1p(np.exp(z))
mish        = lambda z: z * np.tanh(softplus(z))

# Output
softmax = lambda z: (e := np.exp(z - z.max(-1, keepdims=True)),
                     e / e.sum(-1, keepdims=True))[1]

# Hard approximations
hard_sigmoid = lambda z: np.clip((z + 3) / 6, 0, 1)
hard_swish   = lambda z: z * np.clip((z + 3) / 6, 0, 1)
```

### M.2 Verification Checks

```python
z = np.array([-3., -1., 0., 1., 3.])

# Sigmoid: range (0,1), symmetric around 0.5
assert np.all((sigmoid(z) > 0) & (sigmoid(z) < 1))
assert np.allclose(sigmoid(0), 0.5)
assert np.allclose(sigmoid(z) + sigmoid(-z), 1.0)  # symmetry

# Tanh: range (-1,1), antisymmetric
assert np.all((tanh(z) > -1) & (tanh(z) < 1))
assert np.allclose(tanh(z) + tanh(-z), 0.0)  # antisymmetry
assert np.allclose(tanh(z), 2*sigmoid(2*z) - 1)  # identity

# GELU: minimum ≈ -0.170, zero at origin
assert np.allclose(gelu(np.array([0.])), 0., atol=1e-10)
assert gelu(np.array([-0.75])).min() >= -0.18  # approximately -0.170

# SiLU: minimum ≈ -0.278
assert np.allclose(silu(np.array([0.])), 0., atol=1e-10)

# SELU self-normalising
np.random.seed(42)
z_rand = np.random.normal(0, 1, 1_000_000)
s = selu(z_rand)
assert abs(s.mean()) < 0.01   # approximately zero-mean
assert abs(s.std() - 1.0) < 0.01   # approximately unit variance
print("All checks passed.")
```

