[← Back to Multivariate Calculus](../README.md) | [Next: Optimality Conditions →](../04-Optimality-Conditions/notes.md)

---

# Chain Rule and Backpropagation

> _"Backpropagation is an algorithm for computing gradients efficiently in a computational graph. At its heart, it is nothing more than the chain rule of calculus applied repeatedly."_
> — Goodfellow, Bengio & Courville, _Deep Learning_ (2016)

## Overview

The chain rule is the single most important theorem in applied mathematics for machine learning. Every time a neural network is trained — whether a two-layer MLP or a 70-billion parameter language model — the update step depends on computing gradients of a scalar loss with respect to millions or billions of parameters. That computation is **backpropagation**, and backpropagation is nothing more than the multivariate chain rule applied systematically to a computational graph.

This section develops the chain rule from first principles and then derives backpropagation rigorously. We prove every formula from scratch: the general Jacobian chain rule, the VJP (vector-Jacobian product) form that makes backprop efficient, the recurrence relation for the error signal $\boldsymbol{\delta}^{[l]}$, and the gradient formulae for every layer type that appears in modern transformers. We also analyse the pathologies — vanishing and exploding gradients — with mathematical precision, and derive the interventions that cure them: careful initialisation, residual connections, and normalisation layers.

The connection to automatic differentiation (§05) is previewed but not developed here: this section establishes *what* is computed (the chain rule gradient) while §05 establishes *how* it is computed mechanically by an AD engine.

## Prerequisites

- **Partial derivatives, gradient, directional derivative** — [§01 Partial Derivatives and Gradients](../01-Partial-Derivatives-and-Gradients/notes.md)
- **Jacobian matrix, Fréchet derivative, VJP/JVP** — [§02 Jacobians and Hessians](../02-Jacobians-and-Hessians/notes.md)
- **Single-variable chain rule, Taylor series** — [§04 Calculus Fundamentals](../../04-Calculus-Fundamentals/README.md)
- **Matrix multiplication, transpose** — [§02 Linear Algebra Basics](../../02-Linear-Algebra-Basics/README.md)

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | Chain rule verification, backprop from scratch, all layer gradients, vanishing/exploding gradient simulation, checkpointing |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises: chain rule through LoRA backward pass and BPTT |

## Learning Objectives

After completing this section, you will be able to:

1. State and prove the general chain rule $J_{f\circ g}(\mathbf{x}) = J_f(g(\mathbf{x}))\cdot J_g(\mathbf{x})$ via the Fréchet derivative
2. Explain why the VJP form $\nabla_\mathbf{x}\mathcal{L} = J_g(\mathbf{x})^\top\nabla_{g(\mathbf{x})}\mathcal{L}$ is the fundamental equation of backpropagation
3. Define a computational graph, perform forward and backward passes, handle gradient accumulation at branches
4. Derive the backprop recurrence $\boldsymbol{\delta}^{[l]} = (W^{[l+1]})^\top\boldsymbol{\delta}^{[l+1]}\odot\sigma'(\mathbf{z}^{[l]})$ from first principles
5. Derive gradients for linear layers, activations, softmax+CE (fused), LayerNorm, and scaled dot-product attention
6. Analyse vanishing/exploding gradients and derive Xavier/He initialisation from variance propagation
7. Explain residual connections as gradient highways via $J_{\text{block}} = I + J_F$
8. Implement gradient checkpointing and explain the $O(\sqrt{L})$ memory / $O(L)$ extra compute tradeoff
9. Differentiate through discrete operations using the straight-through estimator and REINFORCE
10. Trace the full backward pass through a transformer layer (attention + MLP + LayerNorm + residuals)

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 From Single-Variable to Multivariate Chain Rule](#11-from-single-variable-to-multivariate-chain-rule)
  - [1.2 Backpropagation as Iterated Chain Rule](#12-backpropagation-as-iterated-chain-rule)
  - [1.3 Historical Context](#13-historical-context)
  - [1.4 Why Backprop Defines Modern AI](#14-why-backprop-defines-modern-ai)
- [2. The Multivariate Chain Rule — Full Theory](#2-the-multivariate-chain-rule--full-theory)
  - [2.1 The General Chain Rule — Proof](#21-the-general-chain-rule--proof)
  - [2.2 Three Cases in Increasing Generality](#22-three-cases-in-increasing-generality)
  - [2.3 The VJP Form — Foundation of Backprop](#23-the-vjp-form--foundation-of-backprop)
  - [2.4 Long Chains and Telescoping Products](#24-long-chains-and-telescoping-products)
  - [2.5 Differentiating Through Discrete Operations](#25-differentiating-through-discrete-operations)
- [3. Computation Graphs](#3-computation-graphs)
  - [3.1 Formal DAG Definition](#31-formal-dag-definition)
  - [3.2 Forward Pass](#32-forward-pass)
  - [3.3 Backward Pass — Reverse Accumulation](#33-backward-pass--reverse-accumulation)
  - [3.4 Gradient Accumulation at Branching Nodes](#34-gradient-accumulation-at-branching-nodes)
  - [3.5 Dynamic vs Static Graphs](#35-dynamic-vs-static-graphs)
- [4. Backpropagation — Complete Derivation](#4-backpropagation--complete-derivation)
  - [4.1 Notation and Setup](#41-notation-and-setup)
  - [4.2 Forward Pass Equations](#42-forward-pass-equations)
  - [4.3 Output Layer Gradient](#43-output-layer-gradient)
  - [4.4 Backpropagation Recurrence — Proof](#44-backpropagation-recurrence--proof)
  - [4.5 Weight and Bias Gradients](#45-weight-and-bias-gradients)
  - [4.6 Batched Backpropagation](#46-batched-backpropagation)
- [5. Gradient Derivations for Key ML Operations](#5-gradient-derivations-for-key-ml-operations)
  - [5.1 Linear Layer](#51-linear-layer)
  - [5.2 Elementwise Activations](#52-elementwise-activations)
  - [5.3 Softmax + Cross-Entropy (Fused)](#53-softmax--cross-entropy-fused)
  - [5.4 Layer Normalisation](#54-layer-normalisation)
  - [5.5 Scaled Dot-Product Attention](#55-scaled-dot-product-attention)
  - [5.6 Embedding Lookup](#56-embedding-lookup)
- [6. Vanishing and Exploding Gradients](#6-vanishing-and-exploding-gradients)
  - [6.1 Gradient Magnitude Analysis](#61-gradient-magnitude-analysis)
  - [6.2 Activation Functions and Gradient Flow](#62-activation-functions-and-gradient-flow)
  - [6.3 Weight Initialisation — Xavier and He](#63-weight-initialisation--xavier-and-he)
  - [6.4 Residual Connections as Gradient Highways](#64-residual-connections-as-gradient-highways)
  - [6.5 Gradient Clipping](#65-gradient-clipping)
  - [6.6 Normalisation Effect on Gradient Flow](#66-normalisation-effect-on-gradient-flow)
- [7. Memory-Efficient Backpropagation](#7-memory-efficient-backpropagation)
  - [7.1 Memory Cost of Vanilla Backprop](#71-memory-cost-of-vanilla-backprop)
  - [7.2 Gradient Checkpointing](#72-gradient-checkpointing)
  - [7.3 FlashAttention Backward](#73-flashattention-backward)
  - [7.4 Mixed Precision and Gradient Scaling](#74-mixed-precision-and-gradient-scaling)
- [8. Advanced Chain Rule Topics](#8-advanced-chain-rule-topics)
  - [8.1 Backprop Through Time](#81-backprop-through-time)
  - [8.2 Implicit Differentiation Preview](#82-implicit-differentiation-preview)
  - [8.3 Straight-Through Estimator](#83-straight-through-estimator)
  - [8.4 Higher-Order Gradients](#84-higher-order-gradients)
- [9. Transformer Backpropagation in Depth](#9-transformer-backpropagation-in-depth)
  - [9.1 One Transformer Layer — Full Gradient Flow](#91-one-transformer-layer--full-gradient-flow)
  - [9.2 LoRA Backward Pass](#92-lora-backward-pass)
  - [9.3 Gradient Accumulation for Large Batches](#93-gradient-accumulation-for-large-batches)
  - [9.4 Distributed Gradient Synchronisation](#94-distributed-gradient-synchronisation)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI (2026 Perspective)](#12-why-this-matters-for-ai-2026-perspective)
- [Conceptual Bridge](#conceptual-bridge)

---

## 1. Intuition

### 1.1 From Single-Variable to Multivariate Chain Rule

The single-variable chain rule says: if $y = f(u)$ and $u = g(x)$, then

$$\frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx}$$

The intuition is rates of change compose multiplicatively. If $g$ triples its input and $f$ doubles its input, then $f \circ g$ multiplies by six.

The multivariate generalisation replaces scalars with vectors and scalar derivatives with Jacobian matrices. If $\mathbf{y} = f(\mathbf{u})$ and $\mathbf{u} = g(\mathbf{x})$, then

$$J_{f\circ g}(\mathbf{x}) = J_f(g(\mathbf{x})) \cdot J_g(\mathbf{x})$$

The product is now matrix multiplication. This is not a different rule — it is the same rule, stated in the correct language for vector-valued functions. The single-variable rule is the special case $n = m = p = 1$ where Jacobians degenerate to scalars.

```
SCALAR CHAIN RULE vs JACOBIAN CHAIN RULE
════════════════════════════════════════════════════════════════════════

  Scalar:    x ─→ g ─→ u ─→ f ─→ y
             ℝ    ℝ    ℝ    ℝ    ℝ
             dy/dx = (dy/du)(du/dx)   [scalar multiplication]

  Vector:    x ─→ g ─→ u ─→ f ─→ y
             ℝᵖ   ℝⁿ   ℝⁿ   ℝᵐ   ℝᵐ
             J_{f∘g} = J_f · J_g     [matrix multiplication]
             (m×p) = (m×n) · (n×p)

  The dimensions work out exactly like matrix multiplication.
  The chain rule IS matrix multiplication for Jacobians.

════════════════════════════════════════════════════════════════════════
```

What makes the multivariate version non-trivial is that $J_f$ must be evaluated at $g(\mathbf{x})$ — the output of the inner function — not at $\mathbf{x}$ itself. This point-dependence is where the local linear approximation lives: the Jacobian $J_f(g(\mathbf{x}))$ is the best linear approximation to $f$ at the specific point $g(\mathbf{x})$, and $J_g(\mathbf{x})$ is the best linear approximation to $g$ at $\mathbf{x}$.

### 1.2 Backpropagation as Iterated Chain Rule

A deep neural network is a long composition of functions:

$$\mathcal{L} = \ell\bigl(f_L(\cdots f_2(f_1(\mathbf{x}))\cdots)\bigr)$$

where each $f_l$ is a layer (linear + activation), $\ell$ is the loss function, and $\mathcal{L}$ is a scalar. Computing $\nabla_{\mathbf{w}^{[l]}}\mathcal{L}$ — the gradient of the loss with respect to layer $l$'s parameters — requires applying the chain rule through every layer from $l$ to $L$.

The chain rule gives:

$$\nabla_{\mathbf{w}^{[l]}}\mathcal{L} = J_{f_l,\mathbf{w}}(\mathbf{a}^{[l-1]})^\top \cdot \boldsymbol{\delta}^{[l]}$$

where $\boldsymbol{\delta}^{[l]} = \nabla_{\mathbf{z}^{[l]}}\mathcal{L}$ is the **error signal** at layer $l$, and it satisfies the **backpropagation recurrence**:

$$\boldsymbol{\delta}^{[l]} = J_{f_{l+1},\mathbf{a}}(\mathbf{z}^{[l]})^\top \cdot \boldsymbol{\delta}^{[l+1]}$$

This recurrence propagates the error signal backward from layer $L$ to layer 1 — hence "backpropagation." At each step, we multiply by the transposed Jacobian of the next layer. The entire algorithm is:

1. **Forward pass**: compute and store $\mathbf{z}^{[l]}$, $\mathbf{a}^{[l]}$ for $l = 1, \ldots, L$
2. **Initialise**: $\boldsymbol{\delta}^{[L]} = \nabla_{\mathbf{z}^{[L]}}\mathcal{L}$
3. **Backward pass**: compute $\boldsymbol{\delta}^{[l]}$ for $l = L-1, \ldots, 1$ using the recurrence
4. **Gradients**: extract $\nabla_{W^{[l]}}\mathcal{L}$ from $\boldsymbol{\delta}^{[l]}$ and $\mathbf{a}^{[l-1]}$

Backpropagation is not a fundamentally different concept from the chain rule. It is the chain rule, applied efficiently by sharing intermediate computations (the error signals $\boldsymbol{\delta}^{[l]}$) across all parameters in a layer.

### 1.3 Historical Context

| Year | Contributor | Development |
|------|-------------|-------------|
| 1676 | Leibniz | Differential calculus; first statement of the single-variable chain rule |
| 1755 | Euler | Extended to multiple variables |
| 1960 | Kelley | Gradient computation for optimal control (independent discovery of backprop concept) |
| 1970 | Linnainmaa | First complete description of reverse-mode automatic differentiation for computing gradients |
| 1974 | Werbos | First application to neural networks in his PhD thesis |
| 1986 | Rumelhart, Hinton, Williams | Popularised backpropagation in "Learning representations by back-propagating errors" — the paper that launched the neural network revolution |
| 1989 | LeCun | Applied backprop to convolutional networks for handwritten digit recognition |
| 2012 | Krizhevsky, Sutskever, Hinton | AlexNet demonstrated GPU-accelerated backprop at scale — kicked off the deep learning era |
| 2015 | Google Brain, Facebook AI | PyTorch/TensorFlow: automatic differentiation engines that compute backprop automatically |
| 2017 | Vaswani et al. | Transformer: backprop through multi-head attention; the architecture underlying GPT, BERT, LLaMA |
| 2021 | Hu et al. (LoRA) | Parameter-efficient fine-tuning by limiting gradient flow to low-rank subspaces |
| 2022 | Dao et al. (FlashAttention) | Recompute activations during backward to avoid materialising the $N\times N$ attention matrix |

### 1.4 Why Backprop Defines Modern AI

Every large language model, image classifier, and diffusion model trained today relies on backpropagation for every gradient update. The scale is staggering: training GPT-4 reportedly required $\sim 10^{25}$ floating-point operations, the vast majority of which are forward and backward passes through the transformer network.

**Backprop enables gradient-based learning at scale** because its cost is proportional to the cost of the forward pass — typically $O(n \cdot \text{FLOPs}(f))$ where $n$ is the number of parameters. Alternative approaches (finite differences, evolution strategies, zeroth-order methods) are orders of magnitude more expensive.

**Three properties make backprop indispensable:**

1. **Efficiency**: One backward pass computes $\nabla_{\boldsymbol{\theta}}\mathcal{L}$ for all $|\boldsymbol{\theta}| \sim 10^{10}$ parameters simultaneously. Finite differences would need $10^{10}$ forward passes.

2. **Exactness**: Unlike finite differences, backprop computes the exact gradient (up to floating-point precision), not an approximation.

3. **Composability**: Any differentiable function composed of differentiable primitives has an automatically computable gradient. This is why PyTorch/JAX can differentiate arbitrary Python code that uses differentiable operations.

**For AI in 2026:** The gradient is the workhorse of every training algorithm: SGD, Adam, AdaGrad, Muon, SOAP — all are gradient-based. Fine-tuning (LoRA, QLoRA, DoRA), RLHF (PPO, DPO, GRPO), distillation, and continual learning all depend on backprop. Even methods that appear gradient-free (evolutionary strategies, black-box optimisation) are often used *because* they approximate the gradient in settings where backprop is unavailable (non-differentiable objectives, external APIs).

---

## 2. The Multivariate Chain Rule — Full Theory

### 2.1 The General Chain Rule — Proof

We prove the chain rule using the Fréchet derivative from §02. Recall:

**Definition.** $f: U \subseteq \mathbb{R}^n \to \mathbb{R}^m$ is Fréchet differentiable at $\mathbf{x}$ if there exists a linear map $L_\mathbf{x}: \mathbb{R}^n \to \mathbb{R}^m$ such that

$$\lim_{\|\boldsymbol{\delta}\|\to 0}\frac{\|f(\mathbf{x}+\boldsymbol{\delta}) - f(\mathbf{x}) - L_\mathbf{x}\boldsymbol{\delta}\|}{\|\boldsymbol{\delta}\|} = 0$$

The matrix of $L_\mathbf{x}$ is the Jacobian $J_f(\mathbf{x})$.

**Theorem (Chain Rule).** Let $g: \mathbb{R}^p \to \mathbb{R}^n$ be Fréchet differentiable at $\mathbf{x}$, and $f: \mathbb{R}^n \to \mathbb{R}^m$ be Fréchet differentiable at $g(\mathbf{x})$. Then $h = f \circ g: \mathbb{R}^p \to \mathbb{R}^m$ is Fréchet differentiable at $\mathbf{x}$ and

$$J_h(\mathbf{x}) = J_f(g(\mathbf{x}))\cdot J_g(\mathbf{x})$$

**Proof.** Let $\mathbf{u}_0 = g(\mathbf{x})$. We need to show that $J_f(\mathbf{u}_0)J_g(\mathbf{x})$ is the Fréchet derivative of $h$ at $\mathbf{x}$. Write:

$$h(\mathbf{x}+\boldsymbol{\delta}) - h(\mathbf{x}) = f(g(\mathbf{x}+\boldsymbol{\delta})) - f(g(\mathbf{x}))$$

Let $\boldsymbol{\eta} = g(\mathbf{x}+\boldsymbol{\delta}) - g(\mathbf{x})$. Since $g$ is Fréchet differentiable:

$$\boldsymbol{\eta} = J_g(\mathbf{x})\boldsymbol{\delta} + \mathbf{r}_g(\boldsymbol{\delta}), \quad \frac{\|\mathbf{r}_g(\boldsymbol{\delta})\|}{\|\boldsymbol{\delta}\|} \to 0 \text{ as } \boldsymbol{\delta} \to \mathbf{0}$$

Now apply Fréchet differentiability of $f$ at $\mathbf{u}_0$:

$$f(\mathbf{u}_0 + \boldsymbol{\eta}) - f(\mathbf{u}_0) = J_f(\mathbf{u}_0)\boldsymbol{\eta} + \mathbf{r}_f(\boldsymbol{\eta}), \quad \frac{\|\mathbf{r}_f(\boldsymbol{\eta})\|}{\|\boldsymbol{\eta}\|} \to 0 \text{ as } \boldsymbol{\eta} \to \mathbf{0}$$

Substituting:

$$h(\mathbf{x}+\boldsymbol{\delta}) - h(\mathbf{x}) = J_f(\mathbf{u}_0)(J_g(\mathbf{x})\boldsymbol{\delta} + \mathbf{r}_g(\boldsymbol{\delta})) + \mathbf{r}_f(\boldsymbol{\eta})$$

$$= J_f(\mathbf{u}_0)J_g(\mathbf{x})\boldsymbol{\delta} + \underbrace{J_f(\mathbf{u}_0)\mathbf{r}_g(\boldsymbol{\delta}) + \mathbf{r}_f(\boldsymbol{\eta})}_{\text{remainder}}$$

We show the remainder is $o(\|\boldsymbol{\delta}\|)$:
- First term: $\|J_f(\mathbf{u}_0)\mathbf{r}_g(\boldsymbol{\delta})\| \leq \|J_f(\mathbf{u}_0)\|\cdot\|\mathbf{r}_g(\boldsymbol{\delta})\| = o(\|\boldsymbol{\delta}\|)$.
- Second term: Since $\|\boldsymbol{\eta}\| \leq \|J_g\|\|\boldsymbol{\delta}\| + \|\mathbf{r}_g\| = O(\|\boldsymbol{\delta}\|)$, we have $\|\mathbf{r}_f(\boldsymbol{\eta})\| = o(\|\boldsymbol{\eta}\|) = o(\|\boldsymbol{\delta}\|)$.

Therefore $J_h(\mathbf{x}) = J_f(g(\mathbf{x}))J_g(\mathbf{x})$. $\square$

**When the chain rule fails.** The chain rule requires both $g$ at $\mathbf{x}$ and $f$ at $g(\mathbf{x})$ to be Fréchet differentiable. If either fails — for example at a ReLU kink where $\mathbf{z}^{[l]} = 0$ — the classical chain rule does not apply. In practice, these measure-zero sets are handled by choosing a subgradient (any element of the Clarke subdifferential), which is what deep learning frameworks do automatically.

### 2.2 Three Cases in Increasing Generality

**Case 1: Scalar composition $h: \mathbb{R} \to \mathbb{R}$.** $h(x) = f(g(x))$ where $f, g: \mathbb{R} \to \mathbb{R}$. Jacobians are $1\times 1$ = scalars, so $h'(x) = f'(g(x)) \cdot g'(x)$.

**Case 2: Scalar loss of a vector function.** $h: \mathbb{R}^n \to \mathbb{R}$, $h(\mathbf{x}) = f(g(\mathbf{x}))$ where $g: \mathbb{R}^n \to \mathbb{R}^m$ and $f: \mathbb{R}^m \to \mathbb{R}$. Jacobians: $J_g \in \mathbb{R}^{m\times n}$ and $J_f \in \mathbb{R}^{1\times m}$ (a row vector). So:

$$J_h = J_f(g(\mathbf{x})) \cdot J_g(\mathbf{x}) \in \mathbb{R}^{1\times n}$$

Taking the transpose: $\nabla_\mathbf{x} h = J_g(\mathbf{x})^\top \nabla_{g(\mathbf{x})} f$ — the gradient of $h$ with respect to $\mathbf{x}$ is the transposed Jacobian of $g$ times the gradient of $f$. **This is the VJP equation, the core of backprop.**

**Case 3: Vector composition $h: \mathbb{R}^p \to \mathbb{R}^m$.** The most general case; Jacobians are full matrices and the chain rule is full matrix multiplication:

$$J_h(\mathbf{x}) = J_f(g(\mathbf{x})) \cdot J_g(\mathbf{x}) \in \mathbb{R}^{m\times p}$$

The dimensions verify: $(m\times p) = (m\times n)(n\times p)$. The "inner dimension" $n$ (the dimension of the intermediate space $\mathbb{R}^n$) cancels in the product, exactly as in matrix multiplication.

### 2.3 The VJP Form — Foundation of Backprop

**Definition (VJP).** For $g: \mathbb{R}^n \to \mathbb{R}^m$ and a "cotangent" vector $\mathbf{u} \in \mathbb{R}^m$, the **vector-Jacobian product** is:

$$\text{VJP}(g, \mathbf{x}, \mathbf{u}) = J_g(\mathbf{x})^\top \mathbf{u} \in \mathbb{R}^n$$

**Why this is the right primitive for backprop.** For a scalar loss $\mathcal{L}: \mathbb{R}^m \to \mathbb{R}$ composed with $g: \mathbb{R}^n \to \mathbb{R}^m$:

$$\nabla_\mathbf{x}(\mathcal{L} \circ g) = J_g(\mathbf{x})^\top \nabla_{g(\mathbf{x})}\mathcal{L} = \text{VJP}(g, \mathbf{x}, \nabla_{g(\mathbf{x})}\mathcal{L})$$

The gradient of the composed function with respect to the input is the VJP of the inner function, with the cotangent being the gradient of the outer function.

**The backprop recursion is a chain of VJPs.** For $\mathcal{L} = \ell \circ f_L \circ \cdots \circ f_1$:

$$\nabla_{\mathbf{a}^{[l]}}\mathcal{L} = \text{VJP}(f_{l+1}, \mathbf{a}^{[l]}, \nabla_{\mathbf{a}^{[l+1]}}\mathcal{L}) = J_{f_{l+1}}(\mathbf{a}^{[l]})^\top \nabla_{\mathbf{a}^{[l+1]}}\mathcal{L}$$

Starting from $\nabla_{\mathbf{a}^{[L]}}\mathcal{L}$ and applying VJPs from right to left computes all intermediate gradients.

**Cost comparison.** Computing $\nabla_\mathbf{x}\mathcal{L}$ for $\mathcal{L}: \mathbb{R}^n \to \mathbb{R}$:
- **JVP (forward mode)**: requires $n$ passes (one per input dimension). Cost: $n \times O(\text{FLOPs}(g))$.
- **VJP (reverse mode)**: requires $1$ pass. Cost: $O(\text{FLOPs}(g))$.

For $n \sim 10^{10}$ parameters, reverse mode is $10^{10}$ times cheaper. This asymmetry is why all gradient-based deep learning uses reverse mode (backprop).

### 2.4 Long Chains and Telescoping Products

For a depth-$L$ network $h = f_L \circ \cdots \circ f_1$, the Jacobian of $h$ with respect to $\mathbf{x}$ is:

$$J_h(\mathbf{x}) = J_{f_L}(\mathbf{a}^{[L-1]}) \cdot J_{f_{L-1}}(\mathbf{a}^{[L-2]}) \cdots J_{f_1}(\mathbf{x})$$

This is a product of $L$ matrices. The spectral norm of the product satisfies:

$$\|J_h\|_2 \leq \prod_{l=1}^L \|J_{f_l}\|_2$$

If each $\|J_{f_l}\|_2 = \rho$, then $\|J_h\|_2 \leq \rho^L$. For $\rho < 1$, the gradient vanishes exponentially; for $\rho > 1$, it explodes. This is the mathematical source of the vanishing/exploding gradient problem (§6).

**Efficient computation: reverse order.** In the forward direction, we compute $\mathbf{a}^{[1]}, \ldots, \mathbf{a}^{[L]}$ left to right. In the backward direction, we compute the error signals $\boldsymbol{\delta}^{[L]}, \boldsymbol{\delta}^{[L-1]}, \ldots, \boldsymbol{\delta}^{[1]}$ right to left, reusing stored activations. The key observation: at step $l$, we only need $\boldsymbol{\delta}^{[l+1]}$ and the stored activation $\mathbf{a}^{[l]}$ (or $\mathbf{z}^{[l]}$) — we do not need to recompute from scratch.

### 2.5 Differentiating Through Discrete Operations

Some operations in neural networks are discontinuous or discrete: argmax (in beam search), rounding/quantisation (in QAT), sampling (in VAEs and RL). The chain rule does not directly apply.

**Straight-Through Estimator (STE).** For a quantisation function $q(x) = \lfloor x \rceil$ (round to nearest integer), the derivative is $q'(x) = 0$ almost everywhere, giving zero gradient. The STE replaces the "true" zero gradient with 1 during the backward pass:

$$\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial q(x)} \quad \text{(pass gradient through as if $q$ were the identity)}$$

In code: `y = round(x).detach() + x - x.detach()` — adds $x$ in the forward pass (cancels) but contributes its gradient in the backward pass. STE is used in VQ-VAE, binary neural networks, and quantisation-aware training.

**REINFORCE (score function estimator).** For a stochastic node $y \sim p_\theta(y|x)$ and loss $\mathcal{L}(y)$, the gradient of $\mathbb{E}[\mathcal{L}(y)]$ with respect to $\theta$ is:

$$\nabla_\theta \mathbb{E}_{y \sim p_\theta}[\mathcal{L}(y)] = \mathbb{E}_{y \sim p_\theta}[\mathcal{L}(y) \nabla_\theta \log p_\theta(y|x)]$$

This allows gradient estimation without differentiating through the sampling step. Used in RLHF (PPO, GRPO) and variational inference. High variance; mitigated by baselines.


---

## 3. Computation Graphs

### 3.1 Formal DAG Definition

A **computation graph** is a directed acyclic graph $G = (V, E)$ encoding how scalar or tensor quantities depend on one another.

**Nodes** $V$ partition into three types:

| Type | Symbol | Role |
|------|--------|------|
| Input nodes | $v_1,\ldots,v_n$ | Hold model inputs and parameters; no incoming edges |
| Intermediate nodes | $v_{n+1},\ldots,v_{N-1}$ | Hold computed activations; receive edges from their operands |
| Output node | $v_N$ | Holds the scalar loss $\mathcal{L}$; required to be scalar for standard backprop |

**Edges** $(u, v) \in E$ encode data dependency: $v = \phi(u_1, \ldots, u_k)$ for some primitive $\phi$.  Each edge carries an implicit *local Jacobian* $\partial v / \partial u_i$.

**Primitive operations** are the atomic building blocks with known local gradients:

```
PRIMITIVE OPERATIONS AND THEIR LOCAL GRADIENTS
═══════════════════════════════════════════════════════════════════════

  Operation          Forward           Local gradient (wrt input i)
  ─────────────────────────────────────────────────────────────────
  z = x + y          z = x + y         ∂z/∂x = 1,  ∂z/∂y = 1
  z = x · y          z = xy            ∂z/∂x = y,  ∂z/∂y = x
  z = exp(x)         z = eˣ            ∂z/∂x = eˣ
  z = log(x)         z = ln x          ∂z/∂x = 1/x
  z = relu(x)        z = max(0,x)      ∂z/∂x = 𝟙[x>0]  (a.e.)
  z = W x + b        Wx+b              ∂z/∂W = xᵀ (as outer),  ∂z/∂x = Wᵀ
  z = softmax(x)     eˣⁱ/Σeˣʲ         diag(p) - ppᵀ  (see §02)
  ─────────────────────────────────────────────────────────────────

  Every deep learning framework maintains a lookup table of these
  primitives together with their vjp implementations.

═══════════════════════════════════════════════════════════════════════
```

**Topological ordering** — a linear ordering $\pi$ of $V$ such that for every edge $(u,v) \in E$, $u$ appears before $v$ in $\pi$.  Topological order exists iff $G$ is acyclic (Kahn's algorithm, 1962).  Both the forward pass and the backward pass respect topological order (the latter in reverse).

**For AI:** Every modern deep learning framework (PyTorch, JAX, TensorFlow) represents a neural network as a computation graph.  PyTorch builds the graph dynamically during the forward pass via the autograd tape; JAX traces the graph statically via XLA compilation.

### 3.2 Forward Pass — Value Propagation

The **forward pass** evaluates all node values in topological order, caching intermediates required by the backward pass.

**Algorithm (Forward Pass):**
```
Input:  graph G = (V, E),  input values {x₁,…,xₙ}
Output: loss value v_N,    cache of intermediates

  For v in topological_order(G):
    if v is an input node:
      cache[v] = x_v  (given)
    else:
      cache[v] = φ_v(cache[u₁], …, cache[uₖ])
                 where u₁,…,uₖ = parents(v)

  return cache[v_N]
```

**Memory cost of a naive forward pass:** Caching all intermediates for backprop costs $O(N)$ memory where $N$ is the number of nodes.  For a transformer with $L=96$ layers and activations of size $(B, T, d)$, this is approximately:

$$\text{Memory} = L \cdot B \cdot T \cdot d \cdot \text{sizeof(float16)} \approx 96 \times 32 \times 4096 \times 8192 \times 2 \approx 200\,\text{GB}$$

This is why gradient checkpointing (§7) is essential for large models.

**What gets cached?** A memory-optimal forward pass only caches values that appear in at least one local gradient formula.  For a linear layer $\mathbf{z} = W\mathbf{x} + \mathbf{b}$, the backward needs $\mathbf{x}$ (to compute $\nabla_W$) but not $\mathbf{z}$ (already accumulated into the output).

### 3.3 Backward Pass — Gradient Accumulation

The backward pass evaluates **adjoint values** $\bar{v} = \partial \mathcal{L}/\partial v$ for every node, in **reverse topological order**.

Define the adjoint of node $v$ as:

$$\bar{v} \;:=\; \frac{\partial \mathcal{L}}{\partial v}$$

where we treat $v$ as a scalar intermediate (extending to tensors componentwise).

**Initialisation:** $\bar{v}_N = 1$ (the loss node).

**Backward recurrence:** For a node $v$ with children (successors) $c_1, \ldots, c_m$ — nodes that depend on $v$:

$$\bar{v} = \sum_{j=1}^{m} \bar{c}_j \cdot \frac{\partial c_j}{\partial v}$$

This is exactly the chain rule applied in reverse.

**Algorithm (Backward Pass):**
```
Input:  graph G,  cache from forward pass
Output: ∂ℒ/∂v for all v ∈ V

  adjoint[v_N] ← 1
  For v in reverse_topological_order(G):
    For each parent u of v:
      adjoint[u] += adjoint[v] · ∂v/∂u(cache)
                    ─────────────────────────────
                    local_vjp(v, u, adjoint[v])

  return {adjoint[u] : u is a parameter node}
```

The key observation: each edge $(u, v)$ requires only:
1. The cached *forward value* at $u$ (for the local gradient formula)
2. The downstream adjoint $\bar{v}$ (for the VJP multiplication)

### 3.4 Gradient Accumulation at Branching Nodes

A **fan-out node** $u$ has multiple children $c_1, \ldots, c_m$.  The correct gradient is the *sum* of contributions:

$$\bar{u} = \sum_{j=1}^{m} \bar{c}_j \cdot \frac{\partial c_j}{\partial u}$$

**Proof:** By the total derivative,

$$\frac{\partial \mathcal{L}}{\partial u} = \sum_{j=1}^{m} \frac{\partial \mathcal{L}}{\partial c_j} \cdot \frac{\partial c_j}{\partial u} = \sum_{j=1}^{m} \bar{c}_j \cdot \frac{\partial c_j}{\partial u} \qquad \square$$

**Example — residual connection:**

```
RESIDUAL BRANCH: u feeds into both F(u) and the skip path
═══════════════════════════════════════════════════════════════

          u
         / \
        /   \
      F(u)   \  ← identity skip
        \   /
         \ /
    z = F(u) + u

  Forward:   z = F(u) + u
  Backward:  ū = z̄ · ∂F(u)/∂u  +  z̄ · 1
                = z̄ · J_F(u)  +  z̄

  The identity skip guarantees a gradient highway:
  even if J_F(u) ≈ 0 (saturated layer), z̄ flows back unchanged.

═══════════════════════════════════════════════════════════════
```

This is the deep reason residual networks (He et al., 2016) solved the vanishing gradient problem: the skip connection creates a constant-1 term in the backward accumulation, guaranteeing $\bar{u} \geq \bar{z}$ in gradient magnitude.

### 3.5 Dynamic vs Static Graphs

Two design philosophies produce different tradeoffs:

```
DYNAMIC GRAPHS (PyTorch eager mode)    STATIC GRAPHS (JAX jit / TF graph)
══════════════════════════════════     ══════════════════════════════════

Graph built anew each forward pass     Graph compiled once, reused

✓ Natural Python control flow          ✓ XLA/CUDA fusion, kernel merging
✓ Easy debugging (print anywhere)      ✓ Memory-optimal buffer allocation
✓ Variable-length sequences trivial    ✓ Can export/serve without Python
✗ Graph construction overhead          ✗ Tracing must handle all branches
✗ Less compiler optimisation           ✗ Python side-effects invisible

Examples: PyTorch, early Chainer       Examples: JAX jit, TF2 tf.function,
                                        ONNX Runtime, TensorRT

══════════════════════════════════     ══════════════════════════════════
```

**For transformers:** Most production LLM training uses `torch.compile` (PyTorch 2.0+) which bridges the two: eager-mode graph construction with TorchDynamo tracing and inductor backend compilation, recovering ~30-50% throughput from kernel fusion.


---

## 4. Backpropagation

### 4.1 Network Notation

Consider a feedforward neural network with $L$ layers.  Define:

| Symbol | Meaning |
|--------|---------|
| $\mathbf{a}^{[0]} = \mathbf{x}$ | Input vector, dimension $n_0$ |
| $W^{[l]} \in \mathbb{R}^{n_l \times n_{l-1}}$ | Weight matrix for layer $l$ |
| $\mathbf{b}^{[l]} \in \mathbb{R}^{n_l}$ | Bias vector for layer $l$ |
| $\mathbf{z}^{[l]} = W^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$ | Pre-activation (linear combination) |
| $\mathbf{a}^{[l]} = \sigma(\mathbf{z}^{[l]})$ | Post-activation (elementwise) |
| $\hat{\mathbf{y}} = \mathbf{a}^{[L]}$ | Network output |
| $\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y})$ | Scalar loss |

The forward pass computes $\mathbf{z}^{[l]}$ and $\mathbf{a}^{[l]}$ for $l = 1, \ldots, L$.

### 4.2 Forward Equations

$$\mathbf{z}^{[l]} = W^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}, \qquad l = 1,\ldots,L$$

$$\mathbf{a}^{[l]} = \sigma^{[l]}(\mathbf{z}^{[l]}), \qquad l = 1,\ldots,L$$

$$\hat{\mathbf{y}} = \mathbf{a}^{[L]}$$

Cache for backward: $\{\mathbf{z}^{[l]}, \mathbf{a}^{[l-1]}\}_{l=1}^{L}$.

### 4.3 Output Layer Gradient

For cross-entropy loss with softmax output, the output gradient has the celebrated clean form (derived in §5.3):

$$\boldsymbol{\delta}^{[L]} := \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}} = \hat{\mathbf{y}} - \mathbf{y}$$

where $\mathbf{y}$ is the one-hot label.  This combines the softmax Jacobian with the cross-entropy gradient into a single elegant expression.

For MSE loss ($\mathcal{L} = \tfrac{1}{2}\|\hat{\mathbf{y}} - \mathbf{y}\|^2$) with linear output:

$$\boldsymbol{\delta}^{[L]} = \hat{\mathbf{y}} - \mathbf{y}$$

(Same form, different derivation — a useful coincidence that makes implementation uniform.)

### 4.4 Backpropagation Recurrence — Proof

**Define the error signal:**
$$\boldsymbol{\delta}^{[l]} \;:=\; \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} \in \mathbb{R}^{n_l}$$

**Theorem (Backpropagation Recurrence):**
$$\boldsymbol{\delta}^{[l]} = \left(W^{[l+1]}\right)^\top \boldsymbol{\delta}^{[l+1]} \odot \sigma'^{[l]}(\mathbf{z}^{[l]})$$

**Proof:** Apply the chain rule from $\mathcal{L}$ to $\mathbf{z}^{[l]}$ via $\mathbf{a}^{[l]}$ and $\mathbf{z}^{[l+1]}$:

$$\boldsymbol{\delta}^{[l]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l+1]}} \cdot \frac{\partial \mathbf{z}^{[l+1]}}{\partial \mathbf{z}^{[l]}}$$

Step 1: $\partial \mathcal{L}/\partial \mathbf{z}^{[l+1]} = (\boldsymbol{\delta}^{[l+1]})^\top$.

Step 2: $\mathbf{z}^{[l+1]} = W^{[l+1]}\mathbf{a}^{[l]} + \mathbf{b}^{[l+1]} = W^{[l+1]}\sigma(\mathbf{z}^{[l]}) + \mathbf{b}^{[l+1]}$.

$$\frac{\partial \mathbf{z}^{[l+1]}_i}{\partial \mathbf{z}^{[l]}_j} = W^{[l+1]}_{ij} \cdot \sigma'(\mathbf{z}^{[l]}_j)$$

In matrix form: $J = W^{[l+1]} \operatorname{diag}(\sigma'(\mathbf{z}^{[l]}))$.

Step 3: Multiply by $(\boldsymbol{\delta}^{[l+1]})^\top$ and transpose to get column vector $\boldsymbol{\delta}^{[l]}$:

$$\boldsymbol{\delta}^{[l]} = J^\top \boldsymbol{\delta}^{[l+1]} = \operatorname{diag}(\sigma'(\mathbf{z}^{[l]})) \left(W^{[l+1]}\right)^\top \boldsymbol{\delta}^{[l+1]} = \left(W^{[l+1]}\right)^\top \boldsymbol{\delta}^{[l+1]} \odot \sigma'(\mathbf{z}^{[l]}) \qquad \square$$

The $\odot$ (elementwise) product arises because $\sigma$ is applied elementwise — its Jacobian is diagonal.

### 4.5 Weight and Bias Gradients

Once error signals $\boldsymbol{\delta}^{[l]}$ are computed, parameter gradients follow immediately:

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^\top \in \mathbb{R}^{n_l \times n_{l-1}}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]} \in \mathbb{R}^{n_l}$$

**Derivation of weight gradient:**

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}_{ij}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}_i} \cdot \frac{\partial \mathbf{z}^{[l]}_i}{\partial W^{[l]}_{ij}} = \delta^{[l]}_i \cdot a^{[l-1]}_j$$

Collecting over all $i,j$: $\nabla_{W^{[l]}} \mathcal{L} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^\top$.

This is an **outer product** — the gradient is rank-1 for a single sample.  For a batch of $B$ samples it averages to higher rank.

### 4.6 Batched Backpropagation

With a mini-batch $\{(\mathbf{x}^{(b)}, \mathbf{y}^{(b)})\}_{b=1}^{B}$, stack inputs into a matrix $X \in \mathbb{R}^{n_0 \times B}$.

The forward pass becomes:
$$Z^{[l]} = W^{[l]} A^{[l-1]} + \mathbf{b}^{[l]} \mathbf{1}^\top, \quad A^{[l]} = \sigma(Z^{[l]})$$

where $A^{[l]} \in \mathbb{R}^{n_l \times B}$.

The backward pass produces $\Delta^{[l]} \in \mathbb{R}^{n_l \times B}$ (error signals for all samples simultaneously).

Weight gradient for the batch:
$$\frac{\partial \mathcal{L}_\text{batch}}{\partial W^{[l]}} = \frac{1}{B} \Delta^{[l]} (A^{[l-1]})^\top$$

This is a single matrix multiplication, making batched backprop efficient on GPUs which excel at large GEMM (general matrix multiplication) operations.


---

## 5. Gradient Derivations for Standard Layers

### 5.1 Linear Layer

**Forward:** $\mathbf{z} = W\mathbf{x} + \mathbf{b}$, where $W \in \mathbb{R}^{m \times n}$, $\mathbf{x} \in \mathbb{R}^n$.

**Upstream gradient:** $\bar{\mathbf{z}} = \partial \mathcal{L}/\partial \mathbf{z} \in \mathbb{R}^m$.

**VJP (backward):**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = W^\top \bar{\mathbf{z}} \in \mathbb{R}^n$$

$$\frac{\partial \mathcal{L}}{\partial W} = \bar{\mathbf{z}} \mathbf{x}^\top \in \mathbb{R}^{m \times n}$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \bar{\mathbf{z}} \in \mathbb{R}^m$$

**Derivation of $\partial \mathcal{L}/\partial \mathbf{x}$:** Each $z_i = \sum_k W_{ik} x_k + b_i$, so $\partial z_i / \partial x_j = W_{ij}$.  By VJP: $\partial \mathcal{L}/\partial x_j = \sum_i \bar{z}_i W_{ij} = (W^\top \bar{\mathbf{z}})_j$.

**For AI:** In a transformer with hidden dim $d$ and MLP expansion $4d$: the two linear layers in FFN pass gradients back with $W^\top$ operations — same cost as the forward GEMM.  Gradient computation for $W$ is also a GEMM.

### 5.2 Activation Functions

For elementwise $\mathbf{a} = \sigma(\mathbf{z})$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \bar{\mathbf{a}} \odot \sigma'(\mathbf{z})$$

Gradient formulas for common activations:

| Activation | $\sigma(z)$ | $\sigma'(z)$ | Notes |
|------------|-------------|--------------|-------|
| ReLU | $\max(0,z)$ | $\mathbf{1}[z>0]$ | Sparse gradient; "dead neurons" if $z<0$ always |
| Sigmoid | $1/(1+e^{-z})$ | $\sigma(z)(1-\sigma(z))$ | Saturates; max gradient 0.25 at $z=0$ |
| Tanh | $\tanh(z)$ | $1-\tanh^2(z)$ | Saturates; max gradient 1 at $z=0$ |
| GELU | $z\Phi(z)$ | $\Phi(z) + z\phi(z)$ | $\Phi$ = Gaussian CDF; smooth at 0 |
| SiLU/Swish | $z \sigma(z)$ | $\sigma(z)(1+z(1-\sigma(z)))$ | Used in LLaMA, Mistral |
| SoftPlus | $\log(1+e^z)$ | $\sigma(z)$ | Smooth ReLU; gradient never zero |

**GELU** (Hendrycks & Gimpel, 2016) is the standard activation in GPT-2/3, BERT, and most modern LLMs.  It gates the input by its own probability under a Gaussian, producing richer gradient structure than ReLU.

### 5.3 Fused Softmax + Cross-Entropy Gradient

**Setup:** Output logits $\mathbf{z} \in \mathbb{R}^K$, softmax probabilities $\mathbf{p} = \text{softmax}(\mathbf{z})$, true label $y \in \{1,\ldots,K\}$, loss $\mathcal{L} = -\log p_y$.

**Claim:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{e}_y$$

where $\mathbf{e}_y$ is the $y$-th standard basis vector.

**Proof:** Write $\mathcal{L} = -\log p_y = -z_y + \log \sum_k e^{z_k}$.

$$\frac{\partial \mathcal{L}}{\partial z_j} = -\mathbf{1}[j=y] + \frac{e^{z_j}}{\sum_k e^{z_k}} = p_j - \mathbf{1}[j=y] = (\mathbf{p} - \mathbf{e}_y)_j \qquad \square$$

This direct derivation bypasses the softmax Jacobian computation entirely, which is why modern frameworks implement cross-entropy as a fused operation.  For numerical stability, the $\log\sum\exp$ is computed with the log-sum-exp trick: $\log\sum_k e^{z_k} = m + \log\sum_k e^{z_k - m}$ where $m = \max_k z_k$.

### 5.4 LayerNorm Gradient

**Forward:** LayerNorm normalises each token independently:

$$\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad \mathbf{y} = \boldsymbol{\gamma} \odot \hat{\mathbf{x}} + \boldsymbol{\beta}$$

where $\mu = \tfrac{1}{d}\sum x_i$, $\sigma^2 = \tfrac{1}{d}\sum (x_i - \mu)^2$.

**Backward:** Let $\bar{\mathbf{y}} = \partial \mathcal{L}/\partial \mathbf{y}$ be the upstream gradient.

$$\frac{\partial \mathcal{L}}{\partial \boldsymbol{\gamma}} = \bar{\mathbf{y}} \odot \hat{\mathbf{x}}, \qquad \frac{\partial \mathcal{L}}{\partial \boldsymbol{\beta}} = \bar{\mathbf{y}}$$

For the input gradient, define $\bar{\mathbf{x}}_\text{norm} = \bar{\mathbf{y}} \odot \boldsymbol{\gamma}$.  The full gradient through the normalisation is:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{1}{\sqrt{\sigma^2+\epsilon}} \left( \bar{\mathbf{x}}_\text{norm} - \frac{1}{d}\mathbf{1}^\top \bar{\mathbf{x}}_\text{norm} \cdot \mathbf{1} - \frac{1}{d}\hat{\mathbf{x}} \odot (\hat{\mathbf{x}}^\top \bar{\mathbf{x}}_\text{norm}) \cdot \mathbf{1} \right)$$

This expression subtracts mean and mean-of-hadamard terms, reflecting that LayerNorm's Jacobian projects out two degrees of freedom (§02 exercises).

**For AI:** LayerNorm appears in every transformer layer (pre-norm placement in modern architectures like GPT-NeoX, LLaMA).  The gradient through LayerNorm is never zero — it always passes signal, unlike BatchNorm which can become degenerate at small batch sizes.

### 5.5 Dot-Product Attention Gradient

**Forward (simplified single-head):**

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$
$$S = QK^\top / \sqrt{d_k}, \quad P = \text{softmax}(S), \quad O = PV$$

**Backward:** Given upstream $\bar{O} \in \mathbb{R}^{T \times d_v}$:

$$\bar{V} = P^\top \bar{O}, \quad \bar{P} = \bar{O} V^\top$$

$$\bar{S} = \text{softmax\_backward}(P, \bar{P}) = P \odot (\bar{P} - (P \odot \bar{P})\mathbf{1}\mathbf{1}^\top) \cdot \frac{1}{\sqrt{d_k}}$$

$$\bar{Q} = \bar{S} K, \quad \bar{K} = \bar{S}^\top Q$$

Then $\nabla_{W_Q} = X^\top \bar{Q}$, and similarly for $W_K$, $W_V$.

**Critical memory issue:** Storing $P \in \mathbb{R}^{T \times T}$ for the backward costs $O(T^2)$ — this is what FlashAttention avoids by recomputing $P$ from $Q, K$ during the backward pass (see §7.3).

### 5.6 Embedding Layer Gradient

**Forward:** $\mathbf{h}_t = E[\text{token}_t]$, where $E \in \mathbb{R}^{V \times d}$ is the embedding table.

**Backward:** Given upstream $\bar{\mathbf{h}}_t$ for all tokens $t = 1,\ldots,T$:

$$\frac{\partial \mathcal{L}}{\partial E[i]} = \sum_{t : \text{token}_t = i} \bar{\mathbf{h}}_t$$

This is a **sparse gradient** — only rows corresponding to tokens in the sequence receive nonzero updates.  For vocabulary size $V = 128{,}000$ (LLaMA-3), the embedding matrix is $128{,}000 \times 4{,}096$, but only a tiny fraction of rows are updated per batch.  Distributed training with embedding sharding exploits this sparsity.


---

## 6. Vanishing and Exploding Gradients

### 6.1 Magnitude Analysis — The Core Problem

Consider an $L$-layer network with no activation functions (to isolate the linear case).  The gradient of the loss with respect to layer $l$ parameters involves the product:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} = W^{[L]} W^{[L-1]} \cdots W^{[l+1]} \cdot \boldsymbol{\delta}^{[L]}$$

This is a product of $L - l$ matrices.  By the submultiplicativity of the spectral norm:

$$\left\| \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} \right\|_2 \leq \prod_{k=l+1}^{L} \|W^{[k]}\|_2 \cdot \|\boldsymbol{\delta}^{[L]}\|_2$$

If $\|W^{[k]}\|_2 = \rho < 1$ for all layers:
$$\text{gradient norm} \leq \rho^{L-l} \cdot \|\boldsymbol{\delta}^{[L]}\|_2 \xrightarrow{L \to \infty} 0 \quad \text{(vanishing)}$$

If $\|W^{[k]}\|_2 = \rho > 1$:
$$\text{gradient norm} \geq \rho^{L-l} \cdot \|\boldsymbol{\delta}^{[L]}\|_2 \xrightarrow{L \to \infty} \infty \quad \text{(exploding)}$$

```
GRADIENT MAGNITUDE ACROSS LAYERS
═════════════════════════════════════════════════════════════════════

  gradient norm
  │
  │ ↑ exploding (ρ > 1)
  │  ╲
  │   ╲
  │────╲──────────────────────────────── ideal (ρ = 1)
  │     ╲─────────────────────
  │      ╲─────────────────── vanishing (ρ < 1)
  └──────────────────────────────────────────── layer l
  L                                             0

  With activations, the product includes σ'(z) terms (< 1 for sigmoid)
  compounding the vanishing problem.

═════════════════════════════════════════════════════════════════════
```

This was identified by Hochreiter (1991) as the fundamental obstacle to training deep networks with gradient descent.

### 6.2 Activations and Saturation

For sigmoid $\sigma$: $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$ for all $z$, with equality only at $z=0$.  In the tails ($|z| \gg 0$), $\sigma'(z) \approx 0$.

For tanh: $\tanh'(z) = 1 - \tanh^2(z) \leq 1$, saturating similarly.

In a network with $L$ sigmoid layers and all activations near saturation, the gradient at layer 1 is suppressed by approximately $0.25^L$.  For $L = 20$: $0.25^{20} \approx 10^{-12}$ — numerically zero.

**ReLU** resolves saturation: $\text{relu}'(z) = \mathbf{1}[z > 0]$, which is either 0 or 1.  For active neurons, it passes gradients unchanged.  However, "dying ReLU" (neurons with $z < 0$ always) creates a different problem — those neurons receive zero gradient and never recover.

**GELU** and **SiLU** (used in LLaMA) are smooth approximations that avoid hard zeros, maintaining nonzero gradients everywhere.

### 6.3 Xavier and He Initialisation

**Goal:** Choose initial weights so that gradient (and activation) variance is preserved across layers — avoiding exponential growth or decay from the start of training.

**Xavier Initialisation** (Glorot & Bengio, 2010) — for symmetric activations (tanh, linear):

**Assumption:** Weights $W_{ij} \sim \mathcal{N}(0, \sigma^2)$ i.i.d., inputs $x_j$ with variance $\text{Var}(x_j) = v$.

**Forward variance preservation:** $\text{Var}(z_i) = n_\text{in} \sigma^2 v \Rightarrow \sigma^2 = 1/n_\text{in}$.

**Backward variance preservation:** $\text{Var}(\bar{x}_j) = n_\text{out} \sigma^2 \cdot \text{Var}(\bar{z}_i) \Rightarrow \sigma^2 = 1/n_\text{out}$.

Compromise:

$$\sigma^2 = \frac{2}{n_\text{in} + n_\text{out}}, \qquad \text{or uniform } W_{ij} \sim U\!\left[-\sqrt{\frac{6}{n_\text{in}+n_\text{out}}},\; \sqrt{\frac{6}{n_\text{in}+n_\text{out}}}\right]$$

**He Initialisation** (He et al., 2015) — for ReLU activations:

ReLU zeroes half the distribution, so effective variance is halved: $\text{Var}(\text{relu}(z)) = \tfrac{1}{2}\text{Var}(z)$.  To compensate:

$$\sigma^2 = \frac{2}{n_\text{in}}$$

**For AI:** GPT-2 uses a scaled version: weight initialisation $\mathcal{N}(0, 0.02)$ with the residual projection layers further scaled by $1/\sqrt{2L}$ where $L$ is the number of transformer layers, to control the variance accumulation in the residual stream.

### 6.4 Residual Connections as Gradient Highways

**Theorem:** In a residual network $F^{[l+1]}(\mathbf{x}) = \mathbf{x} + G^{[l]}(\mathbf{x})$, the gradient satisfies:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{[0]}} = \prod_{l=1}^{L}\left(I + J_{G^{[l]}}\right) \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{[L]}}$$

**Key insight:** Expanding the product, we get:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{[0]}} = \left(I + \sum_l J_{G^{[l]}} + \sum_{l < l'} J_{G^{[l']}} J_{G^{[l]}} + \cdots \right) \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{[L]}}$$

The identity term guarantees that even if all $J_{G^{[l]}} \approx 0$ (at initialisation), the gradient $\partial \mathcal{L}/\partial \mathbf{x}^{[0]}$ receives the full upstream signal unchanged.  This is the theoretical explanation for why ResNets (He et al., 2016) can be trained with hundreds of layers.

In modern transformers, the pre-norm architecture (LayerNorm *before* the sublayer, not after) further improves gradient flow by ensuring that the residual path carries a pure copy of the signal.

### 6.5 Gradient Clipping

**Gradient explosion** is addressed pragmatically by **global gradient norm clipping**:

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \min\!\left(1,\; \frac{\tau}{\|\mathbf{g}\|_2}\right)$$

where $\mathbf{g}$ is the concatenated parameter gradient vector and $\tau$ is the clip threshold.

**Typical values:** $\tau = 1.0$ for transformers (used in GPT-3, PaLM, LLaMA).

**Why global (not per-layer)?** Clipping each layer's gradient independently destroys the relative proportions of updates across layers, disrupting the Adam momentum states.  Global clipping preserves direction, only reducing magnitude.

**Relationship to RNNs:** Gradient clipping was originally introduced for RNNs (Mikolov, 2012; Pascanu et al., 2013), where the vanishing/exploding problem is especially severe due to the long chain of time steps.

### 6.6 Batch Normalisation and Layer Normalisation

**BatchNorm** (Ioffe & Szegedy, 2015) normalises each feature across the batch, stabilising the distribution of pre-activations.  Its gradient has a complex form involving batch statistics, but crucially it prevents activations from saturating on average.

**LayerNorm** (Ba et al., 2016) normalises each sample across features — preferred in transformers because:
1. Behaviour is independent of batch size (critical for small-batch inference)
2. Gradient analysis shows it damps large pre-activation magnitudes
3. Pre-norm placement ensures the residual stream grows in a controlled manner

**Empirical gradient norm tracking** is standard practice in LLM training: the gradient norm is logged at every step, and sudden spikes indicate loss spikes or numerical issues.  The Chinchilla and GPT-4 training runs used gradient norm monitoring as a primary signal for training health.


---

## 7. Memory-Efficient Backpropagation

### 7.1 Memory Cost of Standard Backprop

Standard backpropagation caches all intermediate activations for use in the backward pass.  For a transformer with $L$ layers, batch size $B$, sequence length $T$, and hidden dimension $d$:

| Component cached | Size | At FP16 |
|-----------------|------|---------|
| Attention QKV projections | $3 \times L \times B \times T \times d$ | $6LBTd$ bytes |
| Attention scores (pre-softmax) | $L \times B \times H \times T^2$ | $2LBH T^2$ bytes |
| MLP intermediate | $L \times B \times T \times 4d$ | $8LBTd$ bytes |
| LayerNorm stats | $2 \times 2L \times B \times T$ | negligible |

For GPT-3 ($L=96$, $B=512$, $T=2048$, $d=12288$, $H=96$): the attention scores alone require $96 \times 512 \times 96 \times 2048^2 \times 2 \approx 2.4 \text{ TB}$ — clearly infeasible without optimisation.

### 7.2 Gradient Checkpointing

**Idea:** Trade compute for memory.  Instead of caching all activations, cache only a subset of "checkpoint" activations and recompute the rest during the backward pass.

**Algorithm (checkpointing at every $k$-th layer):**

```
GRADIENT CHECKPOINTING
═══════════════════════════════════════════════════════════════════

  Forward pass:
    Compute all layers normally
    Save activations only at layers 0, k, 2k, 3k, ...
    Discard all other intermediate activations

  Backward pass:
    For each segment [lk, (l+1)k]:
      Re-run the forward pass from checkpoint lk
      Now have all intermediates for this segment
      Compute gradients for layers lk+1 to (l+1)k-1
      Discard intermediates (no longer needed)

═══════════════════════════════════════════════════════════════════
```

**Memory–compute tradeoff:**

- Memory: $O(\sqrt{L})$ checkpoints (optimal with $k = \sqrt{L}$) instead of $O(L)$
- Compute: Each layer's forward pass is run twice (once in original forward, once in recomputation) → approximately $+33\%$ compute overhead

**For AI:** `torch.utils.checkpoint.checkpoint()` implements this in PyTorch with a single function call.  LLaMA, Mistral, and most OSS LLM trainers enable activation checkpointing by default for sequences longer than ~2048 tokens.

**Selective recomputation:** Flash Attention (see §7.3) takes a more targeted approach — instead of checkpointing by layer, it recomputes only the attention scores (the $T^2$ term) during the backward pass, since those are the dominant memory consumer.

### 7.3 FlashAttention: Fused Backward Pass

**The $O(T^2)$ problem:** Standard attention stores $P = \text{softmax}(QK^\top/\sqrt{d}) \in \mathbb{R}^{T \times T}$ for the backward pass.  For $T = 32{,}768$ (long-context models), this is $32768^2 \times 2 \approx 2$ GB *per layer per batch element*.

**FlashAttention solution** (Dao et al., 2022): Compute attention in tiles that fit in SRAM (GPU on-chip cache), using the online softmax algorithm (Milakov & Gimelshein, 2018) to avoid materialising the full $T \times T$ matrix.

**Backward pass in FlashAttention:** The backward pass needs $P$ but doesn't store it.  Instead:

1. Store only the softmax normalisation statistics $m_i, \ell_i$ (scalars per row) — $O(T)$ memory
2. During the backward pass, recompute $P$ tile by tile from $Q, K$ and the stored statistics
3. Accumulate gradients $\bar{Q}, \bar{K}, \bar{V}$ tile by tile without ever forming full $P$

**Complexity:**
- Memory: $O(T)$ instead of $O(T^2)$
- FLOPs: $4 \times$ the forward FLOPs (small constant factor)
- Wall-clock speedup: 2-4× over standard PyTorch attention on A100

**For AI:** FlashAttention is the default attention implementation in modern LLM training (vLLM, HuggingFace Transformers, NanoGPT). FlashAttention-3 (2024) further optimises for H100 tensor core and async operations.

### 7.4 Mixed Precision Training

**Observation:** FP32 (32-bit float) is unnecessarily precise for gradients.  FP16 (16-bit float) has $\sim 3\times$ higher memory bandwidth on modern GPUs, but overflow/underflow is common for small/large gradient values.

**AMP (Automatic Mixed Precision) strategy:**

| Component | Precision | Reason |
|-----------|-----------|--------|
| Forward activations | FP16 | Fast compute, lower memory |
| Backward gradients | FP16 | Fast compute |
| Weight updates | FP32 | Avoid precision loss |
| Master weights | FP32 | Preserve small updates ($\Delta w \ll w$) |
| Loss scaling | Dynamic | Prevent FP16 underflow for small gradients |

**Loss scaling:** Multiply the loss by a large scale factor $S$ (typically $2^{12}$ to $2^{16}$) before backward, then divide gradients by $S$ before the weight update.  This shifts gradient values into the representable FP16 range.  The scale factor is increased or decreased based on whether overflow (inf/nan) occurred.

**BF16** (Brain Float 16, used in TPUs and H100): same 16-bit width but with 8 exponent bits (same as FP32) and only 7 mantissa bits.  Eliminates overflow issues while retaining dynamic range — now the preferred format for LLM training.


---

## 8. Advanced Differentiation Topics

### 8.1 Backpropagation Through Time (BPTT)

A recurrent neural network (RNN) with hidden state $\mathbf{h}_t = \sigma(W_h \mathbf{h}_{t-1} + W_x \mathbf{x}_t + \mathbf{b})$ can be viewed as a feedforward network *unrolled* through time:

```
UNROLLED RNN — BPTT VIEW
═══════════════════════════════════════════════════════════════════

  x₁ → [cell] → h₁ → [cell] → h₂ → [cell] → h₃ → ... → hₜ → loss
          ↑               ↑               ↑
          Wₕ             Wₕ             Wₕ         (shared weights)

  BPTT = backprop through the unrolled graph.
  Gradient of loss w.r.t. Wₕ = sum of gradients from all time steps.

═══════════════════════════════════════════════════════════════════
```

The gradient with respect to $W_h$ at time step $t$ involves the product:

$$\frac{\partial \mathcal{L}}{\partial W_h} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial \mathbf{h}_t} \left( \prod_{k=1}^{t-1} \frac{\partial \mathbf{h}_{t-k+1}}{\partial \mathbf{h}_{t-k}} \right) \frac{\partial \mathbf{h}_1}{\partial W_h}$$

Each factor $\partial \mathbf{h}_{t+1}/\partial \mathbf{h}_t = W_h \cdot \text{diag}(\sigma'(\mathbf{z}_t))$.  When $\|W_h\|_2 \cdot \|\sigma'\|_\infty < 1$, the product of $T$ such factors vanishes exponentially.  This is the core failure mode of vanilla RNNs on long sequences (Hochreiter, 1991; Bengio et al., 1994).

**Truncated BPTT:** In practice, gradients are truncated to a window of $k$ steps to reduce memory and compute costs, at the cost of ignoring long-range dependencies beyond step $k$.

**LSTM/GRU solution:** Long Short-Term Memory networks (Hochreiter & Schmidhuber, 1997) use gating mechanisms to maintain a cell state $\mathbf{c}_t$ with additive updates — replacing multiplicative products of weight matrices with additive accumulation, similar to residual connections.

### 8.2 Implicit Differentiation Preview

For optimisation problems or fixed-point iterations, we sometimes need gradients of implicit functions.

**Example:** Consider $\mathbf{y}^* = \arg\min_\mathbf{y} \mathcal{L}(\mathbf{y}, \theta)$ where the optimum satisfies $\nabla_\mathbf{y} \mathcal{L}(\mathbf{y}^*, \theta) = 0$.

By the implicit function theorem:

$$\frac{d\mathbf{y}^*}{d\theta} = -\left[\nabla^2_{\mathbf{y}\mathbf{y}} \mathcal{L}\right]^{-1} \nabla^2_{\mathbf{y}\theta} \mathcal{L}$$

This allows differentiating through optimisation steps without unrolling them — the basis of **MAML** (Model-Agnostic Meta-Learning, Finn et al., 2017) and **DEQs** (Deep Equilibrium Models, Bai et al., 2019).

> **Full treatment:** Implicit differentiation and differentiable optimisation are covered in depth in [§05/05-Automatic-Differentiation](../05-Automatic-Differentiation/notes.md).

### 8.3 Straight-Through Estimator and REINFORCE

**The discrete problem:** When a node in the computation graph applies a discrete operation (argmax, sampling, rounding), the gradient is zero almost everywhere.  The chain rule breaks — the graph is not differentiable at these nodes.

**Straight-Through Estimator (STE)** (Hinton, 2012; Bengio et al., 2013):

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} := \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{x}}} \cdot \mathbf{1} \qquad \text{(treat discretisation as identity in backward)}$$

**Applications:**
- **Quantisation-aware training (QAT):** Simulate INT8 forward, use STE backward.  Used in GPTQ, AWQ, and quantised LLM training.
- **VQ-VAE:** Vector quantisation in the encoder uses STE so gradients flow from decoder back to encoder.
- **Binary neural networks:** Forward uses sign(x), backward uses STE with gradient identity.

**REINFORCE (Williams, 1992):** For stochastic nodes, use the log-derivative trick:

$$\nabla_\theta \mathbb{E}_{z \sim p_\theta}[\mathcal{L}(z)] = \mathbb{E}_{z \sim p_\theta}[\mathcal{L}(z) \nabla_\theta \log p_\theta(z)]$$

This produces an unbiased gradient estimate but with high variance (addressed by baseline subtraction: $(\mathcal{L}(z) - b)\nabla_\theta \log p_\theta(z)$).  REINFORCE is the foundation of policy gradient methods in RL and is used in RLHF's PPO step.

### 8.4 Higher-Order Gradients

**Second-order gradients** arise in:
1. Newton's method: requires Hessian $H = \nabla^2 \mathcal{L}$ (see §02-Jacobians-and-Hessians)
2. Meta-learning (MAML): gradient of gradient w.r.t. outer parameters
3. Gradient penalty in GAN training: $\|\nabla_x D(x)\|^2$

**In PyTorch:** Higher-order gradients are computed by running autograd through itself:

```python
# Second derivative of loss w.r.t. input
loss = model(x).sum()
g = torch.autograd.grad(loss, x, create_graph=True)[0]
g2 = torch.autograd.grad(g.sum(), x)[0]  # second derivative
```

`create_graph=True` tells autograd to build a graph for the gradient computation itself, enabling differentiation through it.

**Hessian-vector products (HVPs):** As shown in §02, the HVP $Hv$ can be computed in $O(n)$ time without forming $H$:
$$Hv = \nabla_\mathbf{x}[(\nabla_\mathbf{x} \mathcal{L})^\top v]$$

This is the primitive operation behind conjugate gradient and Lanczos methods for curvature estimation.


---

## 9. Transformer Backpropagation

### 9.1 Full Transformer Layer Gradient Flow

A pre-norm transformer layer processes the residual stream $\mathbf{x} \in \mathbb{R}^d$ as:

$$\mathbf{x}' = \mathbf{x} + \text{Attn}(\text{LN}_1(\mathbf{x}))$$
$$\mathbf{x}'' = \mathbf{x}' + \text{MLP}(\text{LN}_2(\mathbf{x}'))$$

**Backward through one transformer layer** (given $\bar{\mathbf{x}}''\,$):

```
GRADIENT FLOW — ONE TRANSFORMER LAYER
═══════════════════════════════════════════════════════════════════

  FORWARD                              BACKWARD

  x ──────────────────────────────┐    ↑ x̄'' flows in
        ↓                         │    │
      LN₁(x)                      │    ↓ x̄' = x̄'' + MLP_backward
        ↓                         │                 (x̄'')
      Attn(·)                     │    ↓ x̄ = x̄' + Attn_backward
        ↓                         │                 (x̄')
  x' = x + Attn(LN₁(x)) ─────────┤
                                   │    The two residual additions
        ↓                         │    split the gradient stream
      LN₂(x')                     │    into parallel paths —
        ↓                         │    the identity skip carries
      MLP(·)                      │    the full upstream signal
        ↓                         │    unchanged.
  x'' = x' + MLP(LN₂(x')) ───────┘

═══════════════════════════════════════════════════════════════════
```

The critical observation: both residual additions in the transformer layer act as gradient splitters.  The skip path carries a copy of $\bar{\mathbf{x}}''$ directly back to $\bar{\mathbf{x}}'$ without passing through the MLP Jacobian.  This gives transformers well-behaved gradients even at $L = 96$ layers (GPT-3) or $L = 126$ layers (Grok-1).

### 9.2 LoRA Backward Pass

**Low-Rank Adaptation** (Hu et al., 2022) reparametrises a weight matrix:

$$W = W_0 + BA, \quad W_0 \in \mathbb{R}^{m \times n} \text{ (frozen)}, \quad B \in \mathbb{R}^{m \times r}, A \in \mathbb{R}^{r \times n}$$

**Forward:** $\mathbf{y} = W\mathbf{x} = W_0\mathbf{x} + BA\mathbf{x}$.

**Backward** (given $\bar{\mathbf{y}}$):

$$\bar{A} = B^\top \bar{\mathbf{y}} \mathbf{x}^\top \in \mathbb{R}^{r \times n}$$
$$\bar{B} = \bar{\mathbf{y}} \mathbf{x}^\top A^\top \in \mathbb{R}^{m \times r}$$
$$\bar{\mathbf{x}}_\text{from LoRA} = A^\top B^\top \bar{\mathbf{y}} = (BA)^\top \bar{\mathbf{y}}$$

Note: $W_0$ is frozen, so $\bar{W}_0 = 0$ — no gradient is computed or stored for $W_0$.  The backward pass only updates $A$ and $B$.

**Memory savings:** For $W_0 \in \mathbb{R}^{4096 \times 4096}$ with $r = 16$: gradient storage reduces from $4096^2 = 16.7\text{M}$ to $r(m + n) = 16 \times 8192 = 131\text{K}$ parameters — a $128\times$ reduction in gradient memory for that layer.

**DoRA** (Liu et al., 2024) further decomposes LoRA into magnitude + direction components, improving fine-tuning quality while preserving the low-rank backward structure.

### 9.3 Gradient Accumulation

**Problem:** Large effective batch sizes ($B = 4\text{M}$ tokens, as in GPT-4 training) don't fit in GPU memory for a single forward-backward pass.

**Solution — gradient accumulation:**

```
For each micro-batch b = 1, ..., G:
    loss_b = forward(micro_batch_b) / G   # scaled loss
    backward(loss_b)                       # accumulates gradients
    # gradients are NOT zeroed between micro-batches

optimizer.step()   # update once after G micro-batches
optimizer.zero_grad()
```

The division by $G$ ensures the accumulated gradient is mathematically identical to what a single pass with the full batch would produce.

**For AI:** GPT-3 used gradient accumulation to achieve an effective batch of $\sim 3\text{M}$ tokens with hardware that could only process $\sim 500\text{K}$ tokens per step.

### 9.4 Distributed Gradient Synchronisation

In **data parallelism**, each GPU processes a different micro-batch but shares the same model weights.  After the backward pass, gradients must be synchronised:

**All-Reduce:** Sum gradients across all $N$ GPUs and divide by $N$.  Implemented via ring-all-reduce (NCCL) for $O(N \cdot |\theta|)$ communication that is bandwidth-optimal.

**Gradient sharding (ZeRO):** DeepSpeed's ZeRO (Zero Redundancy Optimizer) partitions gradient storage across GPUs:
- ZeRO Stage 1: Shard optimiser states → $4\times$ memory reduction
- ZeRO Stage 2: Shard gradients additionally → $8\times$ reduction
- ZeRO Stage 3: Shard parameters too → $N\times$ reduction (linear in GPU count)

**For LLaMA-3 70B training:** ZeRO Stage 3 across 1024 H100 GPUs allows storing only $\sim 70\text{B}/1024 \approx 68\text{M}$ parameters per GPU — fitting the model in memory.


---

## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|----------------|-----|
| 1 | Applying scalar chain rule $\frac{dy}{dx} = \frac{dy}{du}\frac{du}{dx}$ for vector functions | Scalar chain rule multiplies; multivariate chain rule composes Jacobians. Order matters: $J_{f\circ g} = J_f \cdot J_g$, not $J_g \cdot J_f$ | Write Jacobians explicitly and multiply left-to-right in the order of composition |
| 2 | Forgetting to sum gradients at fan-out (shared weight) nodes | Each use of a weight contributes a gradient; missing uses means undercounting | Accumulate gradients with `+=` in the backward loop over all uses |
| 3 | Treating $\nabla_W \mathcal{L} = \boldsymbol{\delta} \mathbf{x}^\top$ as shape-correct without checking | The outer product $\boldsymbol{\delta} \mathbf{x}^\top$ has shape $(n_\text{out}, n_\text{in})$ matching $W$; but transposing either vector gives wrong shape | Always verify gradient shapes match parameter shapes before implementation |
| 4 | Using sigmoid/tanh in deep networks expecting no vanishing gradients | Their derivatives are bounded by $0.25$ / $1.0$ — products over many layers vanish exponentially | Use ReLU, GELU, or SiLU with proper initialisation; add residual connections |
| 5 | Initialising all weights to zero (or same value) | Symmetry breaking fails: every neuron in a layer computes the same gradient, so they all update identically and remain identical forever | Use Xavier or He initialisation with random values |
| 6 | Skipping the fused softmax + cross-entropy optimisation and computing them separately | Intermediate probabilities $p_i = e^{z_i}/\sum e^{z_j}$ overflow/underflow for large logits | Always use the log-sum-exp trick or a library's CrossEntropyLoss (which applies it internally) |
| 7 | Confusing JVP and VJP — using JVP for all gradient computations | JVP costs $O(n)$ passes for scalar output; VJP costs $O(1)$ per output dimension. For scalar loss, always use VJP (backprop) | Use VJP (backward) for scalar losses; reserve JVP for computing Jacobian rows or directional derivatives |
| 8 | Clipping per-layer gradients independently instead of global norm | Destroys the relative scale of gradients across layers; disrupts Adam's per-parameter adaptive scaling | Clip the global gradient norm: compute $\|\mathbf{g}\|$ across all parameters, scale down if above threshold |
| 9 | Using STE incorrectly in quantisation-aware training — applying STE to continuous weights | STE should only be applied at the discrete rounding step, not to subsequent continuous operations | Apply STE only at the `round()` or `sign()` node; propagate real gradients elsewhere |
| 10 | Misunderstanding gradient accumulation — forgetting to scale the loss | Accumulating $G$ micro-batch gradients without dividing by $G$ produces $G\times$ too large an effective gradient | Divide loss by $G$ before backward, or divide accumulated gradients by $G$ before the optimiser step |
| 11 | Not using `create_graph=True` when computing higher-order gradients in PyTorch | Without `create_graph=True`, the gradient computation is not tracked, so differentiating through it returns `None` or wrong values | Use `create_graph=True` in the first `torch.autograd.grad()` call when second derivatives are needed |
| 12 | Confusing BPTT truncation with sequence truncation | Truncated BPTT still runs the full forward sequence; it only truncates the *backward* window. Sequence truncation shortens both | These are different operations — read the framework docs to confirm which is applied |


---

## 11. Exercises

### Exercise 1 ★ — Scalar Chain Rule Verification

Let $f(t) = \sin(t^2)$ and $g(t) = e^{3t}$.

**(a)** Compute $\frac{d}{dt}[f(g(t))]$ using the chain rule analytically.

**(b)** Evaluate the derivative at $t = 0$ and $t = 1$.

**(c)** Verify numerically using centred finite differences.

**(d)** Compute $\frac{d}{dt}[g(f(t))]$ — explain why the order of composition matters.

---

### Exercise 2 ★ — Jacobian Composition

Let $f: \mathbb{R}^3 \to \mathbb{R}^2$ and $g: \mathbb{R}^2 \to \mathbb{R}^3$ be defined by:
$$f(\mathbf{u}) = (u_1 u_2,\ u_2 + u_3^2), \quad g(\mathbf{x}) = (x_1^2,\ x_1 x_2,\ e^{x_2})$$

**(a)** Compute $J_f(\mathbf{u})$ and $J_g(\mathbf{x})$ analytically.

**(b)** Compute the Jacobian of $h = f \circ g: \mathbb{R}^2 \to \mathbb{R}^2$ using the chain rule $J_h = J_f(g(\mathbf{x})) \cdot J_g(\mathbf{x})$.

**(c)** Verify using finite differences at $\mathbf{x}_0 = (1, 0)^\top$.

**(d)** Compute $J_h$ directly and confirm it equals part (b).

---

### Exercise 3 ★ — Backprop Through a 2-Layer Network

Two-layer network: $\mathbf{z}^{[1]} = W^{[1]}\mathbf{x} + \mathbf{b}^{[1]}$, $\mathbf{a}^{[1]} = \text{relu}(\mathbf{z}^{[1]})$, $\hat{y} = \mathbf{w}^{[2]} \cdot \mathbf{a}^{[1]} + b^{[2]}$, $\mathcal{L} = \tfrac{1}{2}(\hat{y} - y)^2$.

With $W^{[1]} \in \mathbb{R}^{3 \times 2}$, $\mathbf{x} \in \mathbb{R}^2$, $\mathbf{w}^{[2]} \in \mathbb{R}^3$:

**(a)** Implement forward pass. Compute $\mathcal{L}$ for given values.

**(b)** Implement backward pass manually using the backpropagation recurrence.

**(c)** Verify your gradients using `numpy` finite differences.

**(d)** Implement gradient descent for 100 steps with learning rate $0.01$ and verify loss decreases.

---

### Exercise 4 ★★ — Vanishing Gradients Analysis

**(a)** Construct a 20-layer sigmoid network with all weights $= 0.3$.  Compute the gradient at layer 1 symbolically and numerically.

**(b)** Repeat with ReLU activation.  Compare gradient magnitudes at layers 1, 5, 10, 20.

**(c)** Apply Xavier initialisation to the sigmoid network and compare gradient flow.

**(d)** Add residual connections to the 20-layer sigmoid network.  Quantify the improvement.

**(e)** Plot gradient norm vs. layer depth for all four cases.

---

### Exercise 5 ★★ — Gradient Checkpointing

**(a)** Implement a 10-layer feedforward network with explicit intermediate caching.  Measure peak memory usage.

**(b)** Implement the same network with gradient checkpointing at every 3rd layer.  Measure memory.

**(c)** Verify that both implementations produce identical gradients.

**(d)** Measure the compute overhead of recomputation.  How does it compare to the theoretical $+33\%$?

**(e)** Find the optimal checkpoint interval $k$ that minimises total memory × compute cost.

---

### Exercise 6 ★★ — Attention Gradient

Single-head attention: $O = \text{softmax}(QK^\top/\sqrt{d})V$ with $Q, K, V \in \mathbb{R}^{T \times d}$ for $T = 4$, $d = 3$.

**(a)** Implement forward pass.

**(b)** Implement backward pass computing $\bar{Q}, \bar{K}, \bar{V}$ given $\bar{O}$.

**(c)** Verify all three gradients using finite differences.

**(d)** For causal masking (set $S_{ij} = -\infty$ for $j > i$), show that the backward pass is unchanged except at masked positions.

---

### Exercise 7 ★★★ — LoRA Gradient Analysis

**(a)** Implement a linear layer $\mathbf{y} = (W_0 + BA)\mathbf{x}$ with LoRA adaptation.  Set $m=8, n=6, r=2$.

**(b)** Compute gradients $\nabla_A \mathcal{L}$ and $\nabla_B \mathcal{L}$ analytically and verify numerically.

**(c)** Confirm that $\nabla_{W_0} \mathcal{L} = \bar{\mathbf{y}} \mathbf{x}^\top$ but is not used (frozen).

**(d)** Compare the number of gradient parameters for full fine-tuning vs. LoRA.

**(e)** Implement LoRA training for 200 steps on a toy task and compare convergence with full fine-tuning.

---

### Exercise 8 ★★★ — REINFORCE and STE

**(a)** Implement a stochastic computational graph: $z \sim \text{Bernoulli}(\sigma(\theta))$, $\mathcal{L} = z^2$.

**(b)** Compute the REINFORCE gradient $\nabla_\theta \mathbb{E}[\mathcal{L}]$ analytically.

**(c)** Estimate the REINFORCE gradient with 10000 samples.  Verify against the analytical value.

**(d)** Implement STE for the rounding operation: $\hat{w} = \text{round}(w)$, $\mathcal{L} = (\hat{w} - w_\text{target})^2$.  Compute the STE gradient and update $w$.

**(e)** Compare STE-based quantisation-aware training on a toy example: train for 50 steps and measure quantisation error vs. a post-training quantised model.


---

## 12. Why This Matters for AI (2026 Perspective)

| Concept | Concrete AI Impact |
|---------|-------------------|
| **Multivariate chain rule** | The mathematical foundation of every gradient-based learning algorithm — without it, backprop cannot be defined |
| **VJP as backprop primitive** | Modern autodiff systems (JAX, PyTorch) are built around VJP primitives; the $O(1)$ cost of reverse mode is what makes training billion-parameter models tractable |
| **Computation graphs** | `torch.compile` (PyTorch 2.0), XLA (JAX/TensorFlow), TensorRT all operate by analysing the computation graph to fuse kernels and optimise memory layout |
| **Fused softmax + CE gradient** | The clean gradient $\mathbf{p} - \mathbf{e}_y$ makes language model training numerically stable; Flash Attention's backward uses the same softmax log-sum-exp statistics |
| **Xavier/He initialisation** | Ensures $O(1)$ gradient scale at depth 1 or depth 96 — a critical practical enabler for deep network training |
| **Residual connections** | The "gradient highway" identity term in ResNets/transformers is why 100-layer networks train at all; this was the key insight enabling GPT-3's 96 layers |
| **Gradient checkpointing** | Enables training LLMs with 128K context lengths; without it, the $O(T^2)$ activation memory would be prohibitive |
| **FlashAttention backward** | IO-aware backward pass reduces memory from $O(T^2)$ to $O(T)$ while maintaining numerical equivalence; standard in all production LLM training as of 2024 |
| **LoRA backward** | Only $r(m+n) \ll mn$ parameters accumulate gradients; enables fine-tuning 70B models on a single H100 via the low-rank backward structure |
| **STE / REINFORCE** | STE enables quantisation-aware training (GPTQ, AWQ, QLoRA); REINFORCE enables RLHF's policy gradient step in PPO-based alignment training |
| **BPTT** | The failure of vanilla BPTT for long sequences motivated LSTMs, GRUs, and ultimately the attention mechanism which replaces recurrence with direct pairwise interactions |
| **ZeRO gradient sharding** | Partitions gradient storage across GPUs linearly in GPU count; enables training models that would require $8\times$ more memory per GPU without it |
| **Mixed precision backward** | BF16 backward passes achieve $2-3\times$ memory bandwidth vs FP32 on H100, with dynamic loss scaling preventing underflow; standard in all LLM training since GPT-3 |
| **Higher-order gradients** | Gradient penalties in GANs, MAML's meta-gradient, and Hessian-vector products for learning rate scheduling all require differentiating through the backward pass |

---

## Conceptual Bridge

**Where we came from:** §01 (Partial Derivatives) gave us tools to differentiate multivariate functions component by component.  §02 (Jacobians and Hessians) assembled those into matrix objects capturing full first- and second-order sensitivity.  We now know what a derivative *is* for a function $f: \mathbb{R}^n \to \mathbb{R}^m$.

**What this section added:** The chain rule tells us how derivatives *compose* — allowing us to differentiate functions built from primitives.  Backpropagation is the algorithmic instantiation of this composition for computation graphs, and the VJP (reverse mode) makes the cost of differentiating a scalar loss with respect to millions of parameters equal to the cost of a single forward pass.  This is not an approximation — it is exact and provably optimal.

**What this enables:** Every gradient-based learning algorithm — SGD, Adam, RMSprop, LARS, Shampoo — requires only the gradient $\nabla_\theta \mathcal{L}$, which backprop provides.  The advanced sections of this chapter (§04 Optimisation, §05 Automatic Differentiation) build directly on the VJP abstraction established here.

**Connection to transformer training:** Modern LLM training is essentially an exercise in efficient backpropagation at scale.  Every engineering decision — Flash Attention's tiled backward, ZeRO's gradient sharding, gradient checkpointing, LoRA's low-rank backward, mixed precision loss scaling — is a response to the memory and compute constraints of the backward pass.  Understanding backpropagation is therefore prerequisite to understanding why LLM training systems are designed the way they are.

```
POSITION IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  PREREQUISITES (must know):
    §01 Partial Derivatives — ∂f/∂xᵢ, gradient, directional derivative
    §02 Jacobians & Hessians — J_f, Fréchet derivative, VJP/JVP

  THIS SECTION (§03):
  ┌─────────────────────────────────────────────────────────────────┐
  │  Chain Rule & Backpropagation                                   │
  │  • Multivariate chain rule (J_{f∘g} = J_f · J_g)               │
  │  • Computation graphs (DAG, topological order)                  │
  │  • Backprop recurrence (δ = Wᵀδ⁺ ⊙ σ'(z))                     │
  │  • Gradient derivations (linear, softmax+CE, LN, attention)     │
  │  • Vanishing/exploding gradients + solutions                    │
  │  • Memory-efficient backprop (checkpointing, Flash Attention)   │
  │  • Advanced: BPTT, STE, REINFORCE, higher-order gradients       │
  └─────────────────────────────────────────────────────────────────┘

  WHAT THIS ENABLES:
    §04 Optimisation — gradient descent, Adam, second-order methods
    §05 Automatic Differentiation — AD systems, tape, jit compilation
    §07 Neural Networks — full training loop built on backprop
    §08 Transformer Architecture — FlashAttention, LoRA, gradient flow

  CROSS-CHAPTER CONNECTIONS:
    §03-Advanced-LA/02-SVD — gradient low-rank structure
    §04-Calculus/02-Derivatives — scalar chain rule (special case)
    §06-Probability/03-MLE — loss functions that backprop optimises

════════════════════════════════════════════════════════════════════════
```

---

_For automatic differentiation systems that implement these ideas at scale, see [§05 Automatic Differentiation](../05-Automatic-Differentiation/notes.md)._

_For the optimisation algorithms that consume backprop's output, see [§04 Multivariate Optimisation](../04-Multivariate-Optimisation/notes.md)._


---

## Appendix A: Worked Backpropagation Example

### A.1 Complete Worked Example — 3-Layer Network

To make the backpropagation formulas concrete, we trace through a minimal example end-to-end.

**Network architecture:**
- Input: $\mathbf{x} = (1, 2)^\top$
- Layer 1: $W^{[1]} = \begin{pmatrix}0.5 & 0.1 \\ 0.2 & 0.3\end{pmatrix}$, $\mathbf{b}^{[1]} = (0, 0)^\top$, activation: ReLU
- Layer 2: $\mathbf{w}^{[2]} = (0.4, 0.6)^\top$, $b^{[2]} = 0$, activation: none (scalar output)
- Loss: $\mathcal{L} = \tfrac{1}{2}(\hat{y} - 1)^2$ (MSE with target $y = 1$)

**Forward pass:**

$$\mathbf{z}^{[1]} = W^{[1]}\mathbf{x} = \begin{pmatrix}0.5 \cdot 1 + 0.1 \cdot 2 \\ 0.2 \cdot 1 + 0.3 \cdot 2\end{pmatrix} = \begin{pmatrix}0.7 \\ 0.8\end{pmatrix}$$

$$\mathbf{a}^{[1]} = \text{relu}(\mathbf{z}^{[1]}) = \begin{pmatrix}0.7 \\ 0.8\end{pmatrix} \quad \text{(both positive, no change)}$$

$$\hat{y} = \mathbf{w}^{[2]} \cdot \mathbf{a}^{[1]} = 0.4 \times 0.7 + 0.6 \times 0.8 = 0.28 + 0.48 = 0.76$$

$$\mathcal{L} = \tfrac{1}{2}(0.76 - 1)^2 = \tfrac{1}{2}(0.0576) = 0.0288$$

**Backward pass:**

*Output layer gradient:*
$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \hat{y} - y = 0.76 - 1 = -0.24$$

*Layer 2 gradients (scalar output, linear):*
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}^{[2]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \mathbf{a}^{[1]} = -0.24 \times \begin{pmatrix}0.7 \\ 0.8\end{pmatrix} = \begin{pmatrix}-0.168 \\ -0.192\end{pmatrix}$$

$$\frac{\partial \mathcal{L}}{\partial b^{[2]}} = -0.24$$

*Error signal propagated to layer 1:*
$$\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[1]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \mathbf{w}^{[2]} = -0.24 \times \begin{pmatrix}0.4 \\ 0.6\end{pmatrix} = \begin{pmatrix}-0.096 \\ -0.144\end{pmatrix}$$

*Through ReLU:*
$$\boldsymbol{\delta}^{[1]} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[1]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[1]}} \odot \text{relu}'(\mathbf{z}^{[1]}) = \begin{pmatrix}-0.096 \\ -0.144\end{pmatrix} \odot \begin{pmatrix}1 \\ 1\end{pmatrix} = \begin{pmatrix}-0.096 \\ -0.144\end{pmatrix}$$

*Layer 1 weight gradients:*
$$\frac{\partial \mathcal{L}}{\partial W^{[1]}} = \boldsymbol{\delta}^{[1]} \mathbf{x}^\top = \begin{pmatrix}-0.096 \\ -0.144\end{pmatrix}(1, 2) = \begin{pmatrix}-0.096 & -0.192 \\ -0.144 & -0.288\end{pmatrix}$$

*Verification (finite difference for $W^{[1]}_{11}$):* Perturb $W^{[1]}_{11}$ by $h = 10^{-5}$:
$$\mathcal{L}(W^{[1]}_{11} + h) - \mathcal{L}(W^{[1]}_{11} - h) \approx 2h \cdot (-0.096)$$

Numerically: $\mathcal{L}(0.50001) \approx 0.02880960, \mathcal{L}(0.49999) \approx 0.02881152$. Difference $/2h = -0.096$. ✓

### A.2 Computational Cost Comparison

**Forward pass:** $O(n_0 n_1 + n_1 n_2 + \cdots + n_{L-1} n_L)$ — one GEMM per layer.

**Backward pass:** Also $O(\sum_l n_{l-1} n_l)$ — same asymptotic cost, with constant factor $\approx 2-3$.

**Memory:** Cache all $\mathbf{z}^{[l]}$ and $\mathbf{a}^{[l]}$: $O(\sum_l n_l)$ scalars — linear in total neuron count.

**The fundamental theorem of backpropagation:** Computing $\nabla_\theta \mathcal{L}$ for *all* $|\theta|$ parameters costs only a constant factor more than computing $\mathcal{L}$ itself.  This is the miracle that makes gradient-based learning tractable.

**Formal statement:** Let $T_f$ be the time to evaluate $\mathcal{L}$ in the forward pass.  Then the time to compute all partial derivatives $\partial \mathcal{L}/\partial \theta_i$ via backprop is at most $c \cdot T_f$ where $c \leq 5$ in practice.

This contrasts with finite differences: computing $\partial \mathcal{L}/\partial \theta_i$ for each of $|\theta|$ parameters via finite differences costs $2|\theta|$ forward passes — for GPT-3 with $|\theta| = 175\text{B}$, this would be $350$ billion forward passes, or approximately the heat death of the universe in compute time.


---

## Appendix B: JVP vs VJP — Mode Selection and Complexity

### B.1 Forward Mode vs Reverse Mode

Given $f: \mathbb{R}^n \to \mathbb{R}^m$, both modes compute the same gradient information but with different costs:

| Mode | Computes | Cost per pass | Total cost for full Jacobian |
|------|---------|---------------|------------------------------|
| Forward (JVP) | One column of $J_f$ | $O(T_f)$ | $O(n \cdot T_f)$ |
| Reverse (VJP) | One row of $J_f$ | $O(T_f)$ | $O(m \cdot T_f)$ |

```
COST MATRIX: WHICH MODE WINS?
═══════════════════════════════════════════════════════════════════

  Goal: compute ∂ℒ/∂θ for ℒ: ℝⁿ → ℝ (scalar loss)

  n = |θ| = 175,000,000,000  (GPT-3 parameter count)
  m = 1                       (scalar loss)

  Forward mode: m × Tf = 1 × Tf  ← ONE PASS
  Reverse mode: n × Tf = 175B × Tf  ← 175 BILLION PASSES

  Wait — that's backwards! Reverse mode (backprop) costs 1 pass
  because m=1 means ONE ROW of J_f = the gradient row vector.
  Forward mode would need n=175B passes to fill all columns.

  ┌─────────────────────────────────────────────────────────────┐
  │  RULE: Use reverse mode (backprop) when m ≪ n               │
  │  RULE: Use forward mode (JVP) when n ≪ m                    │
  └─────────────────────────────────────────────────────────────┘

  Most ML: n ≫ m = 1 → backprop is optimal

═══════════════════════════════════════════════════════════════════
```

**When forward mode wins:** Computing the sensitivity of all $m$ outputs to one input parameter — e.g., computing how the entire model output changes as a single hyperparameter varies.  Also: Jacobian-vector products in conjugate gradient (no need for the full Jacobian).

**Mixed strategies:** For functions $f: \mathbb{R}^n \to \mathbb{R}^m$ with $n \approx m$, the optimal choice is to split the Jacobian into row/column blocks and use each mode for the appropriate blocks — the basis of **adjoint methods** in numerical PDE solvers.

### B.2 Tangent Mode for Hessian-Vector Products

As shown in §02, the Hessian-vector product $H\mathbf{v}$ can be computed by composing forward and reverse modes:

$$H\mathbf{v} = J_\mathbf{g}^\top \mathbf{v} \quad \text{where} \quad \mathbf{g} = \nabla_\theta \mathcal{L}$$

**Algorithm (Pearlmutter's R{} trick, 1994):**
1. Forward pass (JVP with direction $\mathbf{v}$): compute $\mathbf{g} = \nabla \mathcal{L}$ and $\dot{\mathbf{g}} = J_\mathbf{g} \mathbf{v}$ simultaneously
2. Cost: same as backprop ($O(T_f)$) — one pass suffices

**Implementation in PyTorch:**
```python
g = torch.autograd.grad(loss, params, create_graph=True)
flat_g = torch.cat([gi.view(-1) for gi in g])
hvp = torch.autograd.grad(flat_g @ v, params)
```

Cost: 2 backprop passes, no $n \times n$ matrix formed.  This is the primitive for:
- **Conjugate gradient** for Newton steps (K-FAC-style)
- **Lanczos iteration** for $\lambda_\text{max}$ of Hessian
- **Eigenvalue monitoring** during training (Cohen et al., 2022 — edge of stability)

---

## Appendix C: Automatic Differentiation Preview

### C.1 The AD Abstraction

**Automatic differentiation** (AD) is a mechanical procedure for transforming any program that computes $f(\mathbf{x})$ into a program that also computes $\nabla f(\mathbf{x})$ (or JVPs/VJPs).  This section previews the idea; the full treatment is in §05.

AD is neither symbolic differentiation (too slow, exponentially large expressions) nor numerical differentiation (finite differences — too imprecise, costs $O(n)$ evaluations).  AD exploits the fact that every program is a composition of primitives, and the chain rule tells us exactly how to compose their derivatives.

**Two flavours:**

```
SYMBOLIC DIFF           NUMERICAL DIFF          AUTO DIFF
═══════════════════════════════════════════════════════════════

  f(x) = x² + sin(x)    Compute f(x+h)          Track ops in
                          and f(x-h)              computation tape

  → d/dx = 2x+cos(x)    → (f(x+h)-f(x-h))/2h    → Exact as FP allows

  Exact but expression   Approximate; costs       Exact, costs O(1)
  size can explode       O(n) evaluations          evaluations

═══════════════════════════════════════════════════════════════
```

### C.2 The Tape (Wengert List)

The **Wengert list** (1964) records, during the forward pass, every primitive operation applied and its operands.  The backward pass replays this tape in reverse, accumulating adjoints.

```
FORWARD TAPE EXAMPLE: f(x) = exp(x) · (x + 1)
═════════════════════════════════════════════════════════════

  Tape (built during forward):
    v₁ = x            (input)
    v₂ = exp(v₁)      (op: exp,  operand: v₁)
    v₃ = v₁ + 1.0     (op: add,  operands: v₁, 1.0)
    v₄ = v₂ × v₃      (op: mul,  operands: v₂, v₃)

  Backward (replay in reverse):
    v̄₄ = 1.0                          (seed)
    v̄₂ += v̄₄ × v₃  = 1.0 × (x+1)    (mul backward)
    v̄₃ += v̄₄ × v₂  = 1.0 × exp(x)   (mul backward)
    v̄₁ += v̄₃ × 1.0 = exp(x)          (add backward)
    v̄₁ += v̄₂ × exp(v₁) = (x+1)exp(x) (exp backward)

  Total: v̄₁ = exp(x) + (x+1)exp(x) = (x+2)exp(x) ✓ (by product rule)

═════════════════════════════════════════════════════════════
```

PyTorch's `Tensor` stores a `grad_fn` attribute at each node — this is the tape in disguise.  Calling `.backward()` replays the tape in reverse.

**For more:** See [§05 Automatic Differentiation](../05-Automatic-Differentiation/notes.md) for the complete treatment of forward/reverse mode AD, source transformation, operator overloading, and the design of JAX vs PyTorch autograd.


---

## Appendix D: Numerical Gradient Verification

In practice, every backpropagation implementation should be verified against finite differences.  This appendix presents the standard toolkit.

### D.1 Centred Finite Differences

For a scalar loss $\mathcal{L}: \mathbb{R}^n \to \mathbb{R}$ and parameter $\theta_i$:

$$\frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\boldsymbol{\theta} + h\mathbf{e}_i) - \mathcal{L}(\boldsymbol{\theta} - h\mathbf{e}_i)}{2h}$$

**Error analysis:** Centred differences have $O(h^2)$ error (vs $O(h)$ for forward differences).  Optimal step size $h$ balances truncation error ($O(h^2)$) against floating-point cancellation error ($O(\epsilon/h)$ where $\epsilon$ is machine epsilon):

$$h_\text{opt} \approx \epsilon^{1/3} \approx (10^{-16})^{1/3} \approx 10^{-5} \quad \text{(for float64)}$$

Use $h = 10^{-5}$ for float64 and $h = 10^{-3}$ for float32.

**Relative error check:** Accept the gradient check if:

$$\frac{\|\nabla_\text{backprop} - \nabla_\text{FD}\|}{\|\nabla_\text{backprop}\| + \|\nabla_\text{FD}\|} < 10^{-7} \quad \text{(float64)}$$

### D.2 When Gradient Checks Fail

Common failure modes:

| Symptom | Likely cause |
|---------|-------------|
| Relative error $\sim 10^{-3}$ throughout | `h` too large (truncation) or float32 precision |
| Relative error $\sim 1$ for specific parameters | Bug in backward for that parameter type |
| Relative error $\sim 0$ for all gradients | Loss is approximately linear in those parameters at the test point |
| Fails at kink (ReLU/max) | Gradient not defined at $z=0$; test point near kink; use $\mathbf{x}$ away from kinks |
| Fails only for batch size 1 | BatchNorm statistics degenerate; use batch size $\geq 2$ for BN checks |

### D.3 Gradient Check in PyTorch

```python
from torch.autograd import gradcheck

def f(x):
    return (x ** 2).sum()

x = torch.randn(5, requires_grad=True, dtype=torch.float64)
gradcheck(f, (x,), eps=1e-6, atol=1e-4, rtol=1e-4)
```

`gradcheck` automates centred finite differences for all inputs with `requires_grad=True`.  Always use `dtype=torch.float64` for gradient checking — float32 precision is insufficient for reliable checks.

---

## Appendix E: Key Formulas Reference

### E.1 Chain Rule Summary

| Setting | Formula |
|---------|---------|
| Scalar composition | $(f \circ g)'(x) = f'(g(x)) \cdot g'(x)$ |
| Vector composition | $J_{f \circ g}(\mathbf{x}) = J_f(g(\mathbf{x})) \cdot J_g(\mathbf{x})$ |
| VJP (backprop step) | $\bar{\mathbf{x}} = J_g(\mathbf{x})^\top \bar{\mathbf{y}}$ |
| JVP (forward step) | $\dot{\mathbf{y}} = J_g(\mathbf{x}) \dot{\mathbf{x}}$ |

### E.2 Backpropagation Formulas

| Layer | Forward | Backward ($\bar{\mathbf{z}}$ given) |
|-------|---------|-------------------------------------|
| Linear $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ | — | $\bar{\mathbf{x}} = W^\top \bar{\mathbf{z}}$, $\bar{W} = \bar{\mathbf{z}}\mathbf{x}^\top$, $\bar{\mathbf{b}} = \bar{\mathbf{z}}$ |
| Elementwise $\mathbf{a} = \sigma(\mathbf{z})$ | — | $\bar{\mathbf{z}} = \bar{\mathbf{a}} \odot \sigma'(\mathbf{z})$ |
| Softmax+CE | $p_i = e^{z_i}/\sum e^{z_j}$ | $\partial \mathcal{L}/\partial \mathbf{z} = \mathbf{p} - \mathbf{e}_y$ |
| Residual $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$ | — | $\bar{\mathbf{x}} = \bar{\mathbf{y}} + J_F^\top \bar{\mathbf{y}}$ |
| LayerNorm | $\hat{x}_i = (x_i-\mu)/\sigma$ | Complex (see §5.4); passes signal |

### E.3 Activation Derivatives

| Name | $\sigma(z)$ | $\sigma'(z)$ |
|------|-------------|--------------|
| ReLU | $\max(0,z)$ | $\mathbf{1}[z>0]$ |
| Sigmoid | $1/(1+e^{-z})$ | $\sigma(1-\sigma)$ |
| Tanh | $(e^z-e^{-z})/(e^z+e^{-z})$ | $1-\tanh^2$ |
| GELU | $z\Phi(z)$ | $\Phi(z)+z\phi(z)$ |
| SiLU | $z/(1+e^{-z})$ | $\sigma(z)(1+z(1-\sigma(z)))$ |
| Softplus | $\log(1+e^z)$ | $\sigma(z)$ |

### E.4 Initialisation Standards

| Method | Distribution | Variance | When |
|--------|-------------|---------|------|
| Xavier uniform | $U[-a,a]$ | $2/(n_\text{in}+n_\text{out})$ | Sigmoid, tanh |
| Xavier normal | $\mathcal{N}(0,\sigma^2)$ | $2/(n_\text{in}+n_\text{out})$ | Sigmoid, tanh |
| He uniform | $U[-a,a]$ | $2/n_\text{in}$ | ReLU |
| He normal | $\mathcal{N}(0,\sigma^2)$ | $2/n_\text{in}$ | ReLU |
| GPT-2 residual | $\mathcal{N}(0,(0.02/\sqrt{2L})^2)$ | — | Transformer residuals |


---

## Appendix F: Deep Dive — Vanishing Gradients in Transformers

### F.1 Why Transformers Don't Vanish

A naive reading of the vanishing gradient analysis (§6.1) suggests that 96-layer transformers should suffer catastrophic vanishing.  They don't.  Here is why.

**The residual stream analysis:** In a pre-norm transformer, the residual stream after layer $l$ is:

$$\mathbf{x}^{[l]} = \mathbf{x}^{[0]} + \sum_{k=1}^{l} F^{[k]}(\mathbf{x}^{[k-1]})$$

where $F^{[k]}$ is the $k$-th sublayer (attention or MLP, wrapped in LayerNorm).

The gradient of the loss with respect to the *input* is:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{[0]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{[L]}} \cdot \prod_{l=1}^{L}\left(I + J_{F^{[l]}}\right)$$

At initialisation, the transformer weights are small, so $J_{F^{[l]}} \approx 0$ and the product $\prod(I + J_{F^{[l]}}) \approx I$.  The gradient flows back unchanged through all $L$ layers.  This is categorically different from a plain deep network where the product of small Jacobians vanishes.

**Gradient norm growth:** As training progresses and weights grow, $J_{F^{[l]}}$ becomes nontrivial.  The gradient norm may grow with depth, but this is controlled by:
1. LayerNorm dampening (see §6.6)
2. GPT-2's $1/\sqrt{2L}$ scaling of residual projections
3. Gradient clipping ($\tau = 1.0$)

**The "edge of stability" phenomenon** (Cohen et al., 2022): In practice, the maximum Hessian eigenvalue $\lambda_\text{max}$ often approaches $2/\eta$ (twice the inverse learning rate) and oscillates there.  This is a gradient flow regime where the training dynamics are neither fully stable nor unstable, and gradients are large enough to cause oscillation but not divergence.

### F.2 Gradient Norm as Training Signal

Modern LLM training monitors gradient norm at every step.  Typical patterns:

```
GRADIENT NORM DURING LLM TRAINING
═══════════════════════════════════════════════════════════════

  ‖∇θ‖₂
    │
    │  spike (loss spike)
    │    │
  1 ─ ─ │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ clip threshold
    │ ╭──┤╮
    │╭╯  └╰────────────────────────── normal training
    │╯
    └────────────────────────────────────────────────── steps

  Patterns:
  • Steady ‖∇θ‖ < 1: healthy training, clipping inactive
  • Sudden spike → loss spike → recovery: numerical event
    (often a "bad" batch; LLM training has ~1-3 such events
     per trillion tokens at scale)
  • Slow upward drift: learning rate may be too high

═══════════════════════════════════════════════════════════════
```

**Loss spike mitigation:** When the gradient norm exceeds the clip threshold, the entire gradient update is scaled down.  If the spike is from a corrupted batch, this prevents permanent damage to the model weights.

**Gradient accumulation and norm:** When using $G$ accumulation steps, each micro-batch contributes $1/G$ of the gradient.  The global norm is computed across the *accumulated* gradient (after summation, before the optimiser step) — not across individual micro-batches.

### F.3 Per-Layer Gradient Norm Analysis

For diagnostic purposes, logging the gradient norm per layer reveals:
- **Embedding gradients:** Often the largest, due to sparse updates (§5.6)
- **Early layers:** Smallest (furthest from loss); potential vanishing
- **Late layers:** Largest; potential exploding
- **LayerNorm parameters:** Very small — $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$ converge quickly

This per-layer analysis guided the design of:
- **LARS/LAMB** optimisers (You et al., 2017): layer-wise adaptive learning rates based on weight-to-gradient ratio
- **Muon** (2024): applies Newton step in gradient space with Nesterov momentum; designed for hidden layers while AdamW handles embedding and output


---

## Appendix G: Historical Development

### G.1 Timeline of Backpropagation

The development of backpropagation spans three centuries and multiple independent discoveries:

| Year | Event | Significance |
|------|-------|-------------|
| 1676 | Leibniz publishes differential calculus (chain rule for single variable) | Mathematical foundation |
| 1744 | Euler uses variational methods (antecedent of reverse mode) | First "adjoint" idea |
| 1847 | Cauchy introduces gradient descent | The algorithm backprop serves |
| 1960 | Kalman filter (reverse-mode for linear dynamical systems) | AD in engineering |
| 1964 | Wengert introduces the "reverse accumulation" algorithm | First explicit AD |
| 1970 | Linnainmaa's thesis: general backpropagation | Full theoretical framework |
| 1974 | Werbos PhD thesis: backprop for neural networks | Connection to ML |
| 1982 | Hopfield networks (energy-based models with gradient) | Alternative to backprop |
| 1986 | Rumelhart, Hinton & Williams — "Learning representations by back-propagating errors" | Popularised backprop for NNs |
| 1991 | Hochreiter: vanishing gradient problem analysed | Identified depth barrier |
| 1997 | LSTM: gating to address vanishing gradient in RNNs | First scalable deep sequence model |
| 2012 | AlexNet: backprop on GPU at scale | Practical deep learning |
| 2015 | ResNets: residual connections for gradient flow | Enabled 100+ layer networks |
| 2016 | PyTorch / TensorFlow 1.0: autodiff frameworks | Democratised backprop |
| 2017 | Transformers: attention replaces BPTT | Solved long-range vanishing |
| 2018 | JAX: functional autodiff, JIT compilation | Research-grade AD |
| 2022 | FlashAttention: IO-aware backward pass | Efficient $O(T^2)$ attention backward |
| 2022 | PyTorch 2.0 `torch.compile` | Graph-based kernel fusion |
| 2023 | FlashAttention-2: improved GPU utilisation | Standard for production |
| 2024 | FlashAttention-3: H100-optimised with async | State-of-art attention backward |

### G.2 The Independent Discoveries

Backpropagation was independently discovered at least four times before becoming widely known:

1. **Linnainmaa (1970):** In his master's thesis, presented the general algorithm for computing exact partial derivatives of any function composed of elementary operations — precisely what we today call reverse-mode AD.

2. **Werbos (1974):** Applied the same idea to multi-layer neural networks in his PhD thesis, but the work was largely ignored for over a decade.

3. **Parker (1985):** Independently rediscovered backpropagation for neural networks.

4. **Rumelhart, Hinton & Williams (1986):** Published the algorithm in *Nature* and produced the critical experimental demonstrations that convinced the community it could work.  Their paper is the one most often cited today.

This pattern of independent rediscovery is common in mathematics — the ideas are "in the air" once the prerequisites are established.  The chain rule (1676) + computation graphs (1960s) + gradient descent (1847) = backpropagation (inevitable).

### G.3 The Hardware-Algorithm Co-evolution

The practical impact of backpropagation depends critically on hardware:

- **CPU era (1986–2011):** Backprop is theoretically valid but computationally slow.  Networks with more than 3-4 layers were impractical.
- **GPU era (2012–present):** NVIDIA's CUDA (2007) enables massively parallel GEMM operations.  The bottleneck shifts from FLOPS to memory bandwidth.
- **Tensor core era (2017–present):** NVIDIA Volta/Ampere/Hopper GPUs have dedicated matrix multiply accelerators.  FP16/BF16 tensor cores achieve 10× the throughput of FP32.
- **Memory wall:** As models scale, the backward pass's memory requirements dominate.  FlashAttention, ZeRO, gradient checkpointing all address the memory wall.

The 2024 FLOP/memory ratio in H100 GPUs ($\sim 2000$ TFLOPS vs $\sim 3.35$ TB/s bandwidth) means that memory access, not computation, is the primary bottleneck for backprop at scale.  This fundamental constraint is why FlashAttention's IO-aware design is so impactful.


---

## Appendix H: Connections to Optimisation and Learning Theory

### H.1 What the Gradient Tells Us

The gradient $\nabla_\theta \mathcal{L}$ computed by backpropagation is the direction of steepest ascent in parameter space (by the first-order Taylor expansion).  Gradient descent moves in the opposite direction:

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$$

**What the gradient does NOT tell us:**
- The curvature of the loss landscape (need Hessian for that)
- The optimal step size $\eta$
- Whether we are near a local minimum, saddle point, or maximum
- Whether the gradient is statistically well-estimated (needs large enough batch)

**What the gradient DOES tell us:**
- The direction of maximal increase (used negated for descent)
- The sensitivity of the loss to each parameter
- Which parameters are "active" (nonzero gradient) vs. saturated (near-zero gradient)

### H.2 Gradient Stochasticity

In practice, the true gradient $\nabla_\theta \mathbb{E}[\mathcal{L}]$ over the full data distribution is approximated by the **stochastic gradient** over a mini-batch:

$$\hat{g}_B = \frac{1}{B}\sum_{b=1}^B \nabla_\theta \mathcal{L}(\mathbf{x}^{(b)}, \mathbf{y}^{(b)})$$

This is an unbiased estimator: $\mathbb{E}[\hat{g}_B] = \nabla_\theta \mathbb{E}[\mathcal{L}]$.

**Variance:** $\text{Var}(\hat{g}_B) = \text{Var}(\nabla_\theta \mathcal{L})/B$.  Larger batches have lower gradient variance (more accurate gradient estimate) but provide diminishing returns beyond the "critical batch size" (McCandlish et al., 2018).

**For LLMs:** The critical batch size for GPT-3-scale models is approximately $B^* \approx 1-4$ million tokens.  Training at this batch size achieves the best loss-per-FLOP tradeoff.  Using larger batches wastes compute; using smaller batches wastes gradient estimation quality.

### H.3 The Gradient as a Sufficient Statistic

For first-order optimisers (SGD, Adam, AdaGrad, RMSprop), the gradient is the only information extracted from the forward-backward pass.  Second-order information (Hessian curvature) is either ignored or approximated.

**Why not use the full Hessian?** For $|\theta| = 70\text{B}$ parameters, the Hessian is a $70\text{B} \times 70\text{B}$ matrix — $\sim 10^{22}$ entries.  Storing it is impossible ($10^{22}$ FP32 values ≈ $4 \times 10^{22}$ bytes ≈ $40 \times 10^{21}$ GB).  Inverting it is even more impossible.

**Practical second-order methods** use approximations:
- **Diagonal:** AdaGrad/Adam maintain diagonal Hessian approximations ($O(|\theta|)$ memory)
- **Kronecker factored:** K-FAC (see §02) uses $A \otimes G$ per layer ($O(n^2)$ per layer)
- **Low-rank:** PSGD, Shampoo maintain low-rank or block-diagonal approximations
- **Newton-Schulz:** Muon (2024) approximates the matrix square root $H^{-1/2}$ efficiently

### H.4 Generalisation and the Implicit Gradient Bias

Gradient descent with small learning rate and large mini-batches does not merely find *any* minimum — it has an **implicit bias** toward flat minima (large regions with low loss) over sharp minima (narrow valleys).

**Conjecture (Keskar et al., 2017):** Flat minima generalise better because small perturbations to the parameters don't change the loss much — robust to noise in the data.

**Mathematical foundation:** The SGD noise $\hat{g}_B - g$ effectively adds a regularisation term proportional to $\eta B^{-1} \text{tr}(H)$ — the trace of the Hessian — biasing toward flat (low-trace-Hessian) minima.

This connects gradient computation (the topic of this section) to generalisation theory (a major open question in deep learning theory) — a reminder that understanding backpropagation fully requires understanding not just the mechanics, but the geometry of the loss landscape it navigates.


---

## Appendix I: Practical Implementation Guide

### I.1 Implementing Backprop from Scratch

When building a neural network framework from scratch, implement these components in order:

**1. Primitive registry:**

```python
primitives = {}

def register_primitive(name, forward_fn, backward_fn):
    """Register a primitive op with its VJP."""
    primitives[name] = (forward_fn, backward_fn)

# Example: multiplication primitive
def mul_forward(x, y): return x * y
def mul_backward(x, y, g_out): return g_out * y, g_out * x  # (g_x, g_y)
register_primitive('mul', mul_forward, mul_backward)
```

**2. Value class with gradient tracking:**

```python
class Value:
    def __init__(self, data, parents=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None  # closure capturing parents
        self._parents = parents
        self._op = op
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), 'mul')
        def _backward():
            self.grad += other.data * out.grad   # VJP for self
            other.grad += self.data * out.grad   # VJP for other
        out._backward = _backward
        return out
    
    def backward(self):
        # Topological sort, then reverse
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for p in v._parents: build_topo(p)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo): v._backward()
```

This is essentially the complete autograd engine from Karpathy's `micrograd` (2020) — approximately 100 lines implement a working backprop engine.

**3. Building blocks:** Extend `Value` with `__add__`, `__pow__`, `exp`, `log`, `relu`, `softmax` — each with its VJP closure.

### I.2 Common Implementation Bugs

**Bug 1: Overwriting instead of accumulating gradients**
```python
# Wrong:
self.grad = other.data * out.grad   # erases previous contributions!

# Correct:
self.grad += other.data * out.grad  # accumulates (fan-out nodes)
```

**Bug 2: Forgetting to zero gradients between batches**
```python
# Wrong: gradient accumulates across batches
loss = model(x)
loss.backward()
optimizer.step()

# Correct:
optimizer.zero_grad()   # ← must come before backward
loss = model(x)
loss.backward()
optimizer.step()
```

**Bug 3: Not detaching from the graph for inference**
```python
# Wrong: builds graph unnecessarily during inference
with torch.no_grad():  # ← this is the correct fix
    prediction = model(x)
```

**Bug 4: Shape mismatch in weight gradient**
```python
# Wrong: grad_W and W may have different shapes
grad_W = delta @ x    # (n_out, 1) @ (1, n_in) only works for batch size 1

# Correct: outer product for single sample
grad_W = np.outer(delta, x)  # (n_out, n_in)

# Correct: batched
grad_W = (1/B) * Delta @ X.T  # (n_out, B) @ (B, n_in) = (n_out, n_in)
```

### I.3 Testing Checklist

Before deploying any backprop implementation:

- [ ] Gradient check passes for all primitive operations (relative error $< 10^{-6}$)
- [ ] Loss decreases monotonically for small enough learning rate (verify on toy problem)
- [ ] Gradients are zero for frozen parameters
- [ ] Gradient accumulation at fan-out nodes verified (shared weight receives sum)
- [ ] Shape of each gradient matches shape of corresponding parameter
- [ ] Memory usage is $O(\text{num\_layers})$ not $O(\text{num\_layers}^2)$
- [ ] Higher-order gradients work if needed (use `create_graph=True` in PyTorch)
- [ ] Mixed precision: FP16 forward, FP32 gradient accumulation, loss scaling in place


---

## Appendix J: Connections to Information Theory and Statistics

### J.1 Fisher Information and the Natural Gradient

The ordinary gradient $\nabla_\theta \mathcal{L}$ measures the steepest direction in parameter space with respect to the Euclidean metric.  But parameter space has a natural metric induced by the probability distribution $p_\theta$ — the **Fisher information metric**.

**Fisher information matrix:**
$$F(\theta) = \mathbb{E}_{x \sim p_\theta}\left[\nabla_\theta \log p_\theta(x) \, (\nabla_\theta \log p_\theta(x))^\top\right]$$

**Natural gradient** (Amari, 1998):
$$\tilde{\nabla}_\theta \mathcal{L} = F(\theta)^{-1} \nabla_\theta \mathcal{L}$$

The natural gradient is the steepest direction in the **distributional geometry** of the model — invariant to reparametrisation.  Computing it exactly requires inverting $F$, which costs $O(|\theta|^3)$.

**K-FAC** (§02) approximates $F^{-1}$ as a Kronecker product, making the natural gradient step tractable.  It remains the most principled second-order optimiser for neural networks.

**For LLMs:** The approximation used in practice is Adam's diagonal $F^{-1}$ (second moment of gradient as proxy for diagonal Fisher).  This is crude but sufficient — Adam is a diagonal natural gradient step.

### J.2 Gradient as Score Function

For a probabilistic model $p_\theta(\mathbf{x})$, the gradient of the log-likelihood is the **score function**:

$$s(\mathbf{x}; \theta) = \nabla_\theta \log p_\theta(\mathbf{x})$$

The score function is the quantity computed by backpropagation during maximum likelihood estimation.  Properties:
- $\mathbb{E}_{x \sim p_\theta}[s(\mathbf{x};\theta)] = 0$ (score has zero mean)
- $\text{Var}_{x \sim p_\theta}[s(\mathbf{x};\theta)] = F(\theta)$ (Fisher information = variance of score)

**For language models:** The negative log-likelihood $\mathcal{L} = -\log p_\theta(\mathbf{y}|\mathbf{x})$ has gradient $-s(\mathbf{y}|\mathbf{x};\theta) = -(p_y - e_y)^\top = e_y - p_y$ — the same $\mathbf{p} - \mathbf{e}_y$ formula from §5.3, now understood as the negative score.

### J.3 KL Divergence and the Gradient of ELBO

In variational inference and RL (RLHF), we often need gradients of KL divergences.  For discrete distributions:

$$\text{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

$$\nabla_\phi \text{KL}(p_\theta \| q_\phi) = -\mathbb{E}_{x \sim p_\theta}\left[\nabla_\phi \log q_\phi(x)\right]$$

This is computed via backprop through the log-probability of the policy under KL regularisation — the precise form used in RLHF's PPO loss, which includes a KL penalty between the fine-tuned policy $\pi_\phi$ and the reference model $\pi_\text{ref}$.

---

## References

1. **Rumelhart, Hinton & Williams (1986)** — "Learning representations by back-propagating errors." *Nature*, 323, 533–536. The canonical backpropagation paper.

2. **Linnainmaa, S. (1970)** — "The representation of the cumulative rounding error of an algorithm as a Taylor expansion of the local rounding errors." Master's thesis, University of Helsinki. First general reverse-mode AD.

3. **Hochreiter, S. (1991)** — "Untersuchungen zu dynamischen neuronalen Netzen." Diploma thesis, TU Munich. First analysis of vanishing gradients.

4. **Glorot, X. & Bengio, Y. (2010)** — "Understanding the difficulty of training deep feedforward neural networks." *AISTATS*. Xavier initialisation.

5. **He, K. et al. (2015)** — "Delving Deep into Rectifiers." *ICCV*. He initialisation for ReLU networks.

6. **He, K. et al. (2016)** — "Deep Residual Learning for Image Recognition." *CVPR*. ResNets and gradient highways.

7. **Ba, J. et al. (2016)** — "Layer Normalization." *arXiv:1607.06450*. LayerNorm for transformers.

8. **Vaswani, A. et al. (2017)** — "Attention Is All You Need." *NeurIPS*. Transformer architecture with attention backward pass.

9. **Amari, S. (1998)** — "Natural Gradient Works Efficiently in Learning." *Neural Computation*. Natural gradient and Fisher information.

10. **Martens, J. & Grosse, R. (2015)** — "Optimizing Neural Networks with Kronecker-factored Approximate Curvature." *ICML*. K-FAC.

11. **Hu, E. et al. (2022)** — "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*. LoRA backward pass.

12. **Dao, T. et al. (2022)** — "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." *NeurIPS*. IO-aware backward for attention.

13. **Cohen, J. et al. (2022)** — "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability." *ICLR*. Edge of stability phenomenon.

14. **Dao, T. (2023)** — "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning." *ICLR 2024*. FlashAttention-2.

15. **Liu, S. et al. (2024)** — "DoRA: Weight-Decomposed Low-Rank Adaptation." DoRA backward analysis.


---

## Appendix K: Summary Tables

### K.1 Backpropagation Algorithm Summary

```
COMPLETE BACKPROPAGATION ALGORITHM
════════════════════════════════════════════════════════════════════════

  INPUT: network weights θ, training pair (x, y)

  PHASE 1 — FORWARD PASS
  ───────────────────────
  a⁰ = x
  For l = 1, 2, ..., L:
    z^l = W^l a^{l-1} + b^l       (cache z^l and a^{l-1})
    a^l = σ^l(z^l)                 (cache a^l)
  ŷ = a^L
  ℒ = loss(ŷ, y)

  PHASE 2 — BACKWARD PASS
  ───────────────────────
  δ^L = ∂ℒ/∂z^L                  (output layer gradient, layer-specific)
  For l = L-1, L-2, ..., 1:
    δ^l = (W^{l+1})ᵀ δ^{l+1} ⊙ σ'^l(z^l)

  PHASE 3 — GRADIENT ASSEMBLY
  ────────────────────────────
  For l = 1, 2, ..., L:
    ∇_{W^l} ℒ = δ^l (a^{l-1})ᵀ
    ∇_{b^l} ℒ = δ^l

  PHASE 4 — PARAMETER UPDATE
  ───────────────────────────
  θ ← θ - η · ∇_θ ℒ             (or Adam/RMSprop update)

════════════════════════════════════════════════════════════════════════
```

### K.2 Complexity Summary

| Operation | Time | Memory |
|-----------|------|--------|
| Forward pass (L layers, width n) | $O(Ln^2)$ | $O(Ln)$ cached activations |
| Backward pass | $O(Ln^2)$ | $O(Ln)$ error signals |
| Full Jacobian via FD | $O(|\theta| \cdot T_f)$ | $O(|\theta|)$ |
| Full Jacobian via backprop | $O(m \cdot T_f)$ | $O(L)$ |
| Hessian-vector product | $O(T_f)$ | $O(L)$ |
| Gradient checkpointing | $O(1.33 T_f)$ | $O(\sqrt{L})$ |
| FlashAttention forward | $O(T^2 d)$ | $O(T)$ |
| FlashAttention backward | $O(T^2 d)$ | $O(T)$ |

### K.3 Gradient Flow Interventions

| Problem | Diagnosis | Intervention |
|---------|-----------|-------------|
| Vanishing gradients | $\|\boldsymbol{\delta}^{[1]}\| \ll 1$ | ReLU/GELU, He init, residual connections |
| Exploding gradients | $\|\boldsymbol{\delta}^{[1]}\| \gg 1$ | Gradient clipping, LR warmup |
| Dead neurons | $\|\boldsymbol{\delta}^{[l]}\| = 0$ for layer | Leaky ReLU, better init, BN |
| Slow convergence | $\|\nabla_\theta \mathcal{L}\| \approx 0$ at saddle | Momentum, Adam, noise injection |
| Oscillating loss | $\|\nabla_\theta \mathcal{L}\|$ spikes | Reduce LR, increase batch |
| NaN gradients | $\|\nabla_\theta \mathcal{L}\| = \infty$ | Loss scaling, check log/softmax |

