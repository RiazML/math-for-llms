[← Back to Multivariate Calculus](../README.md) | [Next: Chain Rule and Backpropagation →](../03-Chain-Rule-and-Backpropagation/notes.md)

---

# Jacobians and Hessians

> _"The Jacobian is the derivative — not a number, not a vector, but a linear map. Once you see it that way, everything in deep learning becomes composition of linear maps."_
> — paraphrasing Michael Spivak, _Calculus on Manifolds_

## Overview

The gradient $\nabla f(\mathbf{x}) \in \mathbb{R}^n$ is the right object when $f$ is scalar-valued. Neural networks are not scalar-valued: every layer maps a vector to a vector, every activation transforms a vector elementwise, and the full forward pass is a long chain of vector-to-vector maps. When you differentiate a vector-valued function you get a **matrix** of partial derivatives — the **Jacobian**. The Jacobian is the derivative in the correct sense: it is the unique linear map that best approximates the function near a point.

The **Hessian** is the second derivative. Once you have the gradient $\nabla f: \mathbb{R}^n \to \mathbb{R}^n$ (which is vector-valued), you can differentiate it again to get the Jacobian of the gradient — an $n \times n$ matrix called the Hessian $H_f$. The Hessian encodes **curvature**: how quickly the gradient rotates and stretches as you move through parameter space. Curvature determines whether a critical point is a minimum, maximum, or saddle; it determines how fast gradient descent converges; it is the object that second-order optimisers (Newton, K-FAC, Shampoo) use to accelerate training.

This section builds both objects completely. We prove every key result from first principles: the Fréchet derivative definition, Clairaut's theorem with a complete proof and a counterexample, the second-order Taylor theorem with remainder, the Newton step derivation, and the efficient HVP identity. We derive from scratch the Jacobian of every major ML operation: linear layers, elementwise activations, softmax, layer normalisation. We connect everything to modern practice: why backpropagation is VJP composition, why the K-FAC approximation works, why SAM seeks flat minima, and how influence functions use HVPs.

## Prerequisites

- **Partial derivatives, gradient, directional derivatives** — [§01 Partial Derivatives and Gradients](../01-Partial-Derivatives-and-Gradients/notes.md)
- **Matrix–vector multiplication, transpose, trace, singular values** — [§02 Linear Algebra Basics](../../02-Linear-Algebra-Basics/README.md)
- **Eigenvalue decomposition, positive definite matrices** — [§07 Positive Definite Matrices](../../03-Advanced-Linear-Algebra/07-Positive-Definite-Matrices/notes.md)
- **Single-variable chain rule and Taylor series** — [§02–§04 Calculus Fundamentals](../../04-Calculus-Fundamentals/README.md)

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | Jacobian and Hessian computation, ML layer Jacobians, Taylor approximation, Hessian spectrum, Newton's method, HVPs |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from Jacobian computation through Gauss-Newton and SAM |

## Learning Objectives

After completing this section, you will be able to:

1. Define the Jacobian $J_f \in \mathbb{R}^{m \times n}$ for $f: \mathbb{R}^n \to \mathbb{R}^m$ via the Fréchet derivative and compute it for any differentiable function
2. Prove Clairaut's theorem on symmetry of mixed partial derivatives, and give a concrete counterexample showing when it fails
3. Define the Hessian as $H_f = J_{\nabla f}$ and compute it analytically for quadratic, log-sum-exp, and neural network loss functions
4. Derive the second-order multivariate Taylor theorem with explicit remainder, and use it to build quadratic models of loss functions
5. Classify critical points using Hessian eigenvalues: positive definite → strict minimum, indefinite → saddle
6. Derive from scratch the Jacobian of a linear layer, elementwise activation, softmax, and layer normalisation — including the full proof for softmax
7. Explain JVPs (forward mode) and VJPs (reverse mode), and prove why reverse mode requires only one pass for scalar loss
8. Derive the Newton step from the quadratic model and explain its quadratic convergence
9. Implement Hessian-vector products in $O(n)$ without materialising the $n \times n$ Hessian
10. State and apply the Gauss-Newton and K-FAC approximations, explaining why $H_{\text{GN}} = J^\top J \succeq 0$

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 From Gradient to Jacobian](#11-from-gradient-to-jacobian)
  - [1.2 From Gradient to Hessian](#12-from-gradient-to-hessian)
  - [1.3 Historical Context](#13-historical-context)
  - [1.4 Why These Matrices Define Deep Learning](#14-why-these-matrices-define-deep-learning)
- [2. The Jacobian Matrix](#2-the-jacobian-matrix)
  - [2.1 Formal Definition via Fréchet Derivative](#21-formal-definition-via-fréchet-derivative)
  - [2.2 Special Cases](#22-special-cases)
  - [2.3 Computing Jacobians Systematically](#23-computing-jacobians-systematically)
  - [2.4 Jacobian Determinant and Volume Scaling](#24-jacobian-determinant-and-volume-scaling)
  - [2.5 Jacobian of a Composition (Preview)](#25-jacobian-of-a-composition-preview)
  - [2.6 Worked Examples with Full Derivations](#26-worked-examples-with-full-derivations)
- [3. Jacobians of Key ML Operations](#3-jacobians-of-key-ml-operations)
  - [3.1 Linear Layer](#31-linear-layer)
  - [3.2 Elementwise Activations](#32-elementwise-activations)
  - [3.3 Softmax Jacobian — Full Proof](#33-softmax-jacobian--full-proof)
  - [3.4 Layer Normalization Jacobian — Full Derivation](#34-layer-normalization-jacobian--full-derivation)
  - [3.5 JVPs and VJPs — Why Reverse Mode Dominates](#35-jvps-and-vjps--why-reverse-mode-dominates)
  - [3.6 Jacobian Condition Number and Training Stability](#36-jacobian-condition-number-and-training-stability)
- [4. The Hessian Matrix](#4-the-hessian-matrix)
  - [4.1 Formal Definition](#41-formal-definition)
  - [4.2 Clairaut's Theorem — Complete Proof and Failure Case](#42-clairauts-theorem--complete-proof-and-failure-case)
  - [4.3 Computing the Hessian](#43-computing-the-hessian)
  - [4.4 Hessian as Jacobian of the Gradient](#44-hessian-as-jacobian-of-the-gradient)
  - [4.5 Worked Examples with Full Derivations](#45-worked-examples-with-full-derivations)
- [5. Second-Order Taylor Expansion](#5-second-order-taylor-expansion)
  - [5.1 Multivariate Taylor Formula — Proof](#51-multivariate-taylor-formula--proof)
  - [5.2 Quadratic Approximation Geometry](#52-quadratic-approximation-geometry)
  - [5.3 Hessian Eigendecomposition and Principal Curvatures](#53-hessian-eigendecomposition-and-principal-curvatures)
  - [5.4 Gradient Descent vs Newton Step — Derivation and Comparison](#54-gradient-descent-vs-newton-step--derivation-and-comparison)
- [6. Hessian Definiteness and Curvature](#6-hessian-definiteness-and-curvature)
  - [6.1 Definiteness via Eigenvalues](#61-definiteness-via-eigenvalues)
  - [6.2 Loss Landscape Curvature and Flat Minima](#62-loss-landscape-curvature-and-flat-minima)
  - [6.3 Hessian Spectrum in Practice](#63-hessian-spectrum-in-practice)
  - [6.4 Condition Number and GD Convergence](#64-condition-number-and-gd-convergence)
  - [6.5 Sharpness-Aware Minimisation (SAM)](#65-sharpness-aware-minimisation-sam)
- [7. Newton's Method and Second-Order Optimisation](#7-newtons-method-and-second-order-optimisation)
  - [7.1 Newton Step — Full Derivation](#71-newton-step--full-derivation)
  - [7.2 Damped Newton and Levenberg-Marquardt](#72-damped-newton-and-levenberg-marquardt)
  - [7.3 Gauss-Newton for Least Squares](#73-gauss-newton-for-least-squares)
  - [7.4 K-FAC — Kronecker Factored Curvature](#74-k-fac--kronecker-factored-curvature)
- [8. Hessian-Vector Products](#8-hessian-vector-products)
  - [8.1 The $O(n)$ HVP Identity — Proof](#81-the-on-hvp-identity--proof)
  - [8.2 Power Iteration for Top Eigenvalue](#82-power-iteration-for-top-eigenvalue)
  - [8.3 Lanczos Algorithm for Full Spectrum](#83-lanczos-algorithm-for-full-spectrum)
  - [8.4 HVPs in ML: Influence Functions, Pruning, SAM](#84-hvps-in-ml-influence-functions-pruning-sam)
- [9. Advanced Topics](#9-advanced-topics)
  - [9.1 Generalised Jacobian for Non-Smooth Functions](#91-generalised-jacobian-for-non-smooth-functions)
  - [9.2 Implicit Function Theorem](#92-implicit-function-theorem)
  - [9.3 Fisher Information Matrix as Expected Hessian](#93-fisher-information-matrix-as-expected-hessian)
  - [9.4 Differential Geometry Perspective](#94-differential-geometry-perspective)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI (2026 Perspective)](#12-why-this-matters-for-ai-2026-perspective)
- [Conceptual Bridge](#conceptual-bridge)

---

## 1. Intuition

### 1.1 From Gradient to Jacobian

Recall from §01 that for a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the gradient $\nabla f(\mathbf{x}) \in \mathbb{R}^n$ captures all the first-order information: knowing $\nabla f(\mathbf{x})$, we can predict the change in $f$ along any direction $\mathbf{u}$ as $\nabla f(\mathbf{x})^\top \mathbf{u}$. The fundamental linear approximation is:
$$f(\mathbf{x}+\boldsymbol{\delta}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^\top\boldsymbol{\delta}$$

Now suppose the function is **vector-valued**: $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$. It produces $m$ output numbers from $n$ input numbers. We need to capture how each of the $m$ outputs changes as we vary each of the $n$ inputs. There are $m \times n$ such partial derivatives — they form a matrix. This matrix is the **Jacobian**.

The linear approximation generalises cleanly: if $\boldsymbol{\delta} \in \mathbb{R}^n$ is a small input perturbation, then the output perturbation is approximately:
$$\mathbf{f}(\mathbf{x}+\boldsymbol{\delta}) \approx \mathbf{f}(\mathbf{x}) + J_f(\mathbf{x})\,\boldsymbol{\delta} \in \mathbb{R}^m$$

where $J_f(\mathbf{x}) \in \mathbb{R}^{m \times n}$ is the Jacobian. The product $J_f\boldsymbol{\delta}$ is a matrix-vector product — not a dot product. This is the key structural difference between scalar and vector differentiation.

```
SCALAR FUNCTION vs VECTOR FUNCTION
════════════════════════════════════════════════════════════════════════

  f: ℝⁿ → ℝ                      g: ℝⁿ → ℝᵐ
  ────────────────────            ───────────────────────────────
  Derivative: ∇f ∈ ℝⁿ            Derivative: Jg ∈ ℝᵐˣⁿ

  df ≈ ∇f · δ                    dg ≈ Jg δ
       (dot product)                   (matrix-vector product)

  Row i of Jg = (∇gᵢ)ᵀ          gradient of i-th output component
  Col j of Jg = ∂g/∂xⱼ          how ALL outputs respond to xⱼ

  Shape check:
    δ ∈ ℝⁿ,  Jg ∈ ℝᵐˣⁿ  →  Jgδ ∈ ℝᵐ  ✓

════════════════════════════════════════════════════════════════════════
```

**Why this matters for neural networks.** A transformer layer takes an embedding $\mathbf{x} \in \mathbb{R}^d$ and produces another embedding $\mathbf{y} \in \mathbb{R}^d$. The Jacobian $J_f(\mathbf{x}) \in \mathbb{R}^{d \times d}$ captures how every output dimension responds to every input dimension. In backpropagation, when a gradient $\boldsymbol{\delta}^{\text{out}} \in \mathbb{R}^d$ arrives from the next layer, the upstream gradient is $\boldsymbol{\delta}^{\text{in}} = J_f(\mathbf{x})^\top \boldsymbol{\delta}^{\text{out}} \in \mathbb{R}^d$ — the transposed Jacobian acting on the incoming gradient. **Backpropagation is Jacobian transposition applied repeatedly**.

The Jacobian also governs whether information propagates through the network without distortion. If $\|J_f\|_2 \gg 1$, gradients explode. If $\|J_f\|_2 \ll 1$, gradients vanish. This is why initialisation schemes (Xavier, He) and normalisation layers (BatchNorm, LayerNorm) are designed to keep Jacobians near the identity in spectral norm.

### 1.2 From Gradient to Hessian

The gradient $\nabla f: \mathbb{R}^n \to \mathbb{R}^n$ is a vector-valued function. We just established that vector-valued functions have Jacobians. So: **what is the Jacobian of the gradient?** It is the Hessian:
$$H_f(\mathbf{x}) = J_{\nabla f}(\mathbf{x}) \in \mathbb{R}^{n \times n}$$

The $(i,j)$ entry of $H_f$ is:
$$[H_f]_{ij} = \frac{\partial [\nabla f]_i}{\partial x_j} = \frac{\partial}{\partial x_j}\frac{\partial f}{\partial x_i} = \frac{\partial^2 f}{\partial x_j \partial x_i}$$

This measures how the $i$-th component of the gradient changes when you increase $x_j$. Geometrically: the Hessian describes how the gradient **rotates and stretches** as you move through space. This is **curvature**.

**Curvature in direction $\mathbf{u}$:** Consider the function restricted to the line through $\mathbf{x}$ in direction $\mathbf{u}$: $g(t) = f(\mathbf{x}+t\mathbf{u})$. The curvature of $g$ at $t=0$ is:
$$g''(0) = \frac{d^2}{dt^2}\bigg|_{t=0} f(\mathbf{x}+t\mathbf{u}) = \mathbf{u}^\top H_f(\mathbf{x})\,\mathbf{u}$$

The Hessian thus answers: "how fast does $f$ curve when I walk in direction $\mathbf{u}$?" The answer is the **Rayleigh quotient** $\mathbf{u}^\top H_f \mathbf{u}$.

**Why sign of curvature matters for optimisation.** If every direction $\mathbf{u}$ has positive curvature ($\mathbf{u}^\top H_f \mathbf{u} > 0$ for all $\mathbf{u} \neq \mathbf{0}$), the function is locally bowl-shaped — every direction goes up from the bottom, confirming we are at a local minimum. If some direction has negative curvature, we can slide downhill in that direction — we are at a saddle point, not a minimum. If all curvatures are negative, we are at a local maximum.

```
CURVATURE AND CRITICAL POINTS
════════════════════════════════════════════════════════════════════════

  All eigenvalues λᵢ > 0:    uᵀHu > 0 ∀u ≠ 0
  ─────────────────────────────────────────────────────────
        f
       ∧ ∧
      / bowl \   ←  curved upward in ALL directions
     /        \     local MINIMUM  (∇f = 0 here)

  Mixed eigenvalues:         ∃u,v: uᵀHu > 0, vᵀHv < 0
  ─────────────────────────────────────────────────────────
      ↗ ↗ ↘ ↘
        saddle               goes up in some, down in others
                             SADDLE POINT  (∇f = 0 here)

  All eigenvalues λᵢ < 0:    uᵀHu < 0 ∀u ≠ 0
  ─────────────────────────────────────────────────────────
     \        /
      \ hill /   ←  curved downward in ALL directions
       ∨ ∨ ∨       local MAXIMUM  (∇f = 0 here)

════════════════════════════════════════════════════════════════════════
```

For neural networks, the Hessian of the loss $H_\mathcal{L}(\boldsymbol{\theta})$ has a rich spectrum: thousands of near-zero eigenvalues (flat directions in parameter space) and a handful of large eigenvalues (sharply curved directions, often corresponding to the most influential parameters). The ratio $\lambda_{\max}/\lambda_{\min}$ — the condition number — determines how hard the optimisation problem is for gradient descent.

### 1.3 Historical Context

| Year | Contributor | Development |
|------|-------------|-------------|
| 1841 | Carl Jacobi | Introduced functional determinants in the study of coordinate transforms — what we now call the Jacobian matrix |
| 1844 | Ludwig Hesse | Second-order determinant for classifying critical points of functions of two variables |
| 1867 | Hesse | Expanded to the full matrix of second partials; term "Hessian" coined posthumously |
| 1912 | Morse | Morse lemma: non-degenerate critical points (det $H \neq 0$) have a normal form determined entirely by the Hessian |
| 1960s | Levenberg, Marquardt | Gauss-Newton + regularisation for nonlinear least squares — still the workhorse for nonlinear optimisation |
| 1987 | Rumelhart, Hinton, Williams | Popularised backpropagation = repeated VJP composition through Jacobians |
| 1992 | LeCun, Simard, Pearlmutter | Diagonal Hessian approximation for adaptive learning rates — precursor to Adagrad/Adam |
| 2002 | Amari | Natural gradient = Fisher information (expected Hessian of log-likelihood) as preconditioner |
| 2015 | Martens & Grosse | K-FAC: Kronecker-factored curvature approximation; practical second-order training |
| 2018 | Ghorbani, Krishnan, Xiao | Empirical Hessian spectrum of deep networks: bulk + outlier structure |
| 2020 | Foret et al. | SAM: sharpness-aware minimisation; seeking parameters with small $\lambda_{\max}(H)$ |
| 2022 | Vyas et al. | Influence functions at GPT scale via HVPs and the Arnoldi method |
| 2023 | Google Brain | Distributed Shampoo in production LLM training; K-FAC for trillion-parameter models |

### 1.4 Why These Matrices Define Deep Learning

Every significant algorithm in modern deep learning uses Jacobians or Hessians, often implicitly:

**Jacobians are present in:**

- **Backpropagation.** For a network $\mathcal{L} = \ell(f_L(\cdots f_1(\mathbf{x})))$, the gradient with respect to the input of layer $l$ is computed via $\boldsymbol{\delta}^{(l)} = J_{f_l}(\mathbf{x}^{(l)})^\top \boldsymbol{\delta}^{(l+1)}$. Every backward pass is a sequence of transposed Jacobian multiplications.

- **LoRA and low-rank updates.** The weight gradient for a linear layer is $\nabla_W \mathcal{L} = \boldsymbol{\delta}\mathbf{a}^\top$ — a rank-1 outer product. LoRA exploits that the **effective** gradient update across a training run lives in a low-rank subspace, parameterising $\Delta W = AB$ with rank $r \ll \min(m,n)$.

- **Normalising flows.** Generative models (RealNVP, Glow, Neural ODE) learn invertible maps. The log-likelihood requires $\log|\det J_f(\mathbf{x})|$ — the log-volume scaling factor of the Jacobian.

- **Gradient explosion/vanishing.** For a depth-$L$ network, $\|\partial\mathcal{L}/\partial\mathbf{x}^{(1)}\|_2 \leq \prod_{l=1}^L \|J_{f_l}\|_2$. If each $\|J_{f_l}\|_2 > 1$, gradients grow exponentially (explosion); if $< 1$, they decay exponentially (vanishing).

**Hessians are present in:**

- **Newton's method and quasi-Newton (L-BFGS).** The Newton step $\boldsymbol{\delta}^* = -H^{-1}\nabla\mathcal{L}$ converges quadratically near the minimum. L-BFGS approximates $H^{-1}$ via a rank-$k$ update from gradient differences.

- **K-FAC and Shampoo.** Approximate the Hessian (or Fisher) using Kronecker products of per-layer statistics. Used in production training at Google (Shampoo for Gemini-scale models).

- **SAM (Sharpness-Aware Minimisation).** Explicitly targets parameters where $\lambda_{\max}(H_\mathcal{L})$ is small, achieving better generalisation by finding flat minima.

- **Influence functions.** The influence of training example $i$ on test loss is $\approx -\nabla\mathcal{L}_{\text{test}}^\top H^{-1}\nabla\mathcal{L}_i$, requiring solving a linear system $H\mathbf{u} = \nabla\mathcal{L}_{\text{test}}$ via Hessian-vector products.

- **Loss landscape analysis.** The ratio $\lambda_{\max}/\lambda_{\min}$ determines convergence speed; the spectral density reveals whether the landscape is nearly flat (overparameterised regime) or sharply curved.

---

## 2. The Jacobian Matrix

### 2.1 Formal Definition via Fréchet Derivative

Before writing down the Jacobian matrix, it helps to understand the conceptual definition from which the matrix naturally emerges.

**Definition (Fréchet derivative).** Let $f: U \subseteq \mathbb{R}^n \to \mathbb{R}^m$ be defined on an open set $U$. We say $f$ is **differentiable at $\mathbf{x} \in U$** if there exists a linear map $L: \mathbb{R}^n \to \mathbb{R}^m$ such that:
$$\lim_{\|\boldsymbol{\delta}\| \to 0} \frac{\|f(\mathbf{x}+\boldsymbol{\delta}) - f(\mathbf{x}) - L\boldsymbol{\delta}\|}{\|\boldsymbol{\delta}\|} = 0$$

In other words: $L$ is the best linear approximation to $f$ near $\mathbf{x}$, in the sense that the error $f(\mathbf{x}+\boldsymbol{\delta}) - f(\mathbf{x}) - L\boldsymbol{\delta}$ is **sublinear** in $\|\boldsymbol{\delta}\|$ (it goes to zero faster than $\boldsymbol{\delta}$ itself).

**Theorem (Uniqueness).** If such $L$ exists, it is unique. Moreover, the matrix of $L$ in the standard bases is exactly the matrix of partial derivatives.

**Definition (Jacobian matrix).** The unique linear map $L$ from the Fréchet definition, written as an $m \times n$ matrix, is the **Jacobian** of $f$ at $\mathbf{x}$:
$$J_f(\mathbf{x}) = \begin{pmatrix} \dfrac{\partial f_1}{\partial x_1} & \dfrac{\partial f_1}{\partial x_2} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\[6pt] \dfrac{\partial f_2}{\partial x_1} & \dfrac{\partial f_2}{\partial x_2} & \cdots & \dfrac{\partial f_2}{\partial x_n} \\[6pt] \vdots & & \ddots & \vdots \\[6pt] \dfrac{\partial f_m}{\partial x_1} & \dfrac{\partial f_m}{\partial x_2} & \cdots & \dfrac{\partial f_m}{\partial x_n} \end{pmatrix} \in \mathbb{R}^{m \times n}$$

The $(i,j)$ entry is $[J_f]_{ij} = \partial f_i / \partial x_j$.

**Proof that the Fréchet derivative equals the Jacobian (sketch).** Applying the Fréchet condition with $\boldsymbol{\delta} = t\mathbf{e}_j$ (scalar multiple of $j$-th basis vector) and dividing by $t \to 0$:
$$[L]_{ij} = \lim_{t\to 0}\frac{f_i(\mathbf{x}+t\mathbf{e}_j) - f_i(\mathbf{x})}{t} = \frac{\partial f_i}{\partial x_j}$$

So the matrix entries of $L$ are precisely the partial derivatives. $\square$

**Important caveat.** The converse fails: the existence of all partial derivatives does **not** guarantee Fréchet differentiability. You additionally need the partial derivatives to be continuous, or equivalently that the limit in the Fréchet definition holds for all directions simultaneously (not just coordinate directions). The standard sufficient condition is: if all partial derivatives exist and are continuous at $\mathbf{x}$, then $f$ is Fréchet differentiable at $\mathbf{x}$ (this is the $C^1$ condition).

**Notation.** We write $J_f(\mathbf{x})$, $Df(\mathbf{x})$, $f'(\mathbf{x})$, or $\partial\mathbf{f}/\partial\mathbf{x}$ interchangeably. The Jacobian matrix depends on the point $\mathbf{x}$ — it is a matrix-valued function of the input.

**Row and column interpretations:**

```
TWO WAYS TO READ THE JACOBIAN
════════════════════════════════════════════════════════════════════════

  J_f ∈ ℝᵐˣⁿ  for  f: ℝⁿ → ℝᵐ

         x₁  x₂  x₃  ...  xₙ
       ┌──────────────────────┐
  f₁   │ row 1 = (∇f₁)ᵀ      │  ← transposed gradient of output 1
  f₂   │ row 2 = (∇f₂)ᵀ      │  ← transposed gradient of output 2
  f₃   │ row 3 = (∇f₃)ᵀ      │
  ⋮    │          ⋮           │
  fₘ   │ row m = (∇fₘ)ᵀ      │
       └──────────────────────┘
           ↑    ↑    ↑     ↑
          col  col  col   col  j = ∂f/∂xⱼ
           j                   sensitivity of ALL outputs to xⱼ

  Row reading:  "Given that I move in all n directions,
                 how does output i respond?"

  Column reading: "If I increase input j alone,
                   how do all m outputs respond?"

════════════════════════════════════════════════════════════════════════
```

### 2.2 Special Cases

The Jacobian unifies all the derivative objects from earlier chapters:

**Case 1: Scalar function $f: \mathbb{R}^n \to \mathbb{R}$ (i.e., $m=1$).**
The Jacobian is a $1 \times n$ row vector: $J_f = (\nabla f)^\top$. The gradient is the transposed Jacobian. This is why we write $\nabla f$ as a column vector and $J_f$ as a row vector for scalar functions — they are transposes of each other.

**Case 2: Linear map $f(\mathbf{x}) = A\mathbf{x}+\mathbf{b}$ with $A \in \mathbb{R}^{m \times n}$.**
$[J_f]_{ij} = \partial(A\mathbf{x}+\mathbf{b})_i/\partial x_j = A_{ij}$.
So $J_f = A$ everywhere — the Jacobian of a linear map is the matrix of that map, constant everywhere. This is the vector analogue of $(ax+b)' = a$.

**Case 3: Identity map $f(\mathbf{x}) = \mathbf{x}$.**
$[J_f]_{ij} = \partial x_i/\partial x_j = \delta_{ij}$, so $J_f = I_n$.

**Case 4: Scalar function of a scalar $f: \mathbb{R} \to \mathbb{R}$.**
$J_f = [f'(x)]$ — a $1 \times 1$ matrix, i.e., the ordinary derivative.

**Case 5: Vector function of a scalar $f: \mathbb{R} \to \mathbb{R}^m$** (a parametric curve).
$J_f = f'(t) \in \mathbb{R}^m$ — a column vector, the velocity vector. Shape: $m \times 1$.

| Map type | $n$ inputs | $m$ outputs | $J_f$ shape | Name |
|----------|-----------|------------|------------|------|
| $f:\mathbb{R}^n\to\mathbb{R}$ | $n$ | 1 | $1\times n$ | Transposed gradient |
| $f:\mathbb{R}^n\to\mathbb{R}^m$ | $n$ | $m$ | $m\times n$ | Jacobian matrix |
| $f:\mathbb{R}^n\to\mathbb{R}^n$ | $n$ | $n$ | $n\times n$ | Square Jacobian |
| $f:\mathbb{R}\to\mathbb{R}$ | 1 | 1 | $1\times 1$ | Ordinary derivative |
| $f(\mathbf{x})=A\mathbf{x}$ | $n$ | $m$ | $m\times n$ | $= A$ |

**Non-example: a function with partials but no Jacobian.**
$$f(x,y) = \begin{cases} \frac{xy}{\sqrt{x^2+y^2}} & (x,y)\neq(0,0) \\ 0 & (x,y)=(0,0) \end{cases}$$
Both partials $\partial f/\partial x$ and $\partial f/\partial y$ equal $0$ at the origin. But $f$ is not differentiable there: along the diagonal $y=x$, $f(t,t)/t = t/\sqrt{2t^2} \cdot t/t = 1/\sqrt{2} \neq 0 = \nabla f \cdot (1,1)/\sqrt{2}$. The Fréchet condition fails. This shows why we need continuity of partials, not just existence.

### 2.3 Computing Jacobians Systematically

**Procedure.** To compute $J_f(\mathbf{x})$ for $f: \mathbb{R}^n \to \mathbb{R}^m$:

**Row-by-row method (one output at a time):**
For each $i = 1,\ldots,m$: treat $f_i(\mathbf{x})$ as a scalar function and differentiate it with respect to each $x_j$. Place the resulting partial derivatives in row $i$.

**Column-by-column method (one input at a time):**
For each $j = 1,\ldots,n$: differentiate the entire vector $\mathbf{f}(\mathbf{x})$ with respect to $x_j$ (treating all other inputs as constants). Place the resulting $m$-vector in column $j$.

**Standard Jacobian table:**

| Function | Domain → Codomain | Jacobian |
|----------|------------------|---------|
| $f(\mathbf{x}) = A\mathbf{x}+\mathbf{b}$ | $\mathbb{R}^n\to\mathbb{R}^m$ | $A$ |
| $f(\mathbf{x}) = \sigma(\mathbf{x})$ (elementwise) | $\mathbb{R}^n\to\mathbb{R}^n$ | $\text{diag}(\sigma'(\mathbf{x}))$ |
| $f(\mathbf{x}) = \mathbf{a}^\top\mathbf{x}$ (scalar) | $\mathbb{R}^n\to\mathbb{R}$ | $\mathbf{a}^\top$ (row vector) |
| $f(\mathbf{x}) = \|\mathbf{x}\|^2$ (scalar) | $\mathbb{R}^n\to\mathbb{R}$ | $2\mathbf{x}^\top$ (row vector) |
| $f(\mathbf{x}) = \mathbf{x}/\|\mathbf{x}\|$ | $\mathbb{R}^n\to\mathbb{R}^n$ | $\frac{1}{\|\mathbf{x}\|}(I - \hat{\mathbf{x}}\hat{\mathbf{x}}^\top)$ |
| $f(\mathbf{x}) = \text{softmax}(\mathbf{x})$ | $\mathbb{R}^K\to\mathbb{R}^K$ | $\text{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top$ |
| $f(\mathbf{x}) = \text{ReLU}(\mathbf{x})$ | $\mathbb{R}^n\to\mathbb{R}^n$ | $\text{diag}(\mathbf{1}[\mathbf{x}>0])$ |
| $f(\mathbf{x}) = \mathbf{x}\mathbf{x}^\top$ (matrix output) | $\mathbb{R}^n\to\mathbb{R}^{n\times n}$ | Tensor: $[J_f]_{ij,k} = x_i\delta_{jk}+x_j\delta_{ik}$ |

### 2.4 Jacobian Determinant and Volume Scaling

When $m = n$ (square Jacobian), the **Jacobian determinant** $\det J_f(\mathbf{x})$ has a direct geometric meaning: it measures by what factor $f$ scales volumes near $\mathbf{x}$.

Formally: if $S \subset \mathbb{R}^n$ is a small region near $\mathbf{x}$, then $\text{vol}(f(S)) \approx |\det J_f(\mathbf{x})| \cdot \text{vol}(S)$.

- $|\det J_f| > 1$: $f$ expands local volume
- $|\det J_f| < 1$: $f$ contracts local volume
- $|\det J_f| = 0$: $f$ collapses volume (the map is locally degenerate, not invertible)
- $\det J_f < 0$: $f$ reverses orientation (a reflection-like component)

**Change of variables formula.** For integration:
$$\int_{f(U)} h(\mathbf{y})\,d\mathbf{y} = \int_U h(f(\mathbf{x}))\,|\det J_f(\mathbf{x})|\,d\mathbf{x}$$

This is the multivariate analogue of substitution: $\int h(f(x))|f'(x)|\,dx$.

**Application to normalising flows.** A normalising flow models the data density $p(\mathbf{x})$ by learning an invertible map $f: \mathbf{z} \mapsto \mathbf{x}$ from a simple base density $p_Z(\mathbf{z})$ (e.g., Gaussian). By change of variables:
$$\log p(\mathbf{x}) = \log p_Z(f^{-1}(\mathbf{x})) + \log|\det J_{f^{-1}}(\mathbf{x})|$$

Training requires computing $\log|\det J_f|$ for each batch element. Standard Gaussian elimination would cost $O(n^3)$ — prohibitive for $n = 10^5$. Flow architectures (RealNVP, GLOW) are designed so that $J_f$ is triangular or has Kronecker structure, making the determinant $O(n)$ to evaluate.

**Singular values and volume scaling.** More precisely: $|\det J_f| = \prod_i \sigma_i(J_f)$, where $\sigma_i$ are the singular values. The two-norm $\|J_f\|_2 = \sigma_{\max}$ tells us the maximum stretching factor in any direction. The condition number $\kappa(J_f) = \sigma_{\max}/\sigma_{\min}$ measures how anisotropic the stretching is.

### 2.5 Jacobian of a Composition (Preview)

The chain rule for Jacobians states: if $h = f \circ g$ (i.e., $h(\mathbf{x}) = f(g(\mathbf{x}))$) then:
$$J_h(\mathbf{x}) = J_f(g(\mathbf{x})) \cdot J_g(\mathbf{x})$$

The Jacobian of a composed function is the **product of Jacobians** — outer function first.

Shape check: if $g: \mathbb{R}^n \to \mathbb{R}^k$ and $f: \mathbb{R}^k \to \mathbb{R}^m$, then $J_g \in \mathbb{R}^{k\times n}$, $J_f \in \mathbb{R}^{m\times k}$, and $J_f \cdot J_g \in \mathbb{R}^{m\times n}$ — matches $J_h \in \mathbb{R}^{m\times n}$ for $h:\mathbb{R}^n\to\mathbb{R}^m$. ✓

This is the mathematical theorem that underlies backpropagation. A neural network $\mathcal{L} = \ell \circ f_L \circ \cdots \circ f_1$ has Jacobian:
$$J_\mathcal{L} = J_\ell \cdot J_{f_L} \cdot J_{f_{L-1}} \cdots J_{f_1}$$

Since $\mathcal{L}$ is scalar, $J_\mathcal{L} = (\nabla_\mathbf{x}\mathcal{L})^\top$ is a $1 \times n$ row vector — the transposed input gradient.

> **Full proof and derivation:** [§03 Chain Rule and Backpropagation](../03-Chain-Rule-and-Backpropagation/notes.md) derives this rigorously and applies it to the full backpropagation algorithm.

### 2.6 Worked Examples with Full Derivations

**Example 1: Componentwise function.**
$f: \mathbb{R}^3 \to \mathbb{R}^2$, $f(x,y,z) = (x^2 y + z,\; y^2 - xz)$.

Row 1 ($f_1 = x^2y + z$):
$$\frac{\partial f_1}{\partial x} = 2xy, \quad \frac{\partial f_1}{\partial y} = x^2, \quad \frac{\partial f_1}{\partial z} = 1$$

Row 2 ($f_2 = y^2 - xz$):
$$\frac{\partial f_2}{\partial x} = -z, \quad \frac{\partial f_2}{\partial y} = 2y, \quad \frac{\partial f_2}{\partial z} = -x$$

$$J_f(x,y,z) = \begin{pmatrix}2xy & x^2 & 1 \\ -z & 2y & -x\end{pmatrix}$$

At $(1, 2, 3)$: $J_f(1,2,3) = \begin{pmatrix}4 & 1 & 1 \\ -3 & 4 & -1\end{pmatrix}$

**Interpretation:** if we perturb the input by $\boldsymbol{\delta} = (0.1, 0, 0)$, the output changes by approximately $J_f \cdot (0.1, 0, 0)^\top = (0.4, -0.3)$.

**Example 2: Unit normalisation $f(\mathbf{x}) = \mathbf{x}/\|\mathbf{x}\|$.**

Let $r = \|\mathbf{x}\|$. Then $f_i = x_i / r$.

$$\frac{\partial f_i}{\partial x_j} = \frac{\partial}{\partial x_j}\frac{x_i}{r} = \frac{\delta_{ij} \cdot r - x_i \cdot \frac{x_j}{r}}{r^2} = \frac{\delta_{ij}}{r} - \frac{x_i x_j}{r^3} = \frac{1}{r}\left(\delta_{ij} - \frac{x_i x_j}{r^2}\right)$$

In matrix form, with $\hat{\mathbf{x}} = \mathbf{x}/r$:
$$J_f = \frac{1}{r}(I - \hat{\mathbf{x}}\hat{\mathbf{x}}^\top)$$

**Interpretation:** $J_f$ is a projection (scaled by $1/r$). The matrix $P = I - \hat{\mathbf{x}}\hat{\mathbf{x}}^\top$ projects onto the hyperplane perpendicular to $\hat{\mathbf{x}}$. So:
- Perturbations **perpendicular** to $\mathbf{x}$: these are preserved (scaled by $1/r$) — perturbing perpendicular to a unit vector changes its direction
- Perturbations **parallel** to $\mathbf{x}$ (i.e., scaling $\mathbf{x}$): these are killed by the projection — scaling a vector doesn't change its direction

**Non-example: Jacobian of a max operation.**
$f(\mathbf{x}) = \max_i x_i$ (scalar). This function is not differentiable when two or more components tie for the maximum. At a point where $x_{i^*}$ is the unique maximum:
$$J_f = \mathbf{e}_{i^*}^\top \quad (\text{row vector with 1 in position } i^*, \text{ zeros elsewhere})$$

At a tie, the Fréchet derivative does not exist. The Clarke subdifferential is the convex hull of $\{\mathbf{e}_i : x_i = \max_j x_j\}$.


---

## 3. Jacobians of Key ML Operations

### 3.1 Linear Layer

**Setup.** A linear (dense) layer computes $f(\mathbf{x}) = W\mathbf{x}+\mathbf{b}$ with $W \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$, $\mathbf{b} \in \mathbb{R}^{d_{\text{out}}}$.

**Jacobian w.r.t. input $\mathbf{x}$:**
$$[J_f]_{ij} = \frac{\partial (W\mathbf{x}+\mathbf{b})_i}{\partial x_j} = \frac{\partial \sum_k W_{ik}x_k + b_i}{\partial x_j} = W_{ij}$$

So $J_f(\mathbf{x}) = W$. The Jacobian of a linear layer is constant (same at every input point) and equals the weight matrix.

**Backpropagation through a linear layer.** Given upstream gradient $\boldsymbol{\delta}^{\text{out}} \in \mathbb{R}^{d_{\text{out}}}$ (i.e., $\partial\mathcal{L}/\partial(W\mathbf{x}+\mathbf{b})$), the downstream gradient is:
$$\boldsymbol{\delta}^{\text{in}} = J_f^\top\,\boldsymbol{\delta}^{\text{out}} = W^\top\boldsymbol{\delta}^{\text{out}} \in \mathbb{R}^{d_{\text{in}}}$$

This is the fundamental backprop equation for a linear layer: multiply by the transposed weight matrix.

**Gradient w.r.t. weight matrix $W$.** The function $g(W) = W\mathbf{x}+\mathbf{b}$ is also linear in $W$. Consider $\text{vec}(W) \in \mathbb{R}^{d_{\text{out}}d_{\text{in}}}$. The Jacobian of $g$ w.r.t. $\text{vec}(W)$ is:
$$J_g = \mathbf{x}^\top \otimes I_{d_{\text{out}}} \in \mathbb{R}^{d_{\text{out}} \times (d_{\text{out}}d_{\text{in}})}$$

where $\otimes$ is the Kronecker product. For scalar loss $\mathcal{L}$, the weight gradient is:
$$\frac{\partial\mathcal{L}}{\partial W} = \boldsymbol{\delta}^{\text{out}}\mathbf{x}^\top \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$$

This is a **rank-1 outer product** — for each training sample, the weight gradient has rank at most 1. The gradient accumulated over a mini-batch of size $B$ has rank at most $\min(B, d_{\text{out}}, d_{\text{in}})$.

**Implication for LoRA.** Since the weight gradient is the sum of rank-1 outer products $\boldsymbol{\delta}_i\mathbf{x}_i^\top$, the effective gradient subspace across an entire training run has rank much less than $\min(d_{\text{out}}, d_{\text{in}})$ — especially for large pre-trained models where the task-specific signal is concentrated in a low-dimensional subspace. LoRA (Hu et al., 2021) exploits this by parameterising the weight update as $\Delta W = AB$ with $A \in \mathbb{R}^{d_{\text{out}}\times r}$, $B \in \mathbb{R}^{r\times d_{\text{in}}}$, $r \ll \min(d_{\text{out}}, d_{\text{in}})$. Training only $A$ and $B$ reduces parameters from $d_{\text{out}}d_{\text{in}}$ to $r(d_{\text{out}}+d_{\text{in}})$.

### 3.2 Elementwise Activations

**Setup.** $f(\mathbf{x}) = \sigma(\mathbf{x})$ applied elementwise: $f_i(\mathbf{x}) = \sigma(x_i)$ for each $i$.

**Jacobian:**
$$[J_f]_{ij} = \frac{\partial f_i}{\partial x_j} = \frac{\partial \sigma(x_i)}{\partial x_j} = \sigma'(x_i)\cdot\mathbf{1}[i=j]$$

$$J_f(\mathbf{x}) = \text{diag}(\sigma'(x_1),\ldots,\sigma'(x_n)) = \text{diag}(\sigma'(\mathbf{x}))$$

The Jacobian is **diagonal** — output $i$ depends only on input $i$. Backpropagation through an elementwise activation is elementwise multiplication: $\boldsymbol{\delta}^{\text{in}} = \sigma'(\mathbf{x}) \odot \boldsymbol{\delta}^{\text{out}}$.

**Common activations in full detail:**

*ReLU: $\sigma(x) = \max(0, x)$.*
$\sigma'(x) = \mathbf{1}[x>0]$ (1 for positive inputs, 0 for negative, undefined at 0). The Jacobian is a binary diagonal matrix. Every negative-input neuron contributes zero gradient — the "dead neuron" problem. At scale, if a large fraction of neurons receive negative pre-activations, the effective gradient signal becomes sparse.

*Sigmoid: $\sigma(x) = 1/(1+e^{-x})$.*
$\sigma'(x) = \sigma(x)(1-\sigma(x))$. Maximum value $1/4$ (at $x=0$). For deep networks, the product of $L$ sigmoid derivatives is at most $(1/4)^L \to 0$ exponentially — the **vanishing gradient problem**. This is why sigmoid was replaced by ReLU in most architectures.

*tanh: $\sigma(x) = (e^x-e^{-x})/(e^x+e^{-x})$.*
$\sigma'(x) = 1-\tanh^2(x) \in [0,1]$. Maximum $1$ (at $x=0$). Better than sigmoid (maximum derivative is 1 not 1/4), but still vanishes for large $|x|$.

*GELU: $\sigma(x) = x\Phi(x)$ where $\Phi$ is the standard Gaussian CDF.*
$\sigma'(x) = \Phi(x) + x\phi(x)$ where $\phi$ is the Gaussian PDF. The derivative is approximately 1 for large positive $x$, 0 for large negative $x$, and smooth around 0. Default activation in BERT, GPT-2/3/4.

*SiLU (Swish): $\sigma(x) = x\cdot\text{sigmoid}(x)$.*
$\sigma'(x) = \text{sigmoid}(x) + x\cdot\text{sigmoid}(x)(1-\text{sigmoid}(x))$. Non-monotone (has a local minimum). Used in LLaMA, Mistral, Gemma.

**Gradient flow analysis.** For a depth-$L$ network with all linear layers having $\|W^{(l)}\|_2 = 1$ and elementwise activation Jacobians $D^{(l)} = \text{diag}(\sigma'(\mathbf{x}^{(l)}))$, the input-to-output Jacobian is:
$$J = D^{(L)}W^{(L)}\cdots D^{(1)}W^{(1)}$$

The spectral norm satisfies $\|J\|_2 \leq \prod_l \|D^{(l)}\|_2 \|W^{(l)}\|_2$. For sigmoid/tanh, $\|D^{(l)}\|_2 \leq 1/4$ or $1$; for large $L$, this product vanishes or explodes depending on the spectrum.

BatchNorm and LayerNorm are specifically designed to keep each factor $\|D^{(l)}W^{(l)}\|_2 \approx 1$ throughout training, preventing both explosion and vanishing.

### 3.3 Softmax Jacobian — Full Proof

**Setup.** The softmax function $\sigma: \mathbb{R}^K \to \Delta^{K-1}$ (the probability simplex) is defined by:
$$\sigma(\mathbf{z})_k = p_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}, \quad k = 1,\ldots,K$$

We want to compute all partial derivatives $\partial p_i / \partial z_j$ for all $i, j \in \{1,\ldots,K\}$.

**Full derivation.** Let $Z = \sum_{j=1}^K e^{z_j}$ (the partition function). Then $p_i = e^{z_i}/Z$.

**Case 1: $i = j$ (diagonal entries).** Use the quotient rule:
$$\frac{\partial p_i}{\partial z_i} = \frac{\partial}{\partial z_i}\frac{e^{z_i}}{Z} = \frac{e^{z_i} \cdot Z - e^{z_i} \cdot e^{z_i}}{Z^2} = \frac{e^{z_i}}{Z} - \left(\frac{e^{z_i}}{Z}\right)^2 = p_i - p_i^2 = p_i(1-p_i)$$

**Case 2: $i \neq j$ (off-diagonal entries).** Since $\partial e^{z_i}/\partial z_j = 0$ (different index):
$$\frac{\partial p_i}{\partial z_j} = \frac{0 \cdot Z - e^{z_i} \cdot \frac{\partial Z}{\partial z_j}}{Z^2} = \frac{-e^{z_i} \cdot e^{z_j}}{Z^2} = -\frac{e^{z_i}}{Z}\cdot\frac{e^{z_j}}{Z} = -p_i p_j$$

**Unifying both cases.** We can write:
$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

where $\delta_{ij}$ is the Kronecker delta. In matrix form:

$$\boxed{J_\sigma(\mathbf{z}) = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top}$$

**Verification.** Entry $(i,j)$ of $\text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ is:
- Diagonal ($i=j$): $p_i - p_i^2 = p_i(1-p_i)$ ✓
- Off-diagonal ($i\neq j$): $0 - p_ip_j = -p_ip_j$ ✓

**Properties of the softmax Jacobian:**

**Symmetry:** $[J_\sigma]_{ij} = p_i(\delta_{ij}-p_j)$ and $[J_\sigma]_{ji} = p_j(\delta_{ji}-p_i)$. Since $\delta_{ij}=\delta_{ji}$, both equal $-p_ip_j$ for $i\neq j$ and $p_i(1-p_i)$ on diagonal. So $J_\sigma = J_\sigma^\top$. (Softmax Jacobian is symmetric despite $f:\mathbb{R}^K\to\mathbb{R}^K$ being non-linear.)

**Singularity:** Consider $J_\sigma\mathbf{1}$. Entry $i$ is $\sum_j [J_\sigma]_{ij} = \sum_j p_i\delta_{ij} - p_i p_j = p_i - p_i\sum_j p_j = p_i - p_i = 0$. So $J_\sigma\mathbf{1} = \mathbf{0}$ — the all-ones vector is in the null space. This reflects that softmax is **shift-invariant**: $\sigma(\mathbf{z}+c\mathbf{1}) = \sigma(\mathbf{z})$ for any scalar $c$. Perturbations in the constant direction produce no change.

**Rank $K-1$:** One zero eigenvalue (direction $\mathbf{1}$), and $K-1$ positive eigenvalues. $J_\sigma$ is positive semidefinite.

**Positive semidefiniteness:** For any $\mathbf{v}$, $\mathbf{v}^\top J_\sigma \mathbf{v} = \mathbf{v}^\top(\text{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top)\mathbf{v} = \sum_i p_i v_i^2 - (\sum_i p_i v_i)^2$. By Jensen's inequality applied to the convex function $t^2$ with weights $p_i > 0$: $\sum_i p_i v_i^2 \geq (\sum_i p_i v_i)^2$. So $\mathbf{v}^\top J_\sigma\mathbf{v} \geq 0$, with equality iff all $v_i$ are equal (i.e., $\mathbf{v} \propto \mathbf{1}$). $\square$

**Cross-entropy gradient elegance.** When softmax feeds into cross-entropy loss $\mathcal{L} = -\log p_y$ for true class $y$:

Step 1: $\nabla_\mathbf{p}\mathcal{L} = -\mathbf{e}_y/p_y$ (gradient of log w.r.t. the relevant probability).

Step 2: Chain rule gives $\nabla_\mathbf{z}\mathcal{L} = J_\sigma^\top \nabla_\mathbf{p}\mathcal{L} = (\text{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top)\cdot(-\mathbf{e}_y/p_y)$.

Entry $k$: $(\text{diag}(\mathbf{p})-\mathbf{p}\mathbf{p}^\top)_{k,:}(-\mathbf{e}_y/p_y) = (p_k\delta_{ky} - p_kp_y) \cdot (-1/p_y) = -\delta_{ky} + p_k$.

So $[\nabla_\mathbf{z}\mathcal{L}]_k = p_k - \delta_{ky}$, i.e., $\nabla_\mathbf{z}\mathcal{L} = \mathbf{p} - \mathbf{e}_y$.

The combined gradient is simply **prediction minus one-hot label** — an elegant result. This is the gradient that flows backward through every transformer's output layer.

### 3.4 Layer Normalization Jacobian — Full Derivation

**Setup.** Layer normalisation (Ba et al., 2016) standardises each token embedding independently:
$$\hat{x}_i = \frac{x_i - \mu}{\sigma}, \quad \mu = \frac{1}{d}\sum_{j=1}^d x_j, \quad \sigma = \sqrt{\frac{1}{d}\sum_{j=1}^d (x_j-\mu)^2 + \varepsilon}$$

We derive the Jacobian $\partial\hat{x}_i/\partial x_k$ for the normalisation step (ignoring the learnable scale/shift $\boldsymbol{\gamma}, \boldsymbol{\beta}$).

**Step 1: Derivatives of $\mu$ and $\sigma$.**

$\partial\mu/\partial x_k = 1/d$ (all components).

$\partial\sigma/\partial x_k$: Using chain rule on $\sigma^2 = \frac{1}{d}\sum_j(x_j-\mu)^2 + \varepsilon$:
$$2\sigma\frac{\partial\sigma}{\partial x_k} = \frac{2}{d}\sum_j(x_j-\mu)\frac{\partial(x_j-\mu)}{\partial x_k} = \frac{2}{d}\sum_j(x_j-\mu)\left(\delta_{jk}-\frac{1}{d}\right)$$
$$= \frac{2}{d}\left[(x_k-\mu) - \frac{1}{d}\sum_j(x_j-\mu)\right] = \frac{2}{d}(x_k-\mu)$$

(since $\sum_j(x_j-\mu)=0$). Therefore $\partial\sigma/\partial x_k = \frac{x_k-\mu}{d\sigma} = \hat{x}_k/d\sigma$.

**Step 2: Derivative of $\hat{x}_i$.**

$$\frac{\partial\hat{x}_i}{\partial x_k} = \frac{\partial}{\partial x_k}\frac{x_i-\mu}{\sigma}$$

Using the quotient rule:
$$= \frac{(\delta_{ik} - 1/d)\cdot\sigma - (x_i-\mu)\cdot\partial\sigma/\partial x_k}{\sigma^2}$$
$$= \frac{\delta_{ik}-1/d}{\sigma} - \frac{(x_i-\mu)\hat{x}_k}{d\sigma^2}$$
$$= \frac{1}{\sigma}\left(\delta_{ik} - \frac{1}{d} - \frac{\hat{x}_i\hat{x}_k}{d}\right)$$

Wait — let's be careful: $(x_i-\mu)/\sigma = \hat{x}_i$ and $(x_k-\mu)/\sigma = \hat{x}_k$, so $(x_i-\mu)\partial\sigma/\partial x_k/\sigma^2 = \hat{x}_i\cdot\hat{x}_k\cdot(1/d)$.

In matrix form:
$$\boxed{J_{\text{LN}} = \frac{1}{\sigma}\left(I - \frac{1}{d}\mathbf{1}\mathbf{1}^\top - \frac{1}{d}\hat{\mathbf{x}}\hat{\mathbf{x}}^\top\right)}$$

Wait — let me recheck. Entry $(i,k)$:
$$[J_{\text{LN}}]_{ik} = \frac{1}{\sigma}\left(\delta_{ik} - \frac{1}{d} - \hat{x}_i\hat{x}_k \cdot \frac{1}{d} \cdot d\right)$$

Retracing: $(x_i-\mu)\cdot\partial\sigma/\partial x_k/\sigma^2 = \hat{x}_i\sigma \cdot \hat{x}_k/(d\sigma) / \sigma = \hat{x}_i\hat{x}_k/(d\sigma)$.

Hmm, $\partial\sigma/\partial x_k = \hat{x}_k/(d\sigma) \cdot \sigma = \hat{x}_k/d$. Wait: from Step 1, $\partial\sigma/\partial x_k = (x_k-\mu)/(d\sigma) = \hat{x}_k\sigma/(d\sigma) = \hat{x}_k/d$.

So:
$$\frac{\partial\hat{x}_i}{\partial x_k} = \frac{\delta_{ik}-1/d}{\sigma} - \frac{(x_i-\mu)}{{\sigma^2}}\cdot\frac{\hat{x}_k}{d} = \frac{\delta_{ik}-1/d}{\sigma} - \frac{\hat{x}_i\hat{x}_k}{d\sigma}$$

$$= \frac{1}{\sigma}\left(\delta_{ik} - \frac{1}{d} - \frac{\hat{x}_i\hat{x}_k}{d}\right)$$

Hmm that's $\delta_{ik} - \frac{1}{d}(1 + \hat{x}_i\hat{x}_k)$. This is slightly different from the usual formula. The standard result (Xu et al., 2019) is:

$$J_{\text{LN}} = \frac{1}{\sigma}\left(I - \frac{1}{d}\mathbf{1}\mathbf{1}^\top - \hat{\mathbf{x}}\hat{\mathbf{x}}^\top\right)$$

Let me redo without the erroneous $1/d$ in the last term. The issue is $\partial\sigma^2/\partial x_k$.

$\sigma^2 = \frac{1}{d}\sum_j(x_j-\mu)^2 + \varepsilon$. Then $\partial\sigma^2/\partial x_k = \frac{2}{d}(x_k-\mu)(1-1/d)\cdot d$... actually more carefully:

$\frac{\partial\sigma^2}{\partial x_k} = \frac{1}{d}\cdot 2(x_k-\mu)\cdot\frac{\partial(x_k-\mu)}{\partial x_k} + \frac{1}{d}\sum_{j\neq k}2(x_j-\mu)\cdot\frac{\partial(x_j-\mu)}{\partial x_k}$

$= \frac{2(x_k-\mu)}{d}(1-\frac{1}{d}) + \frac{1}{d}\sum_{j\neq k}2(x_j-\mu)(-\frac{1}{d})$

$= \frac{2(x_k-\mu)}{d} - \frac{2(x_k-\mu)}{d^2} - \frac{2}{d^2}\sum_{j\neq k}(x_j-\mu)$

$= \frac{2(x_k-\mu)}{d} - \frac{2}{d^2}\sum_j(x_j-\mu) = \frac{2(x_k-\mu)}{d}$

(since $\sum_j(x_j-\mu)=0$). So $\partial\sigma/\partial x_k = \frac{(x_k-\mu)}{d\sigma} = \frac{\hat{x}_k}{d}\cdot\sigma/\sigma = \frac{\hat{x}_k}{d}$. ✓

Then:
$$[J_{\text{LN}}]_{ik} = \frac{\delta_{ik}-1/d}{\sigma} - \frac{\hat{x}_i\cdot\hat{x}_k/d}{\sigma} = \frac{1}{\sigma}\left(\delta_{ik} - \frac{1}{d} - \frac{\hat{x}_i\hat{x}_k}{d}\right)$$

Note $\|\hat{\mathbf{x}}\|^2 = d$ (since $\hat{x}_j = (x_j-\mu)/\sigma$ and $\frac{1}{d}\sum_j\hat{x}_j^2=1$, so $\sum_j\hat{x}_j^2=d$). This means $\frac{1}{d}\hat{\mathbf{x}}\hat{\mathbf{x}}^\top$ is an outer product scaled by $1/d$. Writing:

$$J_{\text{LN}} = \frac{1}{\sigma}\left(I - \frac{1}{d}\mathbf{1}\mathbf{1}^\top - \frac{1}{d}\hat{\mathbf{x}}\hat{\mathbf{x}}^\top\right)$$

**Geometric interpretation.** The matrix $I - \frac{1}{d}\mathbf{1}\mathbf{1}^\top - \frac{1}{d}\hat{\mathbf{x}}\hat{\mathbf{x}}^\top$ is a projection that:

1. Removes the **mean direction** ($\mathbf{1}/\sqrt{d}$): $J_{\text{LN}}\mathbf{1} = \frac{1}{\sigma}\left(\mathbf{1} - \frac{d}{d}\mathbf{1} - \frac{\hat{\mathbf{x}}^\top\mathbf{1}}{d}\hat{\mathbf{x}}\right) = \frac{1}{\sigma}(\mathbf{0}) = \mathbf{0}$ (since $\sum_j\hat{x}_j = \sum_j(x_j-\mu)/\sigma = 0$). ✓

2. Removes the **normalised input direction** ($\hat{\mathbf{x}}/\sqrt{d}$): $J_{\text{LN}}\hat{\mathbf{x}} = \frac{1}{\sigma}\left(\hat{\mathbf{x}} - \frac{\mathbf{1}^\top\hat{\mathbf{x}}}{d}\mathbf{1} - \frac{\|\hat{\mathbf{x}}\|^2}{d}\hat{\mathbf{x}}\right) = \frac{1}{\sigma}(\hat{\mathbf{x}} - 0 - \hat{\mathbf{x}}) = \mathbf{0}$. ✓

**Null space and rank.** The null space is span$\{\mathbf{1}, \hat{\mathbf{x}}\}$ (2-dimensional if $\hat{\mathbf{x}}$ is not parallel to $\mathbf{1}$, which is true generically). So rank$(J_{\text{LN}}) = d-2$.

**Training implication.** Gradients cannot flow through the mean or norm directions. This is a desirable property: it prevents trivial changes (scaling all activations or shifting them) from affecting the gradient, making training more stable.

### 3.5 JVPs and VJPs — Why Reverse Mode Dominates

Given $f: \mathbb{R}^n \to \mathbb{R}^m$ with Jacobian $J_f \in \mathbb{R}^{m\times n}$, we often need to multiply by $J_f$ or $J_f^\top$ without materialising the full matrix.

**Jacobian-vector product (JVP, forward mode):**
$$\text{JVP}(\mathbf{v}) = J_f\,\mathbf{v} \in \mathbb{R}^m \quad \text{for tangent vector } \mathbf{v} \in \mathbb{R}^n$$

Answers: "If the input moves in direction $\mathbf{v}$, how does the output change?"

Computed by a **forward pass with dual numbers**: run the computation on input $(\mathbf{x}, \mathbf{v})$ where $\mathbf{v}$ is the "tangent" component. Cost: $O(n+m)$ — one forward pass plus tracking tangents. Does not require storing intermediate values.

**Vector-Jacobian product (VJP, reverse mode):**
$$\text{VJP}(\mathbf{u}) = J_f^\top\,\mathbf{u} \in \mathbb{R}^n \quad \text{for cotangent vector } \mathbf{u} \in \mathbb{R}^m$$

Answers: "Given a weighting $\mathbf{u}$ over outputs, what is each input's sensitivity?"

Computed by a **backward pass**: first run a forward pass (storing intermediates), then propagate $\mathbf{u}$ backward through the computation graph. Cost: $O(n+m)$ compute + $O(\text{forward pass memory})$ for stored activations.

**Why neural network training uses reverse mode (VJP).** The loss $\mathcal{L}$ is scalar ($m = 1$). We want all $n$ parameter gradients simultaneously. Compare:

- **Reverse mode (VJP):** One backward pass gives $(\nabla\mathcal{L})^\top = J_\mathcal{L}^\top \cdot 1 \in \mathbb{R}^n$ — all $n$ gradients at once. Total cost: $c_{\text{fwd}} + c_{\text{bwd}} \approx 2\text{-}3\times c_{\text{fwd}}$.

- **Forward mode (JVP):** To get gradient in direction $\mathbf{e}_j$, run one JVP with $\mathbf{v}=\mathbf{e}_j$, giving $\partial\mathcal{L}/\partial\theta_j$. Getting all $n$ gradients requires $n$ JVPs. Total cost: $n \times c_{\text{fwd}}$.

For $n = 10^8$ (typical LLM), reverse mode is $10^8\times$ cheaper. This is why PyTorch, JAX, and TensorFlow all implement reverse-mode AD (backpropagation) as their default.

**When forward mode is preferable.** If $n \ll m$ (few inputs, many outputs), forward mode is cheaper: $n$ JVPs vs $m$ VJPs. Applications: uncertainty propagation through a small number of uncertain inputs, sensitivity analysis, directional derivative computation.

```
COST COMPARISON: JVP vs VJP
════════════════════════════════════════════════════════════════════════

  f: ℝⁿ → ℝᵐ,  J_f ∈ ℝᵐˣⁿ

  Full Jacobian via JVPs:   n passes of JVP(eⱼ),  cost = n · c_fwd
  Full Jacobian via VJPs:   m passes of VJP(eᵢ),  cost = m · c_bwd

  For ML (m=1, scalar loss):
    Forward mode: n passes  →  cost n · c_fwd     (EXPENSIVE)
    Reverse mode: 1 pass    →  cost     c_bwd     (CHEAP)
    Speedup: n / (c_bwd/c_fwd) ≈ n/3 ≈ 3·10⁷  for GPT-3

  For normalising flows (need full Jacobian for log|det J|):
    Forward mode: n passes  →  expensive for large n
    Special architectures (triangular J): O(n) determinant

════════════════════════════════════════════════════════════════════════
```

**Full Jacobian recovery.** To get $J_f \in \mathbb{R}^{m\times n}$:
- Apply JVP with $\mathbf{v}=\mathbf{e}_j$ for $j=1,\ldots,n$: fills columns of $J_f$. Cost: $n\cdot c_{\text{fwd}}$.
- Apply VJP with $\mathbf{u}=\mathbf{e}_i$ for $i=1,\ldots,m$: fills rows of $J_f$. Cost: $m\cdot c_{\text{bwd}}$.

Choose the cheaper option: if $n < m$, use JVPs; if $m < n$, use VJPs.

### 3.6 Jacobian Condition Number and Training Stability

**Definition.** The condition number of the Jacobian $J_f \in \mathbb{R}^{m\times n}$ is:
$$\kappa(J_f) = \frac{\sigma_{\max}(J_f)}{\sigma_{\min}(J_f)}$$

where $\sigma_{\max}$ and $\sigma_{\min}$ are the largest and smallest singular values. (For $m \neq n$, $\sigma_{\min}$ refers to the smallest nonzero singular value.)

**Geometric meaning.** The singular value decomposition $J_f = U\Sigma V^\top$ shows that $J_f$ rotates (via $V^\top$) then scales (via $\Sigma$) then rotates again (via $U$). The input directions $\mathbf{v}_j$ (columns of $V$) are mapped to output directions $\sigma_j \mathbf{u}_j$ (scaled columns of $U$). $\kappa = \sigma_{\max}/\sigma_{\min}$ measures how anisotropic this scaling is:

- $\kappa = 1$: all input directions are scaled equally — perfectly isotropic
- $\kappa \gg 1$: some directions are amplified by $\sigma_{\max}$, others suppressed to near zero by $\sigma_{\min}$ — highly anisotropic

**Why $\kappa$ matters for training.** In backpropagation, the gradient flows backward as $\boldsymbol{\delta}^{\text{in}} = J_f^\top \boldsymbol{\delta}^{\text{out}}$. The singular values of $J_f^\top$ are the same as those of $J_f$. So:

- Gradient components in the $\sigma_{\max}$ direction are amplified by $\sigma_{\max}$
- Gradient components in the $\sigma_{\min}$ direction are suppressed by $\sigma_{\min}$

Over $L$ layers, the overall Jacobian $J = J_L \cdots J_1$ has singular values bounded by $\prod_l \sigma_{\max}(J_l)$ (upper) and $\prod_l \sigma_{\min}(J_l)$ (lower). If each $\sigma_{\max} > 1+\varepsilon$: exponential **gradient explosion**. If each $\sigma_{\min} < 1-\varepsilon$: exponential **gradient vanishing**.

**Initialisation strategies.** Xavier (Glorot) initialisation sets $W_{ij} \sim \mathcal{U}(-\sqrt{6/(n_{\text{in}}+n_{\text{out}})}, +\sqrt{6/(n_{\text{in}}+n_{\text{out}})})$, which initialises $\mathbb{E}[\sigma_{\max}(W)] \approx 1$ and $\kappa(W) \approx \sqrt{n_{\text{out}}/n_{\text{in}}}$ (near 1 for square layers). He initialisation uses $W_{ij} \sim \mathcal{N}(0, 2/n_{\text{in}})$, designed for ReLU which zeroes half the neurons.

**Normalisation layers.** BatchNorm and LayerNorm act as implicit preconditioning: they rescale the activations so that the effective Jacobian of each normalised layer has $\sigma_{\max} \approx \sigma_{\min} \approx 1$. This keeps the product of singular values near 1 regardless of depth.


---

## 4. The Hessian Matrix

### 4.1 Formal Definition

**Definition (Hessian).** Let $f: U \subseteq \mathbb{R}^n \to \mathbb{R}$ be twice differentiable at $\mathbf{x} \in U$. The **Hessian matrix** of $f$ at $\mathbf{x}$ is the $n \times n$ matrix of all second-order partial derivatives:

$$H_f(\mathbf{x}) = \nabla^2 f(\mathbf{x}) = \begin{pmatrix} \dfrac{\partial^2 f}{\partial x_1^2} & \dfrac{\partial^2 f}{\partial x_1\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_1\partial x_n} \\[8pt] \dfrac{\partial^2 f}{\partial x_2\partial x_1} & \dfrac{\partial^2 f}{\partial x_2^2} & \cdots & \dfrac{\partial^2 f}{\partial x_2\partial x_n} \\[8pt] \vdots & & \ddots & \vdots \\[8pt] \dfrac{\partial^2 f}{\partial x_n\partial x_1} & \dfrac{\partial^2 f}{\partial x_n\partial x_2} & \cdots & \dfrac{\partial^2 f}{\partial x_n^2} \end{pmatrix}$$

The $(i,j)$ entry is $[H_f]_{ij} = \partial^2 f / \partial x_i \partial x_j$.

**Three equivalent ways to think about the Hessian:**

1. **Matrix of second partial derivatives.** Each entry is differentiate $f$ twice: first with respect to $x_j$, then with respect to $x_i$.

2. **Jacobian of the gradient.** Since $\nabla f: \mathbb{R}^n \to \mathbb{R}^n$ is a vector-valued function, it has a Jacobian. That Jacobian is the Hessian: $H_f(\mathbf{x}) = J_{\nabla f}(\mathbf{x})$, because $[J_{\nabla f}]_{ij} = \partial[\nabla f]_i/\partial x_j = \partial^2 f/\partial x_j\partial x_i = [H_f]_{ij}$ (when $f \in C^2$, the order of differentiation does not matter by Clairaut's theorem).

3. **Curvature operator.** For any unit direction $\mathbf{u}$, the scalar curvature of $f$ in that direction is $\kappa(\mathbf{u}) = \mathbf{u}^\top H_f(\mathbf{x})\,\mathbf{u}$. This is the second derivative of $g(t) = f(\mathbf{x}+t\mathbf{u})$ at $t=0$.

**Notation.** We write $H_f(\mathbf{x})$, $\nabla^2 f(\mathbf{x})$, $D^2 f(\mathbf{x})$, or $\text{Hess}(f)(\mathbf{x})$ interchangeably.

```
HESSIAN STRUCTURE  (n=3 example)
════════════════════════════════════════════════════════════════════════

         x₁           x₂           x₃
       ┌────────────────────────────────────┐
  x₁   │  ∂²f/∂x₁²    ∂²f/∂x₁∂x₂  ∂²f/∂x₁∂x₃ │
  x₂   │  ∂²f/∂x₂∂x₁  ∂²f/∂x₂²    ∂²f/∂x₂∂x₃ │
  x₃   │  ∂²f/∂x₃∂x₁  ∂²f/∂x₃∂x₂  ∂²f/∂x₃²   │
       └────────────────────────────────────┘

  Diagonal: pure second derivatives (curvature along each axis)
  Off-diagonal [i,j]: how ∂f/∂xᵢ changes as xⱼ moves
                      (interaction / cross-curvature)

  When f ∈ C²: H_f is symmetric (Clairaut's theorem)
  i.e., ∂²f/∂xᵢ∂xⱼ = ∂²f/∂xⱼ∂xᵢ for all i,j

════════════════════════════════════════════════════════════════════════
```

### 4.2 Clairaut's Theorem — Complete Proof and Failure Case

**Theorem (Clairaut–Schwarz, 1740/1873).** If $f: U \subseteq \mathbb{R}^n \to \mathbb{R}$ has continuous second-order partial derivatives at $\mathbf{x} \in U$ (i.e., $f \in C^2$ near $\mathbf{x}$), then:
$$\frac{\partial^2 f}{\partial x_i \partial x_j}(\mathbf{x}) = \frac{\partial^2 f}{\partial x_j \partial x_i}(\mathbf{x}) \quad \text{for all } i \neq j$$

Equivalently: $H_f(\mathbf{x})$ is a **symmetric matrix** whenever $f \in C^2$.

**Complete proof (two-variable case).** It suffices to prove the result for $f: \mathbb{R}^2 \to \mathbb{R}$ with variables $(x, y)$. Define the mixed difference quotient:
$$\Delta(h, k) = f(x+h, y+k) - f(x+h, y) - f(x, y+k) + f(x, y)$$

**Step 1: Express $\Delta$ via $\partial^2 f/\partial x \partial y$.** Define $\phi(t) = f(t, y+k) - f(t, y)$. Then $\Delta(h,k) = \phi(x+h) - \phi(x)$. By the mean value theorem (MVT) applied to $\phi$ on $[x, x+h]$:
$$\Delta(h,k) = h\,\phi'(x + \theta_1 h) = h\left[\frac{\partial f}{\partial t}(x+\theta_1 h, y+k) - \frac{\partial f}{\partial t}(x+\theta_1 h, y)\right]$$
for some $\theta_1 \in (0,1)$. Now apply MVT to $\psi(s) = \frac{\partial f}{\partial x}(x+\theta_1 h, s)$ on $[y, y+k]$:
$$\Delta(h,k) = h\,k\,\frac{\partial^2 f}{\partial y\partial x}(x+\theta_1 h,\; y+\theta_2 k) \quad \text{for some } \theta_2 \in (0,1)$$

**Step 2: Express $\Delta$ via $\partial^2 f/\partial y \partial x$.** Now define $\psi(s) = f(x+h, s) - f(x, s)$. Then $\Delta(h,k) = \psi(y+k) - \psi(y)$. By MVT on $\psi$ on $[y, y+k]$:
$$\Delta(h,k) = k\,\psi'(y+\theta_3 k) = k\left[\frac{\partial f}{\partial s}(x+h, y+\theta_3 k) - \frac{\partial f}{\partial s}(x, y+\theta_3 k)\right]$$
for some $\theta_3 \in (0,1)$. Apply MVT to $s \mapsto \frac{\partial f}{\partial y}(s, y+\theta_3 k)$ on $[x, x+h]$:
$$\Delta(h,k) = h\,k\,\frac{\partial^2 f}{\partial x\partial y}(x+\theta_4 h,\; y+\theta_3 k) \quad \text{for some } \theta_4 \in (0,1)$$

**Step 3: Take the limit.** From Steps 1 and 2:
$$\frac{\partial^2 f}{\partial y\partial x}(x+\theta_1 h, y+\theta_2 k) = \frac{\Delta(h,k)}{hk} = \frac{\partial^2 f}{\partial x\partial y}(x+\theta_4 h, y+\theta_3 k)$$

As $h, k \to 0$, both sides approach $(x, y)$ since $\theta_i \in (0,1)$. By **continuity** of the second partial derivatives:
$$\frac{\partial^2 f}{\partial y\partial x}(x, y) = \frac{\partial^2 f}{\partial x\partial y}(x, y) \quad \square$$

**Why continuity is essential.** The proof uses continuity only in the last step — to take the limit. Without continuity, the two sides approach the same point but need not converge to the same value.

**Concrete counterexample (Peano, 1880).** Define:
$$f(x,y) = \begin{cases} \dfrac{xy(x^2-y^2)}{x^2+y^2} & (x,y)\neq(0,0) \\ 0 & (x,y)=(0,0) \end{cases}$$

*Step 1: Compute $\partial f/\partial x$ at $(0,0)$.* By the limit definition:
$$\frac{\partial f}{\partial x}(0,0) = \lim_{h\to 0}\frac{f(h,0)-f(0,0)}{h} = \lim_{h\to 0}\frac{0}{h} = 0$$

Similarly $\partial f/\partial y(0,0) = 0$.

*Step 2: Compute $\partial f/\partial x$ at $(0,y)$ for $y\neq 0$.*
$f(x,y) = xy(x^2-y^2)/(x^2+y^2)$. By the quotient rule at $x=0$:
$$\frac{\partial f}{\partial x}(0,y) = y\cdot\frac{(x^2-y^2)(x^2+y^2) - xy\cdot 2x\cdot(x^2+y^2)/(x^2+y^2) + \ldots}{(\ldots)}\bigg|_{x=0} = y\cdot\frac{-y^2}{y^2} = -y$$

*Step 3: Compute the mixed partial $\partial^2 f/\partial y\partial x$ at $(0,0)$.*
$$\frac{\partial^2 f}{\partial y\partial x}(0,0) = \lim_{k\to 0}\frac{\frac{\partial f}{\partial x}(0,k) - \frac{\partial f}{\partial x}(0,0)}{k} = \lim_{k\to 0}\frac{-k - 0}{k} = -1$$

*Step 4: By symmetry of the roles of $x$ and $y$ (with a sign flip from $x^2-y^2$):*
$$\frac{\partial^2 f}{\partial x\partial y}(0,0) = +1$$

So $\partial^2 f/\partial y\partial x = -1 \neq 1 = \partial^2 f/\partial x\partial y$ at the origin, even though both first partial derivatives exist everywhere. The second partial derivatives exist at $(0,0)$ but are **not continuous** there — exactly what Clairaut requires.

**Practical implication.** For any function built from compositions of smooth operations ($\exp$, $\log$, $+$, $\times$, $\sin$, $\cos$, polynomial), all partial derivatives of all orders are continuous. Neural network losses with smooth activations (GELU, sigmoid, tanh, softmax) therefore always have symmetric Hessians. For ReLU networks, the Hessian is symmetric almost everywhere (where the Hessian exists) but may not exist on the boundaries $\{x_i = 0\}$.

### 4.3 Computing the Hessian

**Method 1: Direct differentiation (symbolic).** Compute $\nabla f$ first, then differentiate each component again.

*Cost:* $O(n^2)$ partial derivatives. For $n = 10^8$ parameters: $10^{16}$ entries — the full Hessian matrix is utterly infeasible to compute or store for any modern neural network.

**Method 2: Numerical Hessian via finite differences.** Use the second-order central difference formula:
$$[H_f]_{ij} \approx \frac{f(\mathbf{x}+h\mathbf{e}_i+h\mathbf{e}_j) - f(\mathbf{x}+h\mathbf{e}_i-h\mathbf{e}_j) - f(\mathbf{x}-h\mathbf{e}_i+h\mathbf{e}_j) + f(\mathbf{x}-h\mathbf{e}_i-h\mathbf{e}_j)}{4h^2}$$

*Error:* $O(h^2)$ truncation error. Round-off error $O(\varepsilon_{\text{mach}}/h^2)$. Optimal $h \approx \varepsilon_{\text{mach}}^{1/4} \approx 10^{-4}$ balances both.

*Cost:* $O(n^2)$ function evaluations. Still infeasible for large $n$.

**Method 3: Via automatic differentiation.**
- PyTorch: `torch.autograd.functional.hessian(f, x)` — materialises full Hessian via $n$ backward passes
- JAX: `jax.hessian(f)(x)` — uses forward-over-reverse AD (one forward pass per output component of $\nabla f$)

Both require $O(n)$ AD passes, each costing $O(n)$ — total $O(n^2)$ time. Infeasible for large networks.

**Method 4: Hessian-vector products (HVPs).** Do not materialise the Hessian; instead compute $H_f\mathbf{v}$ for a given $\mathbf{v}$ in $O(n)$ time. This is the practical approach for all large-scale applications. See §8.1.

### 4.4 Hessian as Jacobian of the Gradient

The identity $H_f = J_{\nabla f}$ is more than a definition — it is a computational recipe.

**Proof.** The gradient $\nabla f: \mathbb{R}^n \to \mathbb{R}^n$ maps $\mathbf{x}$ to the vector $(\partial f/\partial x_1, \ldots, \partial f/\partial x_n)^\top$. The $(i,j)$ entry of its Jacobian $J_{\nabla f}$ is:
$$[J_{\nabla f}]_{ij} = \frac{\partial [\nabla f]_i}{\partial x_j} = \frac{\partial}{\partial x_j}\frac{\partial f}{\partial x_i} = \frac{\partial^2 f}{\partial x_j \partial x_i} = [H_f]_{ij} \quad \text{(when } f\in C^2\text{)} \quad \square$$

**Directional second derivative.** Applying this to the directional derivative of $\nabla f$ in direction $\mathbf{v}$:
$$J_{\nabla f}\,\mathbf{v} = H_f\,\mathbf{v} = \nabla_\mathbf{x}[\nabla f(\mathbf{x})\cdot\mathbf{v}]$$

This identity is the key to efficient HVP computation (§8.1): to compute $H_f\mathbf{v}$, differentiate the scalar function $\mathbf{x}\mapsto\nabla f(\mathbf{x})\cdot\mathbf{v}$ with respect to $\mathbf{x}$. One forward pass computes $\nabla f(\mathbf{x})\cdot\mathbf{v}$ (a JVP); one backward pass gives $\nabla_\mathbf{x}[\nabla f\cdot\mathbf{v}] = H_f\mathbf{v}$.

**Curvature Rayleigh quotient.** For a unit vector $\mathbf{u}$:
$$g(t) = f(\mathbf{x}+t\mathbf{u}) \implies g''(0) = \mathbf{u}^\top H_f(\mathbf{x})\,\mathbf{u}$$

This is the second derivative of $f$ along the line through $\mathbf{x}$ in direction $\mathbf{u}$:
- $g''(0) > 0$: $f$ is concave up (curves upward) in direction $\mathbf{u}$
- $g''(0) < 0$: $f$ is concave down (curves downward) in direction $\mathbf{u}$
- $g''(0) = 0$: $f$ is locally flat in direction $\mathbf{u}$

The maximum curvature is $\lambda_{\max}(H_f)$, achieved in the direction of the top eigenvector $\mathbf{q}_1$. The minimum curvature is $\lambda_{\min}(H_f)$, in the direction of $\mathbf{q}_n$.

### 4.5 Worked Examples with Full Derivations

**Example 1: Quadratic form.**
$f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top A\mathbf{x} + \mathbf{b}^\top\mathbf{x} + c$ where $A$ is symmetric.

*Gradient:* $\nabla f = A\mathbf{x} + \mathbf{b}$.

*Hessian:* $H_f = J_{A\mathbf{x}+\mathbf{b}} = A$ (Jacobian of a linear function is the coefficient matrix).

The Hessian is constant — it does not depend on $\mathbf{x}$. A quadratic function has the same curvature everywhere.

*Verification:* $[H_f]_{ij} = \partial^2(\frac{1}{2}\sum_{k,l}A_{kl}x_kx_l)/\partial x_i\partial x_j = \frac{1}{2}(A_{ij}+A_{ji}) = A_{ij}$ (since $A$ is symmetric). ✓

**Example 2: Mean squared error loss.**
$f(\mathbf{w}) = \frac{1}{m}\|X\mathbf{w}-\mathbf{y}\|^2 = \frac{1}{m}(X\mathbf{w}-\mathbf{y})^\top(X\mathbf{w}-\mathbf{y})$ with $X\in\mathbb{R}^{m\times n}$.

*Gradient:* $\nabla f = \frac{2}{m}X^\top(X\mathbf{w}-\mathbf{y})$.

*Hessian:* $H_f = J\left[\frac{2}{m}X^\top(X\mathbf{w}-\mathbf{y})\right] = \frac{2}{m}X^\top X$.

$H_f$ is positive semidefinite (PSD) since $\mathbf{v}^\top\frac{2}{m}X^\top X\mathbf{v} = \frac{2}{m}\|X\mathbf{v}\|^2 \geq 0$ for all $\mathbf{v}$. Therefore MSE is a **globally convex** function — any local minimum is the global minimum. The eigenvalues of $H_f = \frac{2}{m}X^\top X$ are $\frac{2}{m}\sigma_i(X)^2$ where $\sigma_i(X)$ are singular values of $X$.

**Example 3: Log-sum-exp (softmax partition function).**
$f(\mathbf{x}) = \log\sum_{k=1}^K e^{x_k}$.

*Gradient:* $[\nabla f]_k = e^{x_k}/\sum_j e^{x_j} = p_k$ (the softmax probabilities).

*Hessian:* $H_f = J_\mathbf{p}(\mathbf{x})$ — the Jacobian of the softmax function. From §3.3:
$$H_f = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$$

This is PSD (shown in §3.3), so log-sum-exp is convex. Geometrically: the level sets of $\log\sum_k e^{x_k} = c$ are smooth convex curves in $\mathbb{R}^K$.

**Example 4: Cross-entropy loss for logistic regression.**
$f(\mathbf{w}) = -y\log\sigma(\mathbf{w}^\top\mathbf{x}) - (1-y)\log(1-\sigma(\mathbf{w}^\top\mathbf{x}))$ for a single training example $(\mathbf{x}, y)$.

*Gradient:* $\nabla f = (\sigma(\mathbf{w}^\top\mathbf{x})-y)\mathbf{x} = (\hat{p}-y)\mathbf{x}$.

*Hessian:* Using the chain rule:
$$H_f = \frac{d\hat{p}}{d(\mathbf{w}^\top\mathbf{x})}\cdot\mathbf{x}\mathbf{x}^\top = \sigma(\mathbf{w}^\top\mathbf{x})(1-\sigma(\mathbf{w}^\top\mathbf{x}))\cdot\mathbf{x}\mathbf{x}^\top = \hat{p}(1-\hat{p})\cdot\mathbf{x}\mathbf{x}^\top$$

This is a **rank-1** matrix (outer product of $\mathbf{x}$ with itself, scaled by $\hat{p}(1-\hat{p}) \in [0, 1/4]$). Its single nonzero eigenvalue is $\hat{p}(1-\hat{p})\|\mathbf{x}\|^2$ and the corresponding eigenvector is $\hat{\mathbf{x}} = \mathbf{x}/\|\mathbf{x}\|$. All other eigenvectors are in $\mathbf{x}^\perp$ with eigenvalue 0 — the loss is flat in all directions perpendicular to $\mathbf{x}$.

Over a batch of $m$ examples: $H_f = \sum_{i=1}^m \hat{p}_i(1-\hat{p}_i)\mathbf{x}_i\mathbf{x}_i^\top$ — sum of rank-1 PSD matrices, so PSD with rank at most $\min(m, n)$.

**Non-example: Hessian of a ReLU network.**
A single ReLU unit $f(x) = \max(0,x)$:
- For $x > 0$: $f'(x) = 1$, $f''(x) = 0$
- For $x < 0$: $f'(x) = 0$, $f''(x) = 0$
- At $x = 0$: $f'$ is discontinuous, $f''$ does not exist classically

A deep ReLU network is **piecewise linear** — it is linear on each region of parameter space where all activation patterns are fixed. Therefore its Hessian is **zero almost everywhere**. Second-order curvature only appears at the activation boundaries (measure zero).

This is why Gauss-Newton and K-FAC (which avoid computing the Hessian through nonlinearities) are preferred over exact Newton for ReLU networks. The exact Hessian is nearly zero and gives no useful curvature information.


---

## 5. Second-Order Taylor Expansion

The Taylor expansion is the bridge between the Hessian's definition and its computational use. It explains why the Hessian governs convergence rates, why eigenvalues bound step sizes, and why second-order optimisers are faster than first-order ones.

### 5.1 The Multivariate Taylor Theorem

**Theorem (Taylor's theorem in $\mathbb{R}^n$, exact form).** Let $f: \mathbb{R}^n \to \mathbb{R}$ be $C^3$ in a neighbourhood of $\mathbf{x}_0$. Then for any $\boldsymbol{\delta} \in \mathbb{R}^n$:

$$f(\mathbf{x}_0 + \boldsymbol{\delta}) = f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top\boldsymbol{\delta} + \frac{1}{2}\boldsymbol{\delta}^\top H_f(\mathbf{x}_0)\boldsymbol{\delta} + R_3(\boldsymbol{\delta})$$

where the remainder satisfies $|R_3(\boldsymbol{\delta})| \leq C\|\boldsymbol{\delta}\|^3$ for some constant $C > 0$.

**Proof.** Define $\phi: [0,1] \to \mathbb{R}$ by $\phi(t) = f(\mathbf{x}_0 + t\boldsymbol{\delta})$. This reduces the multivariate theorem to the single-variable one. Compute derivatives by chain rule:

$$\phi'(t) = \nabla f(\mathbf{x}_0 + t\boldsymbol{\delta})^\top \boldsymbol{\delta} = \sum_i \frac{\partial f}{\partial x_i}\bigg|_{\mathbf{x}_0+t\boldsymbol{\delta}} \delta_i$$

$$\phi''(t) = \boldsymbol{\delta}^\top H_f(\mathbf{x}_0 + t\boldsymbol{\delta})\boldsymbol{\delta} = \sum_{i,j} \frac{\partial^2 f}{\partial x_i \partial x_j}\bigg|_{\mathbf{x}_0+t\boldsymbol{\delta}} \delta_i\delta_j$$

Apply the single-variable Taylor theorem with integral remainder:

$$\phi(1) = \phi(0) + \phi'(0) + \frac{1}{2}\phi''(0) + \int_0^1 \frac{(1-t)^2}{2}\phi'''(t)\,dt$$

Substituting: $\phi(0) = f(\mathbf{x}_0)$, $\phi'(0) = \nabla f(\mathbf{x}_0)^\top\boldsymbol{\delta}$, $\phi''(0) = \boldsymbol{\delta}^\top H_f(\mathbf{x}_0)\boldsymbol{\delta}$, and $\phi(1) = f(\mathbf{x}_0 + \boldsymbol{\delta})$. The integral remainder is $R_3$; bounding $|\phi'''(t)| \leq M\|\boldsymbol{\delta}\|^3$ for some $M$ gives $|R_3| \leq \frac{M}{6}\|\boldsymbol{\delta}\|^3$. $\square$

**The quadratic approximation.** Dropping $R_3$ yields the second-order approximation:

$$\boxed{f(\mathbf{x}_0 + \boldsymbol{\delta}) \approx f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top\boldsymbol{\delta} + \frac{1}{2}\boldsymbol{\delta}^\top H_f(\mathbf{x}_0)\boldsymbol{\delta}}$$

This is a **quadratic function in $\boldsymbol{\delta}$** — an elliptic paraboloid (if $H_f \succ 0$), a hyperboloid (if $H_f$ is indefinite), or a cylinder/flat direction (if $H_f$ is singular). The entire geometry of the function near $\mathbf{x}_0$ is encoded in this approximation.

### 5.2 Geometric Picture: Level Sets and Curvature

To see the geometry, write the quadratic approximation as a function of $\boldsymbol{\delta}$:

$$q(\boldsymbol{\delta}) = \frac{1}{2}\boldsymbol{\delta}^\top H_f \boldsymbol{\delta} + \mathbf{g}^\top\boldsymbol{\delta} + c$$

where $\mathbf{g} = \nabla f(\mathbf{x}_0)$ and $c = f(\mathbf{x}_0)$. Complete the square: if $H_f \succ 0$,

$$q(\boldsymbol{\delta}) = \frac{1}{2}(\boldsymbol{\delta} + H_f^{-1}\mathbf{g})^\top H_f (\boldsymbol{\delta} + H_f^{-1}\mathbf{g}) + c - \frac{1}{2}\mathbf{g}^\top H_f^{-1}\mathbf{g}$$

The minimum is at $\boldsymbol{\delta}^* = -H_f^{-1}\mathbf{g}$ — this is the **Newton step**. The level sets $\{q = \text{const}\}$ are ellipses (in 2D) or ellipsoids (in $n$D) aligned with the eigenvectors of $H_f$.

```
QUADRATIC LANDSCAPE GEOMETRY
════════════════════════════════════════════════════════════════════════

  Isotropic Hessian (H = λI):       Anisotropic Hessian:
  Circular level sets                Elliptical level sets

       · · · · · ·                      ·   ·   ·   ·   ·
     ·     (∘)   ·                   ·       ·       ·
   ·   ·    ↑    · ·                ·  ·   (∘)  ·    ·  ·
  ·   · ·←─●─→· · ·               ·  ·    ↑    ·     ·  ·
   ·   ·    ↓    · ·               ·  · ←──●──→ ·    ·  ·
     ·      ·    ·                  ·  ·    ↓    ·    ·
       · · · · · ·                     ·   ·   ·   ·

  GD step = Newton step             GD step ≠ Newton step
  (gradient ∝ steepest descent)     (GD "zigzags"; Newton goes direct)

  Eigenvalues: λ₁ = λ₂ = λ           Eigenvalues: λ₁ >> λ₂
  Condition number: κ = 1             Condition number: κ = λ₁/λ₂ >> 1

════════════════════════════════════════════════════════════════════════
```

**Principal curvatures.** The Hessian at a critical point $\mathbf{x}^*$ (where $\nabla f = 0$) has eigendecomposition $H_f = Q\Lambda Q^\top$ where $Q = [\mathbf{q}_1|\cdots|\mathbf{q}_n]$ is orthogonal and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$. In the basis $\{\mathbf{q}_i\}$, the function is a sum of independent quadratics:

$$f(\mathbf{x}^* + \boldsymbol{\delta}) \approx f(\mathbf{x}^*) + \frac{1}{2}\sum_{i=1}^n \lambda_i ({\mathbf{q}_i^\top\boldsymbol{\delta}})^2$$

Each $\lambda_i$ is the **curvature** along direction $\mathbf{q}_i$:
- $\lambda_i > 0$: $f$ increases parabolically in direction $\mathbf{q}_i$ → local min in that direction
- $\lambda_i < 0$: $f$ decreases parabolically → saddle direction
- $\lambda_i = 0$: $f$ is flat → cannot determine from second-order information

**Second derivative test.** At a critical point $\nabla f(\mathbf{x}^*) = 0$:
- $H_f \succ 0$ (all $\lambda_i > 0$) $\Rightarrow$ **strict local minimum**
- $H_f \prec 0$ (all $\lambda_i < 0$) $\Rightarrow$ **strict local maximum**
- $H_f$ indefinite (mixed signs) $\Rightarrow$ **saddle point**
- $H_f \succeq 0$ but singular $\Rightarrow$ **inconclusive** (need higher-order terms)

### 5.3 Hessian Eigendecomposition and Convergence Rates

The eigenvalues of the Hessian at a minimum $\mathbf{x}^*$ directly control how fast gradient descent converges nearby.

**Local convergence analysis.** Near $\mathbf{x}^*$, linearise the gradient iteration. The gradient update $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\nabla f(\mathbf{x}_t)$ gives, by Taylor:

$$\nabla f(\mathbf{x}_t) = \nabla f(\mathbf{x}^*) + H_f(\mathbf{x}^*)(\mathbf{x}_t - \mathbf{x}^*) + O(\|\mathbf{x}_t - \mathbf{x}^*\|^2) = H_f(\mathbf{x}^*)(\mathbf{x}_t - \mathbf{x}^*) + O(\|\cdot\|^2)$$

Let $\boldsymbol{\epsilon}_t = \mathbf{x}_t - \mathbf{x}^*$. Then:

$$\boldsymbol{\epsilon}_{t+1} = \boldsymbol{\epsilon}_t - \eta H_f \boldsymbol{\epsilon}_t + O(\|\boldsymbol{\epsilon}_t\|^2) = (I - \eta H_f)\boldsymbol{\epsilon}_t + O(\|\boldsymbol{\epsilon}_t\|^2)$$

Ignoring the higher-order term: $\boldsymbol{\epsilon}_{t+1} \approx (I - \eta H_f)\boldsymbol{\epsilon}_t$. In the eigenbasis:

$$(\boldsymbol{\epsilon}_{t+1})_i = (1 - \eta\lambda_i)(\boldsymbol{\epsilon}_t)_i$$

After $T$ steps: $(\boldsymbol{\epsilon}_T)_i = (1 - \eta\lambda_i)^T (\boldsymbol{\epsilon}_0)_i$.

Convergence requires $|1 - \eta\lambda_i| < 1$ for all $i$, i.e., $0 < \eta < 2/\lambda_{\max}$. The optimal step size (minimising the worst-case contraction ratio) is:

$$\eta^* = \frac{2}{\lambda_{\min} + \lambda_{\max}}$$

giving contraction ratio $\rho = \frac{\kappa - 1}{\kappa + 1}$ where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the **condition number**. The number of steps to reduce error by factor $\varepsilon$ scales as $O(\kappa \log(1/\varepsilon))$.

**The condition number catastrophe.** For a poorly conditioned Hessian ($\kappa \gg 1$), GD makes tiny progress per step. A concrete example: for $f(x,y) = \frac{1}{2}(x^2 + 100y^2)$, the Hessian is $\text{diag}(1, 100)$, $\kappa = 100$. With $\eta = 2/(1+100) \approx 0.02$:
- Along $y$: converges at rate $|1 - 0.02\cdot 100| = 0.98$ — very slow
- Along $x$: converges at rate $|1 - 0.02\cdot 1| = 0.98$ — limited by the slow direction

Newton's method avoids this by multiplying by $H_f^{-1}$, giving $\boldsymbol{\epsilon}_{t+1} = O(\|\boldsymbol{\epsilon}_t\|^2)$ — **quadratic convergence**, independent of conditioning.

### 5.4 Gradient Descent vs Newton's Method

**Gradient descent update:**
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\nabla f(\mathbf{x}_t)$$

Move in the steepest descent direction, step size proportional to $\eta$.

**Newton's update:**
$$\mathbf{x}_{t+1} = \mathbf{x}_t - H_f(\mathbf{x}_t)^{-1}\nabla f(\mathbf{x}_t)$$

Move to the minimum of the quadratic approximation. Derivation: minimise $q(\boldsymbol{\delta}) = f(\mathbf{x}_t) + \mathbf{g}^\top\boldsymbol{\delta} + \frac{1}{2}\boldsymbol{\delta}^\top H\boldsymbol{\delta}$ over $\boldsymbol{\delta}$:

$$\frac{\partial q}{\partial \boldsymbol{\delta}} = \mathbf{g} + H\boldsymbol{\delta} = 0 \implies \boldsymbol{\delta}^* = -H^{-1}\mathbf{g}$$

| Property | Gradient Descent | Newton's Method |
|----------|-----------------|-----------------|
| Convergence rate (near $\mathbf{x}^*$) | Linear: $\|\boldsymbol{\epsilon}_{t+1}\| \leq \rho\|\boldsymbol{\epsilon}_t\|$ | Quadratic: $\|\boldsymbol{\epsilon}_{t+1}\| = O(\|\boldsymbol{\epsilon}_t\|^2)$ |
| Dependence on $\kappa$ | $O(\kappa\log(1/\varepsilon))$ steps | $O(\log\log(1/\varepsilon))$ steps |
| Cost per step | $O(n)$ (gradient) | $O(n^3)$ (Hessian inversion) |
| Memory | $O(n)$ | $O(n^2)$ (Hessian storage) |
| Works when $H \not\succ 0$? | Yes (gradient always valid) | No ($H^{-1}$ may not exist or give ascent) |
| Suitable for deep learning? | Yes (SGD, Adam) | No ($n \sim 10^9$) |

**For AI:** Deep networks have $n \sim 10^6$–$10^{10}$ parameters. Exact Newton is completely infeasible. This is why the entire field of second-order optimisation for deep learning focuses on **approximating** $H_f^{-1}$ cheaply:
- **Diagonal approximations**: Adam uses $\sqrt{v_t}$ (RMS of past gradients) as a diagonal Hessian estimate
- **Kronecker factorizations**: K-FAC approximates each layer's Fisher as a Kronecker product
- **Low-rank approximations**: KFAC-Reduce, EKFAC
- **Hessian-vector products**: Lanczos algorithm to find the top eigenvectors


---

## 6. Hessian Definiteness and Flat vs Sharp Minima

The definiteness of the Hessian at a minimum has consequences far beyond the second derivative test. It determines generalisation, robustness to perturbations, and the choice of optimiser. This section examines each case in depth.

### 6.1 The Positive Definite Case: Strongly Convex Minima

If $H_f(\mathbf{x}^*) \succ 0$, the minimum is **strongly convex locally**. All curvatures are positive; the function rises quadratically in every direction. Key properties:

- **Unique minimum**: in the neighbourhood, $\mathbf{x}^*$ is the only critical point
- **Bounded gradient**: $\|\nabla f(\mathbf{x})\| \geq \lambda_{\min}\|\mathbf{x} - \mathbf{x}^*\|$ for $\mathbf{x}$ near $\mathbf{x}^*$
- **Gradient descent converges linearly** (as derived in §5.3)
- **Robust to small perturbations**: a perturbation $\boldsymbol{\epsilon}$ of size $\delta$ changes $f$ by at most $\frac{1}{2}\lambda_{\max}\delta^2$

**Algebraic criterion (Sylvester's criterion).** $H \succ 0$ if and only if all leading principal minors are positive:
$$\det(H_{1:k,1:k}) > 0 \quad \text{for } k = 1, 2, \ldots, n$$

In 2D: $H = \begin{pmatrix}a & b \\ b & c\end{pmatrix} \succ 0 \iff a > 0 \text{ and } ac - b^2 > 0$.

**Eigenvalue criterion.** $H \succ 0 \iff \lambda_{\min}(H) > 0$. The smallest eigenvalue measures how "tight" the well is in the flattest direction.

### 6.2 The Positive Semidefinite Case: Degenerate Minima

If $H_f(\mathbf{x}^*) \succeq 0$ with $\lambda_{\min} = 0$, the quadratic approximation is flat in the null space $\ker(H_f)$. The second-order test is inconclusive.

**What can happen along flat directions:**
- The function might be constant in that direction (a continuum of minima — "flat valley")
- The function might dip lower (a deeper minimum exists nearby)
- The function might rise (a saddle, with flat approach)

**Example: Over-parameterised networks.** Consider fitting $n$ data points with a model having $p \gg n$ parameters. The minimum loss is achieved on a **manifold** of parameters (not a single point). Along this manifold, the loss is constant $= 0$, so $\nabla f = 0$ and $H_f$ has at least $p - n$ zero eigenvalues. The loss landscape near a minimum is extremely flat in the "model space" directions.

**Example: Batch normalisation symmetry.** If layer $l$ has BatchNorm, then scaling $W^{(l)} \to \alpha W^{(l)}$ and $W^{(l+1)} \to W^{(l+1)}/\alpha$ leaves the network function unchanged. This **scale symmetry** creates a zero eigenvalue direction in the Hessian of any loss function.

### 6.3 The Indefinite Case: Saddle Points

If $H_f(\mathbf{x}^*)$ has both positive and negative eigenvalues, $\mathbf{x}^*$ is a **saddle point**. The function rises in some directions and falls in others.

**Strict saddles.** A saddle is *strict* if $\lambda_{\min}(H_f) < 0$. Gradient descent perturbed by noise escapes strict saddles — the noisy gradient has a component in the negative-curvature direction, pulling the iterate away. This is the basis for the claim that SGD finds local minima rather than saddles in practice.

**Prevalence in deep learning.** Dauphin et al. (2014) showed empirically that the loss landscape of deep networks is dominated by saddle points rather than local minima. The ratio of saddle points to local minima grows exponentially with depth. Goodfellow et al. confirmed that most "local minima" found by SGD are actually saddle points or nearly flat regions with very small negative curvature.

**Index of a saddle.** The **Morse index** is the number of negative eigenvalues. A local minimum has index 0; a typical saddle in $\mathbb{R}^n$ has index $k$ for $1 \leq k \leq n$.

### 6.4 Flat vs Sharp Minima and Generalisation

One of the most important practical consequences of Hessian analysis is the **flat minima hypothesis** for neural network generalisation.

**Sharp minimum:** $\lambda_{\max}(H_f) \gg 1$. The loss landscape around $\mathbf{x}^*$ is narrow and steep. A small perturbation of the weights significantly increases the training loss. Empirically associated with **poor generalisation** — the test distribution's slight differences from training push the model off the sharp peak.

**Flat minimum:** $\lambda_{\max}(H_f)$ is small. The loss landscape is broad and shallow. A perturbation of the weights leaves training loss roughly unchanged. Empirically associated with **better generalisation** — the model is robust to distribution shift.

**PAC-Bayes bound.** Hochreiter & Schmidhuber (1997) and Keskar et al. (2017) formalised this: a generalisation bound based on the minimum description length of a model. Flatter minima require fewer bits to specify (the volume of the basin in parameter space is larger), so PAC-Bayes bounds on generalisation error are tighter.

**The SAM algorithm (Sharpness-Aware Minimisation, Foret et al. 2021).** SAM explicitly searches for flat minima:

$$\min_{\mathbf{w}} f_{\text{SAM}}(\mathbf{w}) := \max_{\|\boldsymbol{\epsilon}\| \leq \rho} f(\mathbf{w} + \boldsymbol{\epsilon})$$

The inner maximisation finds the worst-case perturbation within radius $\rho$. By Taylor:
$$\max_{\|\boldsymbol{\epsilon}\| \leq \rho} f(\mathbf{w} + \boldsymbol{\epsilon}) \approx f(\mathbf{w}) + \rho\|\nabla f(\mathbf{w})\|$$

The SAM gradient is $\nabla_{\mathbf{w}} f(\hat{\mathbf{w}})$ where $\hat{\mathbf{w}} = \mathbf{w} + \rho\frac{\nabla f(\mathbf{w})}{\|\nabla f(\mathbf{w})\|}$ (ascent step to the worst-case point). This requires **two gradient computations per step** (one for the ascent, one for the final gradient), doubling cost.

**SAM in practice.** SAM and its variants (ASAM, mSAM, SAM-ON) consistently improve generalisation by 1–3% on ImageNet and CIFAR benchmarks. The improvement is attributed to SAM finding flatter minima, though the exact mechanism is debated (some argue SAM also acts as a regulariser).

**Sharpness measures.** Several quantities measure flatness:
- **Trace of Hessian**: $\text{tr}(H_f) = \sum_i \lambda_i$ — total curvature (sum of all curvatures)
- **Frobenius norm**: $\|H_f\|_F = \sqrt{\sum_{i,j} h_{ij}^2}$ — total variation
- **Largest eigenvalue**: $\lambda_{\max}(H_f)$ — worst-case curvature direction
- **Volume of basin**: $(\det H_f)^{-1/2} \propto \prod_i \lambda_i^{-1/2}$ — Laplace approximation normaliser

### 6.5 Hessian Spectrum in Practice

Ghorbani et al. (2019) performed careful empirical Hessian eigenspectrum analysis of large neural networks, finding a consistent structure:

```
HESSIAN SPECTRUM OF DEEP NETWORKS
════════════════════════════════════════════════════════════════════════

  Eigenvalue density (schematic):

       │
 dens. │████
       │██████
       │████████
       │██████████
       │████████████
       │██████████████
       │█████████████████
  ─────┼──────────────────────────────── λ
       0   small              λ_max (outliers)

  • Bulk: dense cluster of small eigenvalues near 0
    (most directions have very small curvature)

  • Outliers: a few large eigenvalues (often O(10)–O(1000))
    (a handful of directions have sharp curvature)

  • The ratio λ_max / λ_bulk >> 1 (condition number very large)

  • After convergence: more eigenvalues cluster near 0
    (the model has found flat directions = the "flat manifold" of solutions)

════════════════════════════════════════════════════════════════════════
```

**Implications:**
- **Learning rate must be** $\eta < 2/\lambda_{\max}$ for stability — but $\lambda_{\max}$ can be very large ($\sim 10^3$ for ResNets)
- **Learning rate warmup** is necessary because early in training $\lambda_{\max}$ is large and unstable
- **Gradient clipping** effectively caps the "step size" in the sharp directions
- **Adam/AdaGrad normalise** by the gradient magnitude, implicitly compensating for large curvature in some directions


---

## 7. Newton's Method and Second-Order Optimisation

### 7.1 Newton's Method: Full Analysis

We derived the Newton step $\boldsymbol{\delta}_N = -H_f^{-1}\nabla f$ in §5.4. Here we analyse it thoroughly.

**Local quadratic convergence.** Let $\mathbf{x}^*$ be a local minimum with $H_f(\mathbf{x}^*) \succ 0$. For $\mathbf{x}_0$ sufficiently close to $\mathbf{x}^*$:

$$\|\mathbf{x}_{t+1} - \mathbf{x}^*\| \leq \frac{M}{2\lambda_{\min}} \|\mathbf{x}_t - \mathbf{x}^*\|^2$$

where $M = \sup_{\mathbf{x}} \|D^3f(\mathbf{x})\|$ (Lipschitz constant of the Hessian). This is **quadratic convergence**: each iteration squares the error. From $\varepsilon_0 = 0.1$: $\varepsilon_1 \approx 0.01$, $\varepsilon_2 \approx 10^{-4}$, $\varepsilon_3 \approx 10^{-8}$. Only $\sim 30$ steps to reach machine precision!

**Proof sketch.** By Taylor at $\mathbf{x}^*$: $\nabla f(\mathbf{x}_t) = H_f(\mathbf{x}^*)(\mathbf{x}_t - \mathbf{x}^*) + O(\|\mathbf{x}_t - \mathbf{x}^*\|^2)$. The Newton step gives:
$$\mathbf{x}_{t+1} = \mathbf{x}_t - H_f(\mathbf{x}_t)^{-1}\nabla f(\mathbf{x}_t)$$

Substituting and using $H_f(\mathbf{x}_t) = H_f(\mathbf{x}^*) + O(\|\mathbf{x}_t - \mathbf{x}^*\|)$, the linear error term cancels exactly, leaving only $O(\|\mathbf{x}_t - \mathbf{x}^*\|^2)$.

**Problems with vanilla Newton:**
1. **$H_f$ not PD**: If $H_f$ is indefinite, $-H_f^{-1}\mathbf{g}$ may be an ascent direction. At saddle points Newton steps towards the saddle, not away.
2. **Step too large**: Even near a minimum, the step may overshoot if the quadratic approximation is inaccurate far from the current point.
3. **Hessian cost**: Computing $H_f \in \mathbb{R}^{n\times n}$ costs $O(n^2)$ memory and $O(n^2)$ gradient computations. Inverting costs $O(n^3)$.

### 7.2 Damped Newton and Quasi-Newton Methods

**Damped Newton's method:**
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha_t H_f(\mathbf{x}_t)^{-1}\nabla f(\mathbf{x}_t)$$

where $\alpha_t$ is chosen by **line search** to ensure sufficient decrease (Armijo condition):
$$f(\mathbf{x}_t + \alpha_t\boldsymbol{\delta}_N) \leq f(\mathbf{x}_t) + c\alpha_t\nabla f(\mathbf{x}_t)^\top\boldsymbol{\delta}_N$$

for $c \in (0,1)$ (typically $c = 10^{-4}$). This guarantees global descent while preserving superlinear convergence near $\mathbf{x}^*$.

**Modified Cholesky.** To handle non-PD Hessians, add a diagonal: $\tilde{H} = H_f + \mu I$ with $\mu \geq 0$ chosen so $\tilde{H} \succ 0$. This is the **Levenberg-Marquardt** modification. The step $-\tilde{H}^{-1}\mathbf{g}$ interpolates between Newton ($\mu = 0$) and gradient descent ($\mu \to \infty$, step $\to -\mathbf{g}/\mu$).

**L-BFGS (Limited-memory BFGS).** The most widely used quasi-Newton method in practice (used for GPT-2 fine-tuning in some settings, L-BFGS-B for convex ML). Rather than computing $H_f^{-1}$ exactly, L-BFGS maintains an implicit approximation from the last $m$ gradient differences ($m = 5$ to $20$):

$$\mathbf{s}_k = \mathbf{x}_{k+1} - \mathbf{x}_k, \quad \mathbf{y}_k = \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_k)$$

The BFGS update enforces $H_{k+1}\mathbf{s}_k = \mathbf{y}_k$ (secant condition). L-BFGS applies the implicit product $H_k^{-1}\mathbf{g}$ via a two-loop recursion in $O(mn)$ time, avoiding the $O(n^2)$ Hessian matrix.

### 7.3 The Gauss-Newton Method

For **least-squares problems** $f(\mathbf{w}) = \frac{1}{2}\|\mathbf{r}(\mathbf{w})\|^2$ where $\mathbf{r}: \mathbb{R}^n \to \mathbb{R}^m$ is the residual vector, the exact Hessian is:

$$H_f = J_{\mathbf{r}}^\top J_{\mathbf{r}} + \sum_{i=1}^m r_i(\mathbf{w}) H_{r_i}$$

The second term involves Hessians of each residual, which are expensive and may be negative-definite. The **Gauss-Newton approximation** drops this term:

$$\hat{H}_{\text{GN}} = J_{\mathbf{r}}^\top J_{\mathbf{r}} \succeq 0$$

This is always PSD (so the step is always a descent direction) and costs only $O(mn^2)$ — compute $J_{\mathbf{r}}$ once, then form the $n\times n$ product.

The Gauss-Newton step: $\mathbf{w}_{t+1} = \mathbf{w}_t - (J^\top J)^{-1} J^\top \mathbf{r}(\mathbf{w}_t)$.

**For neural networks:** The **Fisher Information Matrix** (FIM) is the natural Gauss-Newton matrix:

$$F = \mathbb{E}_{\mathbf{x},y\sim p_{\text{data}}}[(\nabla_{\mathbf{w}}\log p(y|\mathbf{x};\mathbf{w}))(\nabla_{\mathbf{w}}\log p(y|\mathbf{x};\mathbf{w}))^\top] = J_{p}^\top J_{p}$$

where $J_p$ is the Jacobian of the log-probabilities. This equals the Gauss-Newton matrix for log-likelihood maximisation, and is always PSD. Natural gradient descent uses $F^{-1}\mathbf{g}$ as the update direction.

### 7.4 Kronecker-Factored Approximate Curvature (K-FAC)

K-FAC (Martens & Grosse 2015) is the most successful practical second-order method for deep learning. It exploits the structure of neural network layers to cheaply approximate the FIM.

**The key insight.** For a linear layer $\mathbf{y} = W\mathbf{a}$ with input activations $\mathbf{a} \in \mathbb{R}^{n_\text{in}}$ and pre-activations $\mathbf{y} \in \mathbb{R}^{n_\text{out}}$, the gradient of the loss with respect to $W$ is:

$$\nabla_W \mathcal{L} = \mathbf{g}_y \mathbf{a}^\top$$

where $\mathbf{g}_y = \frac{\partial \mathcal{L}}{\partial \mathbf{y}}$ (backpropagated gradient). Therefore:

$$\text{vec}(\nabla_W \mathcal{L}) = \mathbf{a} \otimes \mathbf{g}_y$$

The exact FIM block for layer $l$ is:

$$F_l = \mathbb{E}[(\mathbf{a} \otimes \mathbf{g}_y)(\mathbf{a} \otimes \mathbf{g}_y)^\top] = \mathbb{E}[\mathbf{a}\mathbf{a}^\top \otimes \mathbf{g}_y\mathbf{g}_y^\top]$$

K-FAC **approximates** this by assuming independence of $\mathbf{a}$ and $\mathbf{g}_y$:

$$\tilde{F}_l = \mathbb{E}[\mathbf{a}\mathbf{a}^\top] \otimes \mathbb{E}[\mathbf{g}_y\mathbf{g}_y^\top] = A_l \otimes G_l$$

where $A_l = \mathbb{E}[\mathbf{a}\mathbf{a}^\top] \in \mathbb{R}^{n_\text{in}\times n_\text{in}}$ and $G_l = \mathbb{E}[\mathbf{g}_y\mathbf{g}_y^\top] \in \mathbb{R}^{n_\text{out}\times n_\text{out}}$.

**The Kronecker structure enables cheap inversion:**
$$(A_l \otimes G_l)^{-1} = A_l^{-1} \otimes G_l^{-1}$$

So inverting the large $n_\text{in}n_\text{out} \times n_\text{in}n_\text{out}$ matrix costs only $O(n_\text{in}^3 + n_\text{out}^3)$ — inverting two small matrices. The natural gradient step is:

$$\Delta W_l = -G_l^{-1}(\nabla_{W_l}\mathcal{L}) A_l^{-1}$$

This can be applied in $O(n_\text{in}^2 + n_\text{out}^2)$ per layer (far less than $O(n_\text{in}^3 n_\text{out}^3)$ for the exact FIM). K-FAC achieves **10-30× speedup** over SGD in terms of wall-clock time to convergence on large models, while maintaining good generalisation.


---

## 8. Hessian-Vector Products and Spectral Methods

Computing the full $n\times n$ Hessian is infeasible for large networks. But many algorithms only need **Hessian-vector products** (HVPs) $H_f\mathbf{v}$ — and these can be computed in $O(n)$ time.

### 8.1 The $O(n)$ HVP Identity

**Theorem.** For $f: \mathbb{R}^n \to \mathbb{R}$ differentiable and any $\mathbf{v} \in \mathbb{R}^n$:

$$H_f\mathbf{v} = \nabla_{\mathbf{x}}[\nabla_{\mathbf{x}} f(\mathbf{x}) \cdot \mathbf{v}]$$

**Proof.** The $(i,j)$-entry of $H_f$ is $\frac{\partial^2 f}{\partial x_i \partial x_j}$. The dot product $\nabla f \cdot \mathbf{v} = \sum_j \frac{\partial f}{\partial x_j} v_j$. Taking the gradient:

$$\frac{\partial}{\partial x_i}(\nabla f \cdot \mathbf{v}) = \sum_j \frac{\partial^2 f}{\partial x_i \partial x_j} v_j = (H_f\mathbf{v})_i$$

So $\nabla[\nabla f \cdot \mathbf{v}] = H_f\mathbf{v}$. $\square$

**Implementation.** In any automatic differentiation framework (PyTorch/JAX):

```python
# v is a fixed vector; x is the point at which to evaluate H(x)v
loss = f(x)
grad = torch.autograd.grad(loss, x, create_graph=True)[0]    # ∇f(x), O(n)
hvp = torch.autograd.grad(grad @ v, x)[0]                    # H(x)v, O(n)
```

The first `grad` call is standard backprop. The second differentiates the scalar $\nabla f \cdot \mathbf{v}$ — also a standard backprop, costing $O(n)$ just like the first. Total: **2 passes, $O(n)$ each**, giving $H_f\mathbf{v}$ without ever forming the $n\times n$ matrix.

### 8.2 Power Iteration and Dominant Eigenvalue

Using HVPs, we can find the **largest eigenvalue** $\lambda_{\max}(H_f)$ via power iteration:

1. Initialise $\mathbf{v}_0 \sim \mathcal{N}(0, I)$, normalise: $\mathbf{v}_0 \leftarrow \mathbf{v}_0/\|\mathbf{v}_0\|$
2. Repeat: $\tilde{\mathbf{v}}_{t+1} = H_f\mathbf{v}_t$ (one HVP), $\mathbf{v}_{t+1} = \tilde{\mathbf{v}}_{t+1}/\|\tilde{\mathbf{v}}_{t+1}\|$
3. Estimate: $\lambda_{\max} \approx \mathbf{v}_t^\top H_f\mathbf{v}_t$ (Rayleigh quotient, another HVP)

Each iteration costs 1–2 HVPs = $O(n)$. Convergence rate: $|\lambda_{\max}/\lambda_2|^t$ (geometric in gap ratio). For well-separated spectra, converges in $\sim 10$–50 iterations.

**For AI:** $\lambda_{\max}(H_f)$ determines the maximum stable learning rate: $\eta_{\max} = 2/\lambda_{\max}$. Computing this via 20 power iterations costs 20 HVPs = 40 backward passes — affordable even for billion-parameter models. The **ProgressiveShrink** and **Catapult** phenomena in transformer training are explained by $\lambda_{\max}$ dynamics: if the learning rate exceeds $2/\lambda_{\max}$, loss explodes; after explosion, a sharp minimum may be replaced by a flatter one (the "edge of stability" phenomenon, Cohen et al. 2022).

### 8.3 Lanczos Algorithm and Full Spectrum

The **Lanczos algorithm** extracts the full eigenspectrum using a sequence of HVPs. Starting from a random unit vector $\mathbf{q}_1$:

1. Build an orthonormal basis $\{q_1, q_2, \ldots, q_k\}$ (Krylov subspace) by repeated HVPs:
   - $\tilde{\mathbf{q}}_{j+1} = H_f\mathbf{q}_j - \alpha_j\mathbf{q}_j - \beta_{j-1}\mathbf{q}_{j-1}$
   - $\mathbf{q}_{j+1} = \tilde{\mathbf{q}}_{j+1}/\|\tilde{\mathbf{q}}_{j+1}\|$
   where $\alpha_j = \mathbf{q}_j^\top H_f\mathbf{q}_j$, $\beta_j = \|\tilde{\mathbf{q}}_{j+1}\|$

2. After $k$ steps, $H_f$ is approximated in the Krylov basis as a tridiagonal matrix $T_k$:
   $$T_k = \begin{pmatrix}\alpha_1 & \beta_1 & & \\ \beta_1 & \alpha_2 & \beta_2 & \\ & \ddots & \ddots & \ddots\\ & & \beta_{k-1} & \alpha_k\end{pmatrix}$$

3. Eigenvalues of $T_k$ approximate eigenvalues of $H_f$. The extreme eigenvalues converge fastest (in $O(\sqrt{\kappa})$ iterations rather than $O(\kappa)$ for power iteration).

**Density estimation.** Using the **stochastic Lanczos quadrature** method (Ghorbani et al. 2019), one can estimate the full spectral density $\rho(\lambda) = \frac{1}{n}\sum_i \delta(\lambda - \lambda_i)$ using $\sim 100$ random starting vectors and $\sim 100$ Lanczos steps each. This gives a smooth density plot revealing the bulk-outlier structure discussed in §6.5.

### 8.4 Applications of HVPs in Deep Learning

**1. Computing $\lambda_{\max}$ for learning rate bounds.**
Cohen et al. (2022) showed that SGD with fixed learning rate $\eta$ converges at the "edge of stability" where $\lambda_{\max}(H_f) \approx 2/\eta$. Measuring $\lambda_{\max}$ via power iteration explains why training stabilises at a specific sharpness level.

**2. Influence functions.**
For convex models, the influence of removing training point $z_i$ on prediction is:
$$\mathcal{I}(z_i, z_{\text{test}}) = -\nabla_{\mathbf{w}}L(z_{\text{test}})^\top H_f^{-1} \nabla_{\mathbf{w}}L(z_i)$$

Computing $H_f^{-1}\mathbf{v}$ requires solving a linear system $H_f\mathbf{x} = \mathbf{v}$, which can be done iteratively (conjugate gradient, LiSSA) using HVPs. Used in data valuation and finding mislabelled examples.

**3. Second-order optimisers (K-FAC, Shampoo, SOAP).**
All second-order optimisers for deep learning reduce to computing products of the form $\hat{H}^{-1}\mathbf{g}$ where $\hat{H}$ is a tractable approximation. The quality of the approximation determines how well the algorithm second-order corrects the gradient.

**4. Hyperparameter optimisation.**
Gradient-based hyperparameter optimisation (MAML meta-learning, bilevel optimisation) requires differentiating through optimisation steps. This involves HVPs of the **meta-loss** with respect to the base loss's Hessian — third-order derivatives!

**5. Neural Tangent Kernel (NTK).**
In the infinite-width limit, the NTK is $\Theta = J^\top J$ where $J = \frac{\partial f(\mathbf{x};\mathbf{w})}{\partial \mathbf{w}}$ is the Jacobian of the network output. HVPs of the squared loss Hessian equal NTK-vector products, connecting second-order optimisation to kernel theory.


---

## 9. Advanced Topics

### 9.1 Jacobians of Implicit Functions

The **Implicit Function Theorem (IFT)** gives the Jacobian of an implicitly defined function without solving for it explicitly.

**Setup.** Suppose $F: \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}^m$ is $C^1$ with $F(\mathbf{x}_0, \mathbf{y}_0) = 0$ and the partial Jacobian $\frac{\partial F}{\partial \mathbf{y}}(\mathbf{x}_0, \mathbf{y}_0) \in \mathbb{R}^{m\times m}$ is invertible. Then there exists a $C^1$ function $\mathbf{y} = g(\mathbf{x})$ near $\mathbf{x}_0$ satisfying $F(\mathbf{x}, g(\mathbf{x})) = 0$, with Jacobian:

$$J_g(\mathbf{x}_0) = -\left[\frac{\partial F}{\partial \mathbf{y}}(\mathbf{x}_0, \mathbf{y}_0)\right]^{-1} \frac{\partial F}{\partial \mathbf{x}}(\mathbf{x}_0, \mathbf{y}_0)$$

**Proof sketch.** Differentiate $F(\mathbf{x}, g(\mathbf{x})) = 0$ with respect to $\mathbf{x}$ using the chain rule:
$$\frac{\partial F}{\partial \mathbf{x}} + \frac{\partial F}{\partial \mathbf{y}}J_g = 0 \implies J_g = -\left(\frac{\partial F}{\partial \mathbf{y}}\right)^{-1}\frac{\partial F}{\partial \mathbf{x}}$$

**Application: Bilevel optimisation (meta-learning).** MAML's inner loop finds $\phi^*(w) = \arg\min_\phi \mathcal{L}_{\text{train}}(\phi; w)$ satisfying $\nabla_\phi \mathcal{L}_{\text{train}} = 0$. The meta-gradient requires $\frac{d\phi^*}{dw}$, which by IFT is:

$$\frac{d\phi^*}{dw} = -H_{\phi\phi}^{-1} \cdot H_{\phi w}$$

where $H_{\phi\phi} = \frac{\partial^2 \mathcal{L}}{\partial\phi^2}$ and $H_{\phi w} = \frac{\partial^2 \mathcal{L}}{\partial\phi\,\partial w}$. This requires second-order information (Hessians involving $\phi$ and $w$), which is why MAML-based methods are costly.

### 9.2 Clarke Subdifferential for Non-Smooth Functions

For non-differentiable $f$ (e.g., networks with ReLU), the Hessian does not exist at activation boundaries. The **Clarke subdifferential** extends the Jacobian to this setting.

**Definition (Clarke subdifferential).** For locally Lipschitz $f: \mathbb{R}^n \to \mathbb{R}$, the Clarke subdifferential is:

$$\partial_C f(\mathbf{x}) = \text{conv}\left\{\lim_{k\to\infty} \nabla f(\mathbf{x}_k) : \mathbf{x}_k \to \mathbf{x}, \mathbf{x}_k \in \Omega_f\right\}$$

where $\Omega_f$ is the set of points where $f$ is differentiable (measure 1 by Rademacher's theorem) and $\text{conv}$ denotes convex hull.

**For ReLU:** At $x = 0$, $\partial_C \text{ReLU}(x) = [0, 1]$ — the interval between the left and right derivatives. Any value in $[0,1]$ is a valid subgradient.

**Clarke Jacobian.** For $f: \mathbb{R}^n \to \mathbb{R}^m$, the Clarke generalised Jacobian $\partial_C f(\mathbf{x})$ is the convex hull of limiting Jacobians. For piecewise-smooth $f$, $\partial_C f(\mathbf{x})$ is the convex hull of Jacobians from each smooth piece meeting at $\mathbf{x}$.

**Generalised second-order theory.** There is no direct analogue of the Hessian in Clarke's framework. Instead, **semismoothness** (Qi & Sun 1993) allows Newton-like methods: a function is semismooth if the Clarke Jacobian converges in a directional sense. Semismooth Newton methods converge superlinearly to solutions of nonsmooth equations.

### 9.3 Fisher Information Matrix and Natural Gradient

The **Fisher Information Matrix** plays a central role in statistical learning theory and connects Jacobians to probabilistic models.

**Definition.** For a parametric model $p(x; \theta)$, the FIM is:

$$F(\theta) = \mathbb{E}_{x\sim p(\cdot;\theta)}\left[\frac{\partial \log p(x;\theta)}{\partial\theta}\frac{\partial \log p(x;\theta)}{\partial\theta}^\top\right] = -\mathbb{E}\left[\frac{\partial^2 \log p(x;\theta)}{\partial\theta^2}\right]$$

The second equality (Fisher's identity) shows $F = -\mathbb{E}[H_{\log p}]$ — the FIM equals the negative expected Hessian of the log-likelihood.

**Geometric interpretation.** The FIM defines a Riemannian metric on the statistical manifold $\mathcal{M} = \{p(\cdot;\theta)\}$. The **KL divergence** between nearby distributions is:

$$D_\text{KL}(p(\cdot;\theta)\|p(\cdot;\theta+\boldsymbol{\delta})) \approx \frac{1}{2}\boldsymbol{\delta}^\top F(\theta)\boldsymbol{\delta}$$

The FIM is the Hessian of the KL divergence. Natural gradient descent moves in the direction that most rapidly decreases the KL divergence, not the Euclidean loss:

$$\tilde{\nabla}_\theta \mathcal{L} = F(\theta)^{-1}\nabla_\theta \mathcal{L}$$

**Equivalence to Gauss-Newton.** For cross-entropy loss $\mathcal{L} = -\mathbb{E}[\log p(y|x;\theta)]$, the FIM equals the Gauss-Newton matrix $J^\top J$ where $J$ is the Jacobian of the log-probability outputs. This provides a PSD, well-conditioned curvature matrix.

**For transformers:** The FIM of a transformer language model is too large to compute exactly. K-FAC approximates it using the Kronecker structure. Recent work (Martens 2020, Bernacchia 2022) shows the FIM of attention layers has special structure exploitable for efficient natural gradient computation.

### 9.4 Jacobians in Normalising Flows and Change of Variables

**Change of variables formula.** If $f: \mathbb{R}^n \to \mathbb{R}^n$ is a diffeomorphism (bijective, smooth, smooth inverse) and $\mathbf{z} = f(\mathbf{x})$ with $\mathbf{x} \sim p_X$, then:

$$p_Z(\mathbf{z}) = p_X(f^{-1}(\mathbf{z}))\cdot |\det J_{f^{-1}}(\mathbf{z})| = p_X(\mathbf{x}) \cdot |\det J_f(\mathbf{x})|^{-1}$$

**Log-likelihood under a flow:**
$$\log p_Z(\mathbf{z}) = \log p_X(\mathbf{x}) - \log|\det J_f(\mathbf{x})|$$

The second term is the **log-determinant of the Jacobian** — the log-volume scaling factor.

**Normalising flows** (RealNVP, Glow, NICE) are composed diffeomorphisms $f = f_K \circ \cdots \circ f_1$. By the chain rule, $J_f = J_{f_K} \cdots J_{f_1}$, so $\det J_f = \prod_k \det J_{f_k}$. Each layer is designed so that $\det J_{f_k}$ is cheap to compute:
- **Coupling layers**: $J_{f_k}$ is triangular, $\det = \prod_i (J_{f_k})_{ii}$, computable in $O(n)$
- **Invertible 1×1 convolutions (Glow)**: $J_{f_k} = W$ (a learnable matrix), $\det W$ computed via LU decomposition

This is why Jacobian theory is central to generative modelling: the entire training objective depends on computing $\log|\det J_f|$ efficiently.


### 8.5 Conjugate Gradient for Solving $H\mathbf{x} = \mathbf{b}$

Many applications require not just $H\mathbf{v}$ but $H^{-1}\mathbf{v}$ — solving a linear system with the Hessian. The **Conjugate Gradient (CG)** method solves $H\mathbf{x} = \mathbf{b}$ for symmetric $H \succ 0$ using only matrix-vector products:

**CG algorithm:**
1. Initialise: $\mathbf{x}_0 = 0$, $\mathbf{r}_0 = \mathbf{b}$, $\mathbf{p}_0 = \mathbf{r}_0$
2. Repeat until convergence:
   - $\alpha_k = \|\mathbf{r}_k\|^2 / (\mathbf{p}_k^\top H \mathbf{p}_k)$ (one HVP: $H\mathbf{p}_k$)
   - $\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k \mathbf{p}_k$
   - $\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k H\mathbf{p}_k$
   - $\beta_k = \|\mathbf{r}_{k+1}\|^2 / \|\mathbf{r}_k\|^2$
   - $\mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k \mathbf{p}_k$ (update direction)

**Convergence.** CG converges in at most $n$ steps (exact arithmetic). In $k$ steps, the error satisfies:

$$\|\mathbf{x}_k - \mathbf{x}^*\|_H \leq 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k \|\mathbf{x}_0 - \mathbf{x}^*\|_H$$

where $\|\mathbf{y}\|_H = \sqrt{\mathbf{y}^\top H\mathbf{y}}$ and $\kappa = \lambda_{\max}/\lambda_{\min}$. For well-conditioned systems ($\kappa$ small), CG converges in a few dozen iterations — far fewer than $n$.

**For influence functions.** The LiSSA algorithm (Agarwal et al. 2017) solves $H^{-1}\mathbf{v}$ via stochastic approximation: draw batches $B_1, B_2, \ldots$, compute $H_{{B_t}}^{-1}$ via Taylor series truncation, average. Each iteration uses one stochastic HVP. For large networks ($n = 10^8$), this is the only feasible approach.

### 9.5 Jacobians and Automatic Differentiation: The Full Picture

This section ties together the JVP/VJP duality from §3.5 with the full structure of automatic differentiation.

**The AD computational graph.** Every computation is a directed acyclic graph (DAG) where:
- Nodes represent intermediate values $v_1, v_2, \ldots, v_k$
- Edges represent elementary operations (add, multiply, exp, sin, ...)
- Input nodes: $v_1, \ldots, v_n$ (the input variables $\mathbf{x}$)
- Output node: $v_k$ (the scalar loss $\mathcal{L}$)

Each elementary operation $v_j = g_j(v_{\text{pa}(j)})$ has a Jacobian $J_{g_j}$ — a small, cheap-to-compute matrix.

**Forward mode (JVP, tangent propagation).** Compute $\dot{v}_i = J_{f \text{ up to node } i} \cdot \dot{\mathbf{x}}$ for a fixed tangent vector $\dot{\mathbf{x}} = \mathbf{v}$. At each node:
$$\dot{v}_j = \sum_{i \in \text{pa}(j)} \frac{\partial g_j}{\partial v_i} \dot{v}_i$$

Running forward mode for each basis vector $\mathbf{e}_k$ separately gives column $k$ of the full Jacobian $J_f$. Cost: $n$ forward passes total, each $O(\text{time}(f))$.

**Reverse mode (VJP, adjoint propagation).** Compute $\bar{v}_i = \frac{\partial \mathcal{L}}{\partial v_i}$ backward through the graph. At each node, going backward:
$$\bar{v}_i = \sum_{j \in \text{ch}(i)} \frac{\partial g_j}{\partial v_i} \bar{v}_j$$

Starting from $\bar{v}_k = 1$ (derivative of $\mathcal{L}$ with respect to itself), this accumulates all partial derivatives in one backward pass. Cost: $1$ backward pass, $O(\text{time}(f))$.

**Why reverse mode wins for scalar $\mathcal{L}$.** Computing the full gradient $\nabla_{\mathbf{x}}\mathcal{L} \in \mathbb{R}^n$ requires one VJP (one backward pass). Computing it with JVPs requires $n$ forward passes — $n$ times more expensive. Since $n \sim 10^9$ for LLMs, reverse mode is the only option.

**Forward mode wins when $m \gg n$.** If $f: \mathbb{R}^n \to \mathbb{R}^m$ with $m \gg n$, computing the full Jacobian $J_f \in \mathbb{R}^{m \times n}$ requires $n$ JVPs (forward) or $m$ VJPs (reverse). When $n \ll m$, forward mode is cheaper. Example: sensitivity analysis of a physical simulation with $n = 5$ parameters and $m = 10^4$ output measurements.

**Higher-order derivatives.** The Hessian $H_f \in \mathbb{R}^{n \times n}$ can be computed by applying the AD operators twice:

| Method | Cost | When to use |
|--------|------|-------------|
| Reverse over reverse | $O(n^2)$ (full $H$) | Never for large $n$ |
| Forward over reverse | $O(n)$ per row (builds $H$ row-by-row) | When you need full $H$ at small $n$ |
| Reverse over forward | $O(n)$ per column | Same as above |
| HVP (reverse-of-dot) | $O(n)$ per HVP | **Standard**: only need $H\mathbf{v}$ |

The HVP trick from §8.1 (differentiate $\nabla f \cdot \mathbf{v}$) is exactly "reverse over the dot product with $\mathbf{v}$" — it's forward-then-reverse in the AD sense, but implemented as two reverse passes.

**Mixed-mode AD.** Some computations benefit from mixing forward and reverse within the same graph. JAX's `jax.hessian` function automatically selects forward-over-reverse or reverse-over-forward based on the function's input/output dimensions, using the cheaper composition.


---

## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|----------------|-----|
| 1 | Confusing Jacobian rows and columns: "the $(i,j)$ entry is $\partial x_i / \partial f_j$" | The $(i,j)$ entry of $J_f$ for $f:\mathbb{R}^n\to\mathbb{R}^m$ is $\partial f_i/\partial x_j$: row $i$ = gradient of output $i$, column $j$ = sensitivity of all outputs to input $j$ | Remember: rows index outputs, columns index inputs. Shape: $(m,n)$ for $f:\mathbb{R}^n\to\mathbb{R}^m$ |
| 2 | Forgetting the Hessian is symmetric | The Hessian $H_{ij} = \partial^2 f/\partial x_i\partial x_j$ is symmetric when $f\in C^2$ (Clairaut's theorem). Without symmetry assumption, algorithms that diagonalise $H$ are invalid | Check $H = H^\top$; if computing numerically, average with its transpose |
| 3 | Applying the second derivative test when $\nabla f(\mathbf{x}^*) \neq 0$ | The test "if $H \succ 0$ then $\mathbf{x}^*$ is a local minimum" requires $\mathbf{x}^*$ to be a critical point. A PD Hessian at a non-critical point just means the quadratic approximation has a minimum elsewhere | First verify $\nabla f(\mathbf{x}^*) = 0$, then check $H$ definiteness |
| 4 | Treating the Jacobian chain rule as commutative: $J_{f\circ g} = J_f \cdot J_g = J_g \cdot J_f$ | Matrix multiplication is not commutative. $J_{f\circ g}(\mathbf{x}) = J_f(g(\mathbf{x})) \cdot J_g(\mathbf{x})$ — evaluate outer function's Jacobian at the *output* of the inner function | Write dimensions explicitly: if $g:\mathbb{R}^n\to\mathbb{R}^k$, $f:\mathbb{R}^k\to\mathbb{R}^m$, then $J_{f\circ g}$ is $(m,k)\cdot(k,n)=(m,n)$ |
| 5 | Using GD learning rate $\eta = 1$ near a non-convex region | The stable range for GD is $0 < \eta < 2/\lambda_{\max}$. If $\lambda_{\max}$ is large (sharp direction), even $\eta = 0.01$ can cause oscillation | Estimate $\lambda_{\max}$ via power iteration (10–20 HVPs) and set $\eta < 2/\lambda_{\max}$ |
| 6 | Confusing JVP and VJP modes | JVP (forward mode): computes $J_f\mathbf{v}$, cost $O(n)$ per vector; VJP (reverse mode): computes $J_f^\top\mathbf{u}$, cost $O(m)$ per vector. For scalar loss ($m=1$), VJP is $O(1)$ passes regardless of $n$; JVP would cost $O(n)$ passes | Use reverse mode (backprop) for scalar losses; use forward mode when $m \gg n$ |
| 7 | Concluding a flat Hessian means global minimum | $H_f = 0$ at $\mathbf{x}$ only means the quadratic approximation is flat. The actual function could be a plateau, a flat saddle, or a valley floor. Example: $f(x) = x^3$ has $f''(0) = 0$ but $x=0$ is an inflection point, not a minimum | Check higher-order derivatives or sufficient decrease conditions from all directions |
| 8 | Computing $J_f\mathbf{v}$ by forming $J_f$ first | For large $n$, forming $J_f \in \mathbb{R}^{m\times n}$ costs $O(mn)$ memory and $O(n)$ backward passes. Computing $J_f\mathbf{v}$ directly via forward-mode AD costs $O(n)$ in one pass (single JVP) | Use `torch.autograd.functional.jvp` or JAX's `jvp` for forward-mode; never materialise the full Jacobian if you only need the product |
| 9 | Assuming the softmax Jacobian is diagonal | The softmax $\sigma:\mathbb{R}^K\to\mathbb{R}^K$ has Jacobian $J_\sigma = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ — a full symmetric rank-$(K-1)$ matrix. Only the diagonal entries $p_i(1-p_i)$ appear on the diagonal | When backpropagating through softmax, use the full Jacobian. Many frameworks handle this correctly, but manual implementations often miss the $-p_ip_j$ off-diagonal terms |
| 10 | Ignoring that the Hessian of a ReLU network is zero almost everywhere | ReLU networks are piecewise linear. Their Hessian is 0 in each linear region. Newton's method applied to a ReLU network Hessian gives a zero-curvature degenerate system | Use Gauss-Newton or K-FAC (which use $J^\top J$ from the output Jacobian, not the parameter Hessian) for ReLU networks |
| 11 | Confusing the FIM with the Hessian of the loss | The FIM $F = \mathbb{E}[(\nabla\log p)(\nabla\log p)^\top]$ is not the same as $H_{\mathcal{L}}$ (Hessian of the training loss). They are equal only when the model is at its optimum (Fisher identity). Away from the optimum, $H_{\mathcal{L}} = F + \mathbb{E}[H_{\log p}]$ (the Hessian includes an extra term for model misspecification) | Natural gradient uses $F^{-1}$; full second-order Newton uses $H_{\mathcal{L}}^{-1}$. K-FAC approximates $F$, not $H_{\mathcal{L}}$ |
| 12 | Expecting Newton's method to work with indefinite Hessians | Newton's step $-H^{-1}\mathbf{g}$ is only guaranteed to be a descent direction when $H \succ 0$. At saddle points, the Newton step points toward the saddle, making loss increase | Add a regularisation term: $-(H + \mu I)^{-1}\mathbf{g}$ with $\mu > 0$ chosen so $H + \mu I \succ 0$ (Levenberg-Marquardt) |


---

## 11. Exercises

**Exercise 1 (★).** *Computing Jacobians.*

Let $f: \mathbb{R}^3 \to \mathbb{R}^2$ be defined by $f(x_1, x_2, x_3) = (x_1^2 + x_2x_3,\; e^{x_1} - x_3^2)$.

(a) Compute $J_f(\mathbf{x})$ for general $\mathbf{x}$.

(b) Evaluate $J_f$ at $\mathbf{x}_0 = (0, 1, 1)$.

(c) Compute $J_f\mathbf{v}$ at $\mathbf{x}_0$ for $\mathbf{v} = (1, 0, -1)^\top$ (the JVP).

(d) Compute $J_f^\top\mathbf{u}$ at $\mathbf{x}_0$ for $\mathbf{u} = (1, 2)^\top$ (the VJP).

(e) Verify that $\mathbf{u}^\top(J_f\mathbf{v}) = (J_f^\top\mathbf{u})^\top\mathbf{v}$ (duality of JVP and VJP).

---

**Exercise 2 (★).** *Softmax Jacobian properties.*

Let $\sigma: \mathbb{R}^3 \to \mathbb{R}^3$ be the softmax function.

(a) For $\mathbf{z} = (1, 2, 0)^\top$, compute $\mathbf{p} = \sigma(\mathbf{z})$ and the full $3\times 3$ Jacobian $J_\sigma(\mathbf{z}) = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$.

(b) Verify $J_\sigma\mathbf{1} = \mathbf{0}$ (the all-ones vector is in the null space).

(c) Verify $J_\sigma$ is symmetric and compute its eigenvalues numerically.

(d) Show that $\text{rank}(J_\sigma) = K-1$ by arguing that $\mathbf{1}$ spans the null space.

(e) **For AI:** In cross-entropy loss $\mathcal{L} = -\log p_y$, the gradient of $\mathcal{L}$ with respect to the logits $\mathbf{z}$ is $\mathbf{p} - \mathbf{e}_y$ (where $\mathbf{e}_y$ is the one-hot vector). Derive this from $\frac{\partial\mathcal{L}}{\partial\mathbf{z}} = J_\sigma^\top \frac{\partial\mathcal{L}}{\partial\mathbf{p}}$.

---

**Exercise 3 (★).** *Hessian computation and classification.*

Let $f(x, y) = x^3 + 3xy - y^3$.

(a) Find all critical points (where $\nabla f = 0$).

(b) Compute $H_f(x, y)$ at each critical point.

(c) Classify each critical point as local min, local max, or saddle using the second derivative test.

(d) For the saddle point, find the two eigenvectors of $H_f$ and describe the geometry: in which directions is $f$ increasing/decreasing quadratically?

---

**Exercise 4 (★★).** *LayerNorm Jacobian.*

Consider LayerNorm $\text{LN}: \mathbb{R}^d \to \mathbb{R}^d$ (without learnable parameters $\gamma, \beta$):

$$\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \bar{x}\mathbf{1}}{\sigma}, \quad \bar{x} = \frac{1}{d}\sum_i x_i, \quad \sigma = \sqrt{\frac{1}{d}\sum_i(x_i-\bar{x})^2}$$

(a) Show that $J_{\text{LN}}(\mathbf{x}) = \frac{1}{\sigma}(I - \frac{1}{d}\mathbf{1}\mathbf{1}^\top - \frac{1}{d}\hat{\mathbf{x}}\hat{\mathbf{x}}^\top)$ where $\hat{\mathbf{x}} = (\mathbf{x}-\bar{x}\mathbf{1})/\sigma$.

(b) Verify that $\mathbf{1}$ and $\hat{\mathbf{x}}$ are in the null space of $J_{\text{LN}}$.

(c) Compute the rank of $J_{\text{LN}}$ and explain its geometric meaning (what input variations LayerNorm cannot distinguish).

(d) For $d = 3$ and $\mathbf{x} = (1, 2, 3)^\top$, compute $J_{\text{LN}}$ numerically and verify your formula.

---

**Exercise 5 (★★).** *Hessian-vector product and edge of stability.*

For the MSE loss $f(\mathbf{w}) = \frac{1}{2m}\|X\mathbf{w} - \mathbf{y}\|^2$ with $X \in \mathbb{R}^{m\times n}$:

(a) Show that $H_f = \frac{1}{m}X^\top X$ and $\lambda_{\max}(H_f) = \frac{1}{m}\lambda_{\max}(X^\top X)$.

(b) Implement the HVP formula: $H_f\mathbf{v} = \frac{1}{m}X^\top(X\mathbf{v})$ (matrix-free, $O(mn)$).

(c) Use 10 iterations of power iteration starting from a random $\mathbf{v}$ to estimate $\lambda_{\max}(H_f)$.

(d) Compute the maximum stable learning rate $\eta_{\max} = 2/\lambda_{\max}$ and verify numerically that GD with $\eta = 1.1\eta_{\max}$ diverges while $\eta = 0.9\eta_{\max}$ converges.

(e) **For AI:** Explain why the "edge of stability" phenomenon (Cohen et al. 2022) — where SGD's training loss oscillates as $\lambda_{\max} \approx 2/\eta$ — is consistent with this analysis.

---

**Exercise 6 (★★).** *Newton's method convergence.*

Consider $f(x) = \log(1 + e^x)$ (softplus).

(a) Compute $f'(x) = \sigma(x)$ (sigmoid) and $f''(x) = \sigma(x)(1-\sigma(x))$.

(b) Apply 5 steps of Newton's method starting from $x_0 = 5$ to minimise $g(x) = f(x) - 2x$ (the minimum is at $x^* = \log 3$). Record the error $|x_t - x^*|$ at each step.

(c) Plot the sequence $\log|x_t - x^*|$ vs $t$. Is the convergence roughly linear or quadratic (does the log-error decrease linearly or super-linearly)?

(d) Now add a constraint: minimise the same $g$ but also track the Newton decrement $\lambda_N = \sqrt{(g'(x))^2/g''(x)}$ at each step. Interpret its role as a stopping criterion.

---

**Exercise 7 (★★★).** *K-FAC structure.*

Consider a single linear layer: $\mathbf{y} = W\mathbf{x}$ with $W \in \mathbb{R}^{n_\text{out}\times n_\text{in}}$, loss $\mathcal{L}$.

(a) Show that $\text{vec}(\nabla_W\mathcal{L}) = \mathbf{x}\otimes\mathbf{g}$ where $\mathbf{g} = \partial\mathcal{L}/\partial\mathbf{y}$ and $\otimes$ is the Kronecker product.

(b) The exact FIM is $F = \mathbb{E}[(\mathbf{x}\otimes\mathbf{g})(\mathbf{x}\otimes\mathbf{g})^\top] = \mathbb{E}[\mathbf{x}\mathbf{x}^\top\otimes\mathbf{g}\mathbf{g}^\top]$. Show that K-FAC's independence assumption $F \approx A\otimes G$ (with $A = \mathbb{E}[\mathbf{x}\mathbf{x}^\top]$, $G = \mathbb{E}[\mathbf{g}\mathbf{g}^\top]$) gives $F^{-1} \approx A^{-1}\otimes G^{-1}$.

(c) Show that the K-FAC natural gradient step $\Delta W = -G^{-1}(\nabla_W\mathcal{L})A^{-1}$ (reshape form) implements $-\text{vec}^{-1}(F^{-1}\text{vec}(\nabla_W\mathcal{L}))$.

(d) Generate random $n_\text{in}=4$, $n_\text{out}=3$, compute exact FIM from 100 samples, compute K-FAC approximation, and compare $F^{-1}\mathbf{g}$ (exact) with $(A^{-1}\otimes G^{-1})\mathbf{g}$ (K-FAC) numerically.

---

**Exercise 8 (★★★).** *Jacobians in normalising flows.*

Implement a single coupling layer (as in RealNVP) and verify the log-determinant formula.

(a) Define a coupling layer $f: \mathbb{R}^4\to\mathbb{R}^4$: split $\mathbf{x} = (\mathbf{x}_1, \mathbf{x}_2) \in \mathbb{R}^2\times\mathbb{R}^2$; output $(\mathbf{y}_1, \mathbf{y}_2) = (\mathbf{x}_1, \mathbf{x}_2 \odot e^{s(\mathbf{x}_1)} + t(\mathbf{x}_1))$ where $s, t: \mathbb{R}^2\to\mathbb{R}^2$ are simple networks.

(b) Show that the Jacobian $J_f$ is lower-triangular, and the log-determinant is $\log|\det J_f| = \sum_i s_i(\mathbf{x}_1)$ (sum of scale outputs).

(c) Numerically compute $J_f$ by finite differences for a specific $\mathbf{x}$, and verify $\log|\det J_f| = \sum_i s_i(\mathbf{x}_1)$.

(d) Implement the inverse $f^{-1}(\mathbf{y}) = (\mathbf{y}_1, (\mathbf{y}_2 - t(\mathbf{y}_1))\odot e^{-s(\mathbf{y}_1)})$ and verify $f^{-1}(f(\mathbf{x})) = \mathbf{x}$.

(e) **For generative modelling:** Explain why the tractable log-determinant (which avoids $O(n^3)$ computation) is essential for training normalising flows via maximum likelihood.


---

## 12. Why This Matters for AI (2026 Perspective)

| Concept | Concrete AI/ML Usage |
|---------|---------------------|
| **Jacobian of layer outputs** | Backpropagation is iterated Jacobian-vector products (VJPs) through each layer. Every training step of every neural network in the world computes these products implicitly |
| **Softmax Jacobian** | Understanding that $J_\sigma = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ is rank-$(K-1)$ explains why large vocabulary softmax is numerically tricky (singular Jacobian) and why temperature scaling affects gradient magnitudes |
| **LayerNorm Jacobian** | The rank-$(d-2)$ Jacobian of LayerNorm means gradients cannot pass through in the directions $\mathbf{1}$ and $\hat{\mathbf{x}}$ — this "gradient bottleneck" explains why Pre-LN transformers (LN before attention) converge faster than Post-LN (LN after attention) |
| **JVP vs VJP (forward vs reverse mode)** | All modern deep learning frameworks (PyTorch, JAX) use reverse mode. Forward mode is used in specific settings: sensitivity analysis, Hessian-diagonal estimation, gradient checkpointing trade-offs |
| **Jacobian condition number** | High condition number → vanishing/exploding gradients in deep networks. Xavier and He initialisation set the initial Jacobian's singular values near 1 across layers, ensuring stable gradient flow at init |
| **Hessian eigenspectrum** | The largest Hessian eigenvalue determines the maximum stable learning rate. The "edge of stability" phenomenon (Cohen et al. 2022) shows SGD self-regulates to $\eta \approx 2/\lambda_{\max}$ — foundational for understanding training dynamics |
| **Flat vs sharp minima (SAM)** | SAM and its variants (ASAM, mSAM, SAM-ON, Adan-SAM) are widely used in production systems. SAM finds flatter minima, improving generalisation by 1-3% on vision and NLP benchmarks. The analysis of flat vs sharp minima requires understanding Hessian definiteness and condition numbers |
| **Hessian-vector products** | Power iteration on the Hessian (using HVPs) gives $\lambda_{\max}$ for learning rate selection. Influence functions (for data valuation, finding mislabelled examples, understanding model behaviour) use iterative HVP solvers |
| **K-FAC / natural gradient** | Second-order methods that approximate the Fisher Information Matrix (= Gauss-Newton matrix) achieve 10-30× speedup over Adam in terms of steps to convergence. EKFAC, KFAC-Reduce used in large-scale transformer training at Google and DeepMind |
| **Jacobian determinant / normalising flows** | The change-of-variables formula requires $\log|\det J_f|$. Tractable log-determinants (via triangular Jacobians in coupling layers) are the key insight behind RealNVP, Glow, and Flow Matching for generative modelling |
| **IFT / bilevel optimisation** | MAML meta-learning, hyperparameter optimisation via gradient, and NAS all require differentiating through optimisation loops. This requires Hessian-vector products in the IFT formula $d\phi^*/dw = -H_{\phi\phi}^{-1}H_{\phi w}$ |
| **Neural Tangent Kernel (NTK)** | In the infinite-width limit, the kernel is $\Theta = J^\top J$ (output Jacobian squared). Understanding NTK evolution during training (NTK changes vs stays constant) explains why wide networks behave like kernel machines and why the lazy training regime exists |
| **RoPE and attention Jacobians** | Rotary Position Embeddings (RoPE, used in LLaMA, Mistral, Qwen) are orthogonal transformations applied to queries and keys. Their Jacobians are rotation matrices (orthogonal, $|\det| = 1$, condition number = 1) — a mathematically elegant design that preserves gradient norms |

---

## Conceptual Bridge

**Looking backward.** This section builds directly on §01-Partial-Derivatives-and-Gradients, which introduced the gradient as the collection of partial derivatives for scalar functions. The Jacobian is the natural generalisation: instead of a single gradient vector, we have a matrix of gradients for each output. Every formula from the gradient section — chain rule, directional derivative, gradient descent — has a Jacobian analogue: chain rule for Jacobians ($J_{f\circ g} = J_f \cdot J_g$), directional derivative as JVP ($J_f\mathbf{v}$), and steepest ascent/descent replacing gradient with VJP backpropagation.

**Looking forward.** The Hessian established here is the foundation for the next two sections:

- **§03-Optimization-on-Manifolds**: When parameters are constrained to a manifold (e.g., orthogonal matrices $O(n)$, unit sphere), the Riemannian Hessian replaces the Euclidean Hessian. The Fisher Information Matrix defines a natural Riemannian metric on statistical manifolds.

- **§04-Automatic-Differentiation**: The JVP and VJP operators computed throughout this section are exactly what forward-mode and reverse-mode AD implement. The Jacobian-vector product is the primitive of forward-mode; the vector-Jacobian product is the primitive of reverse-mode. Understanding both modes, and when to use each, is the foundation of efficient AD system design.

Beyond this chapter: Jacobians and Hessians appear throughout the rest of the curriculum:
- **Optimisation (Chapter 8)**: gradient descent convergence rates, condition numbers, Newton's method, and conjugate gradients all depend on the Hessian eigenspectrum
- **Probability and Statistics (Chapter 9)**: the Fisher Information Matrix and Cramér-Rao bound require Jacobians of log-likelihood functions
- **Differential Geometry (Chapter 10)**: the pushforward and pullback are Jacobian maps between tangent spaces on manifolds

```
POSITION IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  § 01-Partial-Derivatives-and-Gradients
  (scalar functions: ∇f, directional derivatives, chain rule)
           │
           ▼
  § 02-Jacobians-and-Hessians  ← YOU ARE HERE
  (vector functions: J_f, H_f, second-order analysis)
           │
           ├─────────────────────────────────────────────────────────
           │                                                         │
           ▼                                                         ▼
  § 03-Optimization-on-Manifolds                          § 04-Automatic-Differentiation
  (Riemannian Hessian, natural gradient)                  (JVP/VJP implementation)
           │                                                         │
           └─────────────────┬───────────────────────────────────────┘
                             ▼
              Chapter 8: Optimisation
              (convergence theory, Newton, second-order methods)

════════════════════════════════════════════════════════════════════════
```

The Jacobian and Hessian are the core analytical tools of multivariate analysis. Every gradient-based learning algorithm, every convergence theorem, every second-order method in deep learning is a consequence of these fundamental objects. With this section complete, you have the mathematical foundation to understand not just *how* modern optimisers work, but *why* they work — and where they break down.

[← Back to Chapter 05: Multivariate Calculus](../README.md) | [Next: §03 Optimization on Manifolds →](../03-Optimization-on-Manifolds/notes.md)


---

## 13. Numerical Methods and Finite Differences

Before automatic differentiation, Jacobians and Hessians were computed numerically. Even today, numerical methods are essential for verifying analytical gradients (gradient checking), debugging implementations, and understanding numerical stability.

### 13.1 Finite Difference Approximations

**First-order differences.** For $f: \mathbb{R}^n \to \mathbb{R}$:

$$\frac{\partial f}{\partial x_j} \approx \frac{f(\mathbf{x} + h\mathbf{e}_j) - f(\mathbf{x})}{h} \quad \text{(forward difference, error } O(h)\text{)}$$

$$\frac{\partial f}{\partial x_j} \approx \frac{f(\mathbf{x} + h\mathbf{e}_j) - f(\mathbf{x} - h\mathbf{e}_j)}{2h} \quad \text{(centred difference, error } O(h^2)\text{)}$$

The centred difference is preferred: it is exact for quadratics (cancels the $O(h^2)$ error term via symmetry) and reduces to forward difference cost only when both evaluations can be reused.

**Second-order differences (Hessian diagonal).** 

$$\frac{\partial^2 f}{\partial x_j^2} \approx \frac{f(\mathbf{x}+h\mathbf{e}_j) - 2f(\mathbf{x}) + f(\mathbf{x}-h\mathbf{e}_j)}{h^2} \quad \text{(centred, error } O(h^2)\text{)}$$

**Off-diagonal Hessian entries.** Two ways:
$$\frac{\partial^2 f}{\partial x_i \partial x_j} \approx \frac{f(\mathbf{x}+h(\mathbf{e}_i+\mathbf{e}_j)) - f(\mathbf{x}+h\mathbf{e}_i) - f(\mathbf{x}+h\mathbf{e}_j) + f(\mathbf{x})}{h^2}$$

Computing the full Hessian this way requires $O(n^2)$ function evaluations — completely infeasible for large $n$.

### 13.2 Gradient Checking

Gradient checking compares the analytical gradient $\mathbf{g}_{\text{analytical}} = \nabla f(\mathbf{x})$ (computed by backprop) against the numerical estimate $\mathbf{g}_{\text{numerical}}$ (computed by centred differences):

$$\text{relative error} = \frac{\|\mathbf{g}_{\text{analytical}} - \mathbf{g}_{\text{numerical}}\|}{\|\mathbf{g}_{\text{analytical}}\| + \|\mathbf{g}_{\text{numerical}}\|}$$

A relative error of $< 10^{-5}$ usually indicates a correct implementation; $> 10^{-3}$ indicates a bug. This is a **gold standard debugging technique** — every significant ML framework implementation should be gradient-checked before trusting gradients.

**Common pitfalls in gradient checking:**
1. **Kink near the evaluation point**: ReLU has $f'(0)$ undefined. If the evaluation point happens to be near $x = 0$ for some neuron, the numerical gradient straddles a kink and gives an inaccurate estimate. Fix: perturb the input slightly away from known kinks.
2. **Float32 precision**: Centred differences in float32 have catastrophic cancellation for small $h$ (both terms round to the same value). Use $h \approx 10^{-5}$ for float32, $h \approx 10^{-7}$ for float64.
3. **Dropout and batch norm**: These layers behave differently in train vs eval mode. Gradient check with these layers frozen or in eval mode.

**Stochastic gradient checking.** Instead of checking all $n$ coordinates (cost: $2n$ function evaluations), check a random projection: compute $\mathbf{v}^\top \mathbf{g}_{\text{analytical}}$ (one dot product) and compare against $(f(\mathbf{x}+h\mathbf{v}) - f(\mathbf{x}-h\mathbf{v}))/(2h)$ (one centred difference in direction $\mathbf{v}$). This costs only 2 function evaluations regardless of $n$, and catches most gradient bugs.

### 13.3 Numerical Stability of Jacobian Computations

**Condition number and ill-conditioned Jacobians.** The condition number $\kappa(J_f) = \sigma_{\max}(J_f)/\sigma_{\min}(J_f)$ measures how amplified input perturbations become in the output. For a neural network layer, $\kappa(J_f)$ large means:
- Small input perturbations (e.g., numerical noise) produce large output perturbations
- Backpropagated gradients through this layer are magnified or shrunk by $\kappa$

**Ill-conditioning accumulates through depth.** For a depth-$L$ network, the condition number of the full Jacobian $J = J_L \cdots J_1$ satisfies:
$$\kappa(J) \leq \prod_{l=1}^L \kappa(J_l)$$

This bounds can be exponential in $L$ — explaining gradient explosion. Good architecture design (residual connections, normalisation layers) keeps $\kappa(J_l) \approx 1$ for each layer.

**Log-sum-exp trick.** The softmax Jacobian involves $p_i = e^{z_i}/\sum_j e^{z_j}$. Direct computation overflows for large $z_i$. The numerically stable implementation subtracts $\max_j z_j$ before exponentiation:
$$p_i = \frac{e^{z_i - z_{\max}}}{\sum_j e^{z_j - z_{\max}}}$$

This cancellation doesn't change the output but keeps all exponentials $\leq 1$. The Jacobian formula $J_\sigma = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ remains the same — only the computation of $\mathbf{p}$ is stabilised.

**Mixed precision Hessian computations.** When computing HVPs in mixed precision (float16/bfloat16 for activations, float32 for gradients), the second differentiation step (gradient of gradient) may accumulate significant rounding error. Google's T5X and JAX-based training systems perform HVP computations in float32 even during mixed-precision training to maintain numerical accuracy of second-order estimates.


---

## 14. Jacobians in Modern Transformer Architectures

Transformers introduced several operations with non-trivial Jacobians. Understanding these helps explain training behaviour, initialisation requirements, and the design of efficient fine-tuning methods.

### 14.1 Attention Jacobian

The scaled dot-product attention function maps $(\mathbf{Q}, \mathbf{K}, \mathbf{V}) \to \mathbf{O}$:
$$\mathbf{A} = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right), \qquad \mathbf{O} = \mathbf{A}\mathbf{V}$$

The Jacobian of $\mathbf{O}$ with respect to $\mathbf{Q}$ (for a single head, single query position $i$):

$$\frac{\partial O_i}{\partial Q_i} = \frac{1}{\sqrt{d_k}} (J_\sigma(\mathbf{s}_i))^\top \mathbf{V}$$

where $\mathbf{s}_i = \mathbf{K}\mathbf{q}_i/\sqrt{d_k}$ and $J_\sigma = \text{diag}(\mathbf{a}_i) - \mathbf{a}_i\mathbf{a}_i^\top$ is the softmax Jacobian. The full attention Jacobian is a complex composition involving the softmax Jacobian, the key matrix $\mathbf{K}$, and the value matrix $\mathbf{V}$.

**Implications of the softmax Jacobian in attention:**
- The singular softmax Jacobian (null space = span{$\mathbf{1}$}) means attention weights are translation-invariant: shifting all logits $\mathbf{s}_i$ by a constant doesn't change the output
- Temperature $\tau$ scaling ($\text{softmax}(\mathbf{s}/\tau)$) scales the Jacobian: $J_\sigma^{(\tau)} = (1/\tau) J_\sigma^{(1)}$. High temperature ($\tau \gg 1$) → nearly-uniform attention → small Jacobian → slow learning. Low temperature ($\tau \ll 1$) → peaked attention → large Jacobian → sharp gradients.
- **Attention entropy collapse**: if attention concentrates on one token (small $\tau$ or sharp $\mathbf{s}$), $\mathbf{a} \approx \mathbf{e}_j$ for some $j$, and $J_\sigma \approx 0$ (PSD, small norm). Gradients vanish through the attention softmax.

### 14.2 RoPE and Orthogonal Jacobians

**Rotary Position Embeddings (RoPE)**, used in LLaMA, Mistral, Qwen, DeepSeek, apply a position-dependent rotation to queries and keys. For position $m$ and dimension pair $(2i, 2i+1)$:

$$\begin{pmatrix}q_{2i}' \\ q_{2i+1}'\end{pmatrix} = \begin{pmatrix}\cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i)\end{pmatrix}\begin{pmatrix}q_{2i} \\ q_{2i+1}\end{pmatrix}$$

The Jacobian of the RoPE transformation (with respect to the query) is a block-diagonal rotation matrix $R_m \in O(d)$ — a **rotation matrix** with:
- $J_{\text{RoPE}} = R_m$ (orthogonal matrix)
- $\det(R_m) = 1$ (volume-preserving)
- $\sigma_{\max} = \sigma_{\min} = 1$ (condition number = 1)
- $\|J_{\text{RoPE}}\|_2 = 1$ (isometry: no gradient scaling)

RoPE's orthogonal Jacobian is a mathematically elegant design: gradient norms are preserved through the positional encoding transformation, preventing it from being a source of gradient instability.

### 14.3 LoRA and the Low-Rank Jacobian Approximation

Low-Rank Adaptation (LoRA, Hu et al. 2022) freezes the original weight matrix $W_0 \in \mathbb{R}^{m\times n}$ and adds a trainable low-rank perturbation:
$$W = W_0 + BA, \quad B \in \mathbb{R}^{m\times r},\; A \in \mathbb{R}^{r\times n},\; r \ll \min(m,n)$$

The Jacobian of the layer output $\mathbf{y} = W\mathbf{x}$ with respect to the LoRA parameters $(A, B)$:

$$\frac{\partial \mathbf{y}}{\partial \text{vec}(A)} = B \otimes \mathbf{x}^\top, \qquad \frac{\partial \mathbf{y}}{\partial \text{vec}(B)} = I_m \otimes (A\mathbf{x})^\top$$

The gradient of the loss with respect to $B$: $\nabla_B \mathcal{L} = \boldsymbol{\delta}(A\mathbf{x})^\top$ — an outer product of rank 1, matching the original weight gradient structure. The gradient with respect to $A$: $\nabla_A \mathcal{L} = B^\top\boldsymbol{\delta}\mathbf{x}^\top$ — again rank at most $r$.

**Why LoRA works (Jacobian perspective).** The empirical observation that full fine-tuning has low "intrinsic dimensionality" (Aghajanyan et al. 2021) means the weight gradient $\nabla_W\mathcal{L}$ concentrates in a low-rank subspace during fine-tuning. LoRA parameterises updates in this subspace directly. The Jacobian of the LoRA update rule has $mr + rn$ parameters vs $mn$ for full fine-tuning — a reduction of $\min(m,n)/r$ in trainable parameters.

**DoRA (Weight-Decomposed LoRA, Liu et al. 2024).** Decomposes the weight as $W = m\cdot(W_0 + BA)/\|W_0 + BA\|$ (magnitude × direction). The Jacobian of the direction normalisation introduces a LayerNorm-like structure — the gradient of the magnitude scalar and the direction matrix decouple, enabling more stable fine-tuning.


### 14.4 Gradient Flow Through Residual Connections

The **residual connection** $\mathbf{y} = \mathbf{x} + F(\mathbf{x})$ (introduced in ResNets, universal in transformers) has a special Jacobian structure:

$$J_{\mathbf{y}}(\mathbf{x}) = I + J_F(\mathbf{x})$$

This is the key insight: the Jacobian of a residual block is **always $I$ plus a perturbation**. Even if $J_F$ has small singular values (vanishing), the sum $I + J_F$ has singular values bounded below by $1 - \|J_F\|_2$. For a well-initialised network, $\|J_F\|_2 \ll 1$ at init, so $J_{\mathbf{y}} \approx I$ — the identity map passes gradients through unchanged.

**Depth effect.** For an $L$-layer residual network:
$$J_{\text{network}} = \prod_{l=1}^L (I + J_{F_l}) = I + \sum_l J_{F_l} + \sum_{l < l'} J_{F_{l'}}J_{F_l} + \cdots$$

The leading term is $I$ (identity shortcut from input to output), followed by first-order terms (single-layer Jacobians), then higher-order interaction terms. This **identity shortcut** is why ResNets train stably at depth $L = 1000+$: the gradient can always flow through the identity path without attenuation.

**Comparison to plain networks.** For a plain (non-residual) network: $J_{\text{network}} = J_{F_L}\cdots J_{F_1}$. This product can have exponentially small singular values at depth — the vanishing gradient problem. Residual connections solve this by ensuring the Jacobian includes an identity component.

**For transformers:** Every transformer layer adds a residual connection around both the attention block and the MLP block. The combined Jacobian is $(I + J_{\text{attn}})(I + J_{\text{MLP}}) = I + J_{\text{attn}} + J_{\text{MLP}} + J_{\text{attn}}J_{\text{MLP}}$ — dominated by $I$ at initialisation, with the attention and MLP Jacobians as small corrections.


---

## 15. Worked Examples: End-to-End Computations

### 15.1 Complete Backprop Trace for a Two-Layer Network

Let us trace the full computation of Jacobians and VJPs for the small network:

$$\mathbf{z}^{(1)} = W_1\mathbf{x}, \qquad \mathbf{h}^{(1)} = \text{ReLU}(\mathbf{z}^{(1)}), \qquad \mathbf{z}^{(2)} = W_2\mathbf{h}^{(1)}, \qquad \mathcal{L} = \frac{1}{2}\|\mathbf{z}^{(2)} - \mathbf{y}\|^2$$

with $W_1 \in \mathbb{R}^{d_1\times d_0}$, $W_2 \in \mathbb{R}^{d_2\times d_1}$.

**Forward pass (computing values):**
1. $\mathbf{z}^{(1)} = W_1\mathbf{x}$ (linear)
2. $\mathbf{h}^{(1)} = \text{ReLU}(\mathbf{z}^{(1)})$ (elementwise)
3. $\mathbf{z}^{(2)} = W_2\mathbf{h}^{(1)}$ (linear)
4. $\mathcal{L} = \frac{1}{2}\|\mathbf{z}^{(2)} - \mathbf{y}\|^2$ (loss)

**Backward pass (computing Jacobians and VJPs):**

Start from the loss: $\frac{\partial\mathcal{L}}{\partial\mathbf{z}^{(2)}} = \mathbf{z}^{(2)} - \mathbf{y} \in \mathbb{R}^{d_2}$ (call this $\boldsymbol{\delta}^{(2)}$).

*Step 1: Through linear layer $W_2$.*

The Jacobian of $\mathbf{z}^{(2)} = W_2\mathbf{h}^{(1)}$ with respect to $W_2$ and $\mathbf{h}^{(1)}$:
- $J_{\mathbf{z}^{(2)}, W_2}$: $\frac{\partial z^{(2)}_i}{\partial [W_2]_{ij}} = h^{(1)}_j$, so $\nabla_{W_2}\mathcal{L} = \boldsymbol{\delta}^{(2)}(\mathbf{h}^{(1)})^\top \in \mathbb{R}^{d_2\times d_1}$
- $J_{\mathbf{z}^{(2)}, \mathbf{h}^{(1)}} = W_2$, so VJP: $\boldsymbol{\delta}^{(1)} = W_2^\top\boldsymbol{\delta}^{(2)} \in \mathbb{R}^{d_1}$

*Step 2: Through ReLU.*

$J_{\mathbf{h}^{(1)}, \mathbf{z}^{(1)}} = \text{diag}(\mathbf{1}[\mathbf{z}^{(1)} > 0])$ (diagonal mask). VJP:
$$\frac{\partial\mathcal{L}}{\partial\mathbf{z}^{(1)}} = \boldsymbol{\delta}^{(1)} \odot \mathbf{1}[\mathbf{z}^{(1)} > 0] \in \mathbb{R}^{d_1}$$

(elementwise multiply by the activation mask)

*Step 3: Through linear layer $W_1$.*

Let $\boldsymbol{\delta}^{(0)} = \frac{\partial\mathcal{L}}{\partial\mathbf{z}^{(1)}}$.
- $\nabla_{W_1}\mathcal{L} = \boldsymbol{\delta}^{(0)}\mathbf{x}^\top \in \mathbb{R}^{d_1\times d_0}$
- $\nabla_{\mathbf{x}}\mathcal{L} = W_1^\top\boldsymbol{\delta}^{(0)} \in \mathbb{R}^{d_0}$ (if we need the input gradient)

**Summary of the structure.** Every backward step computes:
- Weight gradient = (backpropagated error) × (forward activation)$^\top$ — always a rank-1 outer product in a single example
- Activation gradient = (weight matrix)$^\top$ × (backpropagated error) — transposed Jacobian times incoming gradient

This pattern repeats identically for any depth: backward pass = chain of transposed-Jacobian-vector products (VJPs), with weight gradients as outer products at each layer.

### 15.2 Numerical Verification of the Softmax Jacobian

A worked numerical example. Take $\mathbf{z} = (1, 0, -1)^\top$ and $K = 3$.

Compute $\mathbf{p} = \text{softmax}(\mathbf{z})$:
$$e^1 \approx 2.718, \quad e^0 = 1, \quad e^{-1} \approx 0.368, \qquad Z = 2.718+1+0.368 = 4.086$$
$$p_1 = 2.718/4.086 \approx 0.665, \quad p_2 = 1/4.086 \approx 0.245, \quad p_3 = 0.368/4.086 \approx 0.090$$

Jacobian: $J_\sigma = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$:

$$J_\sigma = \begin{pmatrix}0.665 & 0 & 0\\ 0 & 0.245 & 0 \\ 0 & 0 & 0.090\end{pmatrix} - \begin{pmatrix}0.665\\ 0.245\\ 0.090\end{pmatrix}\begin{pmatrix}0.665 & 0.245 & 0.090\end{pmatrix}$$

$$= \begin{pmatrix}0.665 - 0.442 & -0.163 & -0.060\\ -0.163 & 0.245 - 0.060 & -0.022\\ -0.060 & -0.022 & 0.090 - 0.008\end{pmatrix} = \begin{pmatrix}0.222 & -0.163 & -0.060\\ -0.163 & 0.185 & -0.022\\ -0.060 & -0.022 & 0.082\end{pmatrix}$$

**Verification:**
- Row sums: $0.222 - 0.163 - 0.060 = -0.001 \approx 0$ ✓ (rounding; exact = 0)
- $J_\sigma\mathbf{1} = 0$ (null space contains $\mathbf{1}$) ✓
- Symmetry: $J_{12} = J_{21} = -0.163$ ✓
- Diagonal = $p_i(1-p_i)$: $0.665 \times 0.335 = 0.223 \approx 0.222$ ✓

**Eigenvalues** (computed numerically): $\lambda_1 \approx 0.487$, $\lambda_2 \approx 0.003$, $\lambda_3 = 0$. Two nonzero eigenvalues (rank 2 = $K-1$ ✓), one zero eigenvalue corresponding to eigenvector $\mathbf{1}/\sqrt{3}$ ✓.


### 15.3 Newton's Method: Traced Convergence on a Quadratic

Take $f(x, y) = \frac{1}{2}(x^2 + 25y^2)$ (condition number $\kappa = 25$). Minimum at $(0, 0)$.

$\nabla f = (x, 25y)^\top$, $H_f = \text{diag}(1, 25)$.

**Gradient descent** with $\eta = 2/(1+25) = 0.0769$:

| $t$ | $x_t$ | $y_t$ | $f(\mathbf{x}_t)$ | $\|(x_t,y_t)\|$ |
|-----|--------|--------|-------------------|-----------------|
| 0 | 1.0 | 1.0 | 13.0 | 1.414 |
| 1 | 0.923 | 0.923 | 11.08 | 1.305 |
| 5 | 0.652 | 0.652 | 5.51 | 0.922 |
| 20 | 0.215 | 0.215 | 0.599 | 0.304 |
| 50 | 0.029 | 0.029 | 0.011 | 0.041 |

Contraction ratio: $\rho = (25-1)/(25+1) \approx 0.923$ — slow!

**Newton's method** with exact Hessian $H^{-1} = \text{diag}(1, 1/25)$:

Newton step: $\boldsymbol{\delta} = -H^{-1}\nabla f = -(x, y)^\top$.

From $(x_0, y_0) = (1, 1)$: Newton step gives $(1-1, 1-1) = (0, 0)$ — **exactly one step!**

This is because $f$ is exactly quadratic: the Newton step minimises the quadratic approximation exactly, and the approximation is exact for quadratic functions. For non-quadratic functions, Newton converges in $O(\log\log(1/\varepsilon))$ steps near the minimum.

**Key insight.** GD takes $\sim 50$ steps to get to error $0.03$; Newton takes 1 step to get to error $0$. The difference is the Hessian inverse which scales the poorly-conditioned direction (the $y$-direction with curvature 25) by $1/25$, compensating exactly.


---

## 16. Summary of Key Formulas

A quick-reference table of all major Jacobians and Hessians covered in this section.

```
JACOBIAN REFERENCE TABLE
════════════════════════════════════════════════════════════════════════

  Function                  | Jacobian J_f                        | Shape
  ──────────────────────────┼─────────────────────────────────────┼────────
  f(x) = Ax + b             | A                                   | (m,n)
  f(x) = σ(x)  elementwise  | diag(σ'(x))                         | (n,n)
  f(x) = softmax(x)         | diag(p) - pp^T                      | (K,K)
  f(x) = LayerNorm(x)       | (1/σ)(I - (1/d)11^T - (1/d)x̂x̂^T) | (d,d)
  f(x) = ||x||₂             | x^T / ||x||₂  (row vector)          | (1,n)
  f(x) = x/||x||₂           | (1/||x||)(I - x̂x̂^T)               | (n,n)
  f(x) = log(sum(exp(x)))   | softmax(x)^T  (row vector)          | (1,n)
  f(x) = ReLU(x)            | diag(1[x > 0])  (a.e.)              | (n,n)
  f(x,y) = xy               | [y, x]  (row Jacobian)              | (1,2)
  Chain: f∘g                | J_f(g(x)) · J_g(x)                  | (m,n)

════════════════════════════════════════════════════════════════════════

HESSIAN REFERENCE TABLE
════════════════════════════════════════════════════════════════════════

  Function f(x)             | Hessian H_f                         | Definiteness
  ──────────────────────────┼─────────────────────────────────────┼────────
  (1/2)x^T A x              | A                                   | same as A
  (1/2)||Ax - b||²          | A^T A                               | PSD
  log-sum-exp(x)            | diag(p) - pp^T                      | PSD, rank K-1
  logistic CE, 1 sample     | p(1-p) xx^T  (rank-1)               | PSD, rank 1
  logistic CE, m samples    | Σᵢ p̂ᵢ(1-p̂ᵢ)xᵢxᵢ^T                | PSD
  f(x) = ||x||²             | 2I                                  | PD
  f(x) = 1/||x||            | (3xx^T - ||x||²I)/||x||⁵           | Indefinite
  ReLU network              | 0 almost everywhere                  | PSD (trivially)

════════════════════════════════════════════════════════════════════════
```

**Cost summary:**

| Computation | Naive cost | Efficient cost |
|-------------|-----------|----------------|
| Full Jacobian $J_f \in \mathbb{R}^{m\times n}$ | $O(mn)$ AD calls | — |
| JVP $J_f\mathbf{v}$ (single vector) | 1 forward pass | $O(n)$ |
| VJP $J_f^\top\mathbf{u}$ (single vector) | 1 backward pass | $O(n)$ |
| Full gradient $\nabla f$ for scalar $f$ | — | 1 backward pass, $O(n)$ |
| Full Hessian $H_f \in \mathbb{R}^{n\times n}$ | $n$ backward passes | $O(n^2)$ |
| HVP $H_f\mathbf{v}$ | — | 2 backward passes, $O(n)$ |
| Top eigenvalue $\lambda_{\max}$ | — | $\sim 20$ HVPs, $O(n)$ each |
| Solve $H_f\mathbf{x} = \mathbf{b}$ (CG) | — | $O(\sqrt{\kappa})$ HVPs |
| K-FAC update per layer | $O(n_{\text{in}}^3 n_{\text{out}}^3)$ | $O(n_{\text{in}}^3 + n_{\text{out}}^3)$ |


---

## References

1. **Spivak, M.** (1965). *Calculus on Manifolds*. Benjamin. — The rigorous treatment of Fréchet derivatives and the inverse/implicit function theorems.

2. **Rumelhart, D., Hinton, G., Williams, R.** (1986). "Learning representations by back-propagating errors." *Nature*, 323. — The original backpropagation paper; backprop = iterated VJP.

3. **Baydin, A., Pearlmutter, B., Radul, A., Siskind, J.** (2018). "Automatic differentiation in machine learning: a survey." *JMLR*, 18(153). — Comprehensive AD survey covering forward/reverse mode and higher-order derivatives.

4. **Pearlmutter, B.** (1994). "Fast exact multiplication by the Hessian." *Neural Computation*, 6(1). — The original HVP identity paper.

5. **Martens, J., Grosse, R.** (2015). "Optimizing neural networks with Kronecker-factored approximate curvature." *ICML*. — K-FAC paper.

6. **Ghorbani, B., Krishnan, S., Xiao, Y.** (2019). "An investigation into neural net optimization via Hessian eigenvalue density." *ICML*. — Hessian spectrum analysis of deep networks.

7. **Cohen, J., Kaur, S., Li, Y., Kolter, J., Talwalkar, A.** (2022). "Gradient descent on neural networks typically occurs at the edge of stability." *ICLR*. — Edge of stability and $\lambda_{\max}$ dynamics.

8. **Foret, P., Kleiner, A., Mobahi, H., Neyshabur, B.** (2021). "Sharpness-Aware Minimization for Efficiently Improving Generalization." *ICLR*. — SAM algorithm.

9. **Hu, E., Shen, Y., Wallis, P., et al.** (2022). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR*. — LoRA; Jacobian perspective on parameter-efficient fine-tuning.

10. **Su, J., Lu, Y., Pan, S., et al.** (2024). "RoFormer: Enhanced transformer with rotary position embedding." *Neurocomputing*. — RoPE; orthogonal Jacobian design for position embeddings.

11. **Kook, Y., Sra, S.** (2024). "Geometric analysis of neural collapse." — Geometric perspective on loss landscape and Hessian structure.

12. **Boyd, S., Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge. — Definitive reference for second-order analysis, Newton's method, and optimality conditions.


---

*End of §02 Jacobians and Hessians*

[← Back to Chapter 05: Multivariate Calculus](../README.md) | [Next: §03 Chain Rule and Backpropagation →](../03-Chain-Rule-and-Backpropagation/notes.md)

