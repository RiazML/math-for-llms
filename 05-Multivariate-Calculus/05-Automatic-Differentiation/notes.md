[← Back to Multivariate Calculus](../README.md)

---

# Automatic Differentiation

> _"Differentiating programs, not formulas — that is the key insight that makes modern deep learning possible."_
> — Automatic differentiation community folklore

## Overview

Automatic differentiation (AD) is the algorithmic machinery that makes gradient-based learning at scale possible. Unlike numerical finite differences (slow, inaccurate) or symbolic differentiation (expression-swell, rigid), AD evaluates exact derivatives of arbitrary computer programs at machine precision, in time proportional to the original computation. Every modern deep-learning framework — PyTorch, JAX, TensorFlow, Flax — is, at its core, an automatic differentiation system wrapped around a linear algebra library.

This section builds AD from the ground up. We begin with dual number algebra (the mathematical foundation of forward-mode AD), derive the adjoint accumulation algorithm (the foundation of reverse-mode AD and backpropagation), and study computational graphs, topological sort, JVP/VJP operations, higher-order derivatives, gradient checkpointing, and custom autograd functions. Throughout, we connect the mathematics to the engineering decisions inside real frameworks.

**Relationship to adjacent sections:** The chain rule and its role in backpropagation were derived in §05/03. Jacobians and Hessians as mathematical objects were studied in §05/02. This section focuses on *how to compute them efficiently for arbitrary programs*. Gradient descent algorithms that *consume* these gradients are covered in §08/02.

## Prerequisites

- Partial derivatives and the chain rule (`§05/01-Partial-Derivatives`)
- Jacobian matrices and directional derivatives (`§05/02-Jacobians-and-Hessians`)
- Backpropagation derivation for neural networks (`§05/03-Chain-Rule-and-Backpropagation`)
- Basic Python and NumPy; familiarity with operator overloading

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive derivations: dual numbers, reverse-mode from scratch, JVP/VJP, higher-order AD, gradient checkpointing, MAML |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises: dual arithmetic through implicit differentiation and meta-learning gradients |

## Learning Objectives

After completing this section, you will:

- Explain why automatic differentiation is superior to finite differences and symbolic differentiation for large-scale ML
- Implement dual number arithmetic and use it to compute exact first derivatives in forward mode
- Derive the adjoint accumulation (reverse-mode) algorithm from first principles and trace it through a worked example
- Construct and topologically sort a computational graph, and execute a backward pass by hand
- Distinguish JVP (forward-mode) and VJP (reverse-mode) and choose the efficient mode for a given Jacobian shape
- Compute Hessian-vector products via nested AD without materialising the full Hessian
- Implement gradient checkpointing and quantify the memory–compute tradeoff
- Write a custom `autograd.Function` with a manually specified VJP
- Explain implicit differentiation and differentiate through fixed-point solvers
- Connect AD to meta-learning (MAML), differentiable programming, and hyperparameter optimisation

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 The Three Ways to Differentiate](#11-the-three-ways-to-differentiate)
  - [1.2 Why AD Matters for AI](#12-why-ad-matters-for-ai)
  - [1.3 Historical Context](#13-historical-context)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Computational Graphs](#21-computational-graphs)
  - [2.2 The Wengert List](#22-the-wengert-list)
  - [2.3 Elementary Operations](#23-elementary-operations)
- [3. Dual Numbers and Forward-Mode AD](#3-dual-numbers-and-forward-mode-ad)
  - [3.1 Dual Number Algebra](#31-dual-number-algebra)
  - [3.2 Derivative Extraction](#32-derivative-extraction)
  - [3.3 Forward-Mode Trace and JVP](#33-forward-mode-trace-and-jvp)
  - [3.4 Forward-Mode Complexity](#34-forward-mode-complexity)
  - [3.5 Implementation from Scratch](#35-implementation-from-scratch)
- [4. Reverse-Mode AD and Backpropagation](#4-reverse-mode-ad-and-backpropagation)
  - [4.1 Adjoint Variables](#41-adjoint-variables)
  - [4.2 Reverse-Mode Trace](#42-reverse-mode-trace)
  - [4.3 VJP — Vector-Jacobian Product](#43-vjp--vector-jacobian-product)
  - [4.4 Reverse-Mode Complexity](#44-reverse-mode-complexity)
  - [4.5 Forward vs Reverse — When to Use Each](#45-forward-vs-reverse--when-to-use-each)
- [5. Computational Graph Mechanics](#5-computational-graph-mechanics)
  - [5.1 DAG Construction](#51-dag-construction)
  - [5.2 Topological Sort](#52-topological-sort)
  - [5.3 Memory Layout — Tape vs Closure](#53-memory-layout--tape-vs-closure)
  - [5.4 In-place Operations and Graph Invalidation](#54-in-place-operations-and-graph-invalidation)
- [6. JVP, VJP, and the Jacobian](#6-jvp-vjp-and-the-jacobian)
  - [6.1 JVP Formal Definition](#61-jvp-formal-definition)
  - [6.2 VJP Formal Definition](#62-vjp-formal-definition)
  - [6.3 Full Jacobian via JVPs and VJPs](#63-full-jacobian-via-jvps-and-vjps)
  - [6.4 Hessian-Vector Products](#64-hessian-vector-products)
- [7. Higher-Order Derivatives](#7-higher-order-derivatives)
  - [7.1 Second Derivatives via create_graph](#71-second-derivatives-via-create_graph)
  - [7.2 Hessian-Vector Products via Nested AD](#72-hessian-vector-products-via-nested-ad)
  - [7.3 Taylor-Mode Forward AD](#73-taylor-mode-forward-ad)
  - [7.4 Cost of Higher-Order AD](#74-cost-of-higher-order-ad)
- [8. Advanced Implementation Topics](#8-advanced-implementation-topics)
  - [8.1 Gradient Checkpointing](#81-gradient-checkpointing)
  - [8.2 Custom Backward Functions](#82-custom-backward-functions)
  - [8.3 Non-Differentiable Operations](#83-non-differentiable-operations)
  - [8.4 Mixed-Precision and Numerical Stability](#84-mixed-precision-and-numerical-stability)
- [9. Applications in Machine Learning](#9-applications-in-machine-learning)
  - [9.1 Training Neural Networks](#91-training-neural-networks)
  - [9.2 Meta-Learning and MAML](#92-meta-learning-and-maml)
  - [9.3 Differentiable Programming](#93-differentiable-programming)
  - [9.4 Implicit Differentiation](#94-implicit-differentiation)
  - [9.5 Gradient-Based Hyperparameter Optimisation](#95-gradient-based-hyperparameter-optimisation)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI (2026 Perspective)](#12-why-this-matters-for-ai-2026-perspective)
- [Conceptual Bridge](#conceptual-bridge)

---

## 1. Intuition

### 1.1 The Three Ways to Differentiate

Given a function $f : \mathbb{R}^n \to \mathbb{R}^m$ implemented as a computer program, there are exactly three systematic approaches to computing its derivative $J_f \in \mathbb{R}^{m \times n}$:

**1. Numerical differentiation (finite differences):**

$$\frac{\partial f_i}{\partial x_j} \approx \frac{f_i(\mathbf{x} + h\mathbf{e}_j) - f_i(\mathbf{x})}{h}$$

This requires $n$ forward evaluations (one per input dimension) and suffers from two fundamental problems. First, truncation error: the finite-difference formula approximates the derivative with $O(h)$ error. Second, cancellation error: for small $h$, subtracting two nearly equal floating-point numbers destroys significant digits. The sweet spot $h \approx \sqrt{\epsilon_{\text{machine}}} \approx 10^{-8}$ still gives only $O(10^{-8})$ relative accuracy — barely enough to verify a gradient, never enough to use as a training signal over millions of steps. Cost: $O(n)$ function evaluations.

**2. Symbolic differentiation:**

Apply the rules of calculus to the *symbolic expression* for $f$, producing a new symbolic expression for $\nabla f$. This gives exact derivatives and is the approach taken by computer algebra systems (Mathematica, SymPy). The fatal flaw for ML is **expression swell**: differentiating a composition of $n$ functions can produce a symbolic expression exponentially larger than the original. A 100-layer network cannot be symbolically differentiated — the resulting formula would be astronomically large. Cost: $O(\text{expression size})$, which can be $O(n!)$ in the worst case.

**3. Automatic differentiation:**

AD applies the chain rule mechanically at the level of *elementary operations* ($+$, $\times$, $\sin$, $\exp$, etc.), tracking either tangent vectors (forward mode) or adjoint variables (reverse mode). It produces exact derivatives at machine precision, in time $O(c \cdot T(f))$ where $T(f)$ is the runtime of $f$ and $c$ is a small constant (typically $c \leq 5$). No symbolic expression is ever formed; derivatives are computed numerically as the program executes.

```
COMPARISON: THREE DIFFERENTIATION METHODS
════════════════════════════════════════════════════════════════════════

  Method          | Accuracy   | Cost (grad) | Scales to   | Used for
  ─────────────────────────────────────────────────────────────────────
  Finite Diff.    | ~1e-8 rel  | O(n) evals  | n < 1000    | Grad check
  Symbolic        | Exact      | Expression  | n < 20      | Analysis
                  |            | swell: bad  |             |
  AD (forward)    | Exact FP   | O(n) × T(f) | n ≤ m       | JVPs, physics
  AD (reverse)    | Exact FP   | O(1) × T(f) | n >> m      | Training NNs

════════════════════════════════════════════════════════════════════════
```

The $O(1) \times T(f)$ cost of reverse-mode AD for scalar outputs is the mathematical miracle underlying all of deep learning. A network with $n = 10^{11}$ parameters (GPT-4 scale) can have its full gradient computed in roughly the time of two forward passes — regardless of $n$.

### 1.2 Why AD Matters for AI

Every training step of every neural network in production today uses reverse-mode automatic differentiation. The gradient $\nabla_\theta \mathcal{L}$ of a scalar loss $\mathcal{L}$ with respect to billions of parameters $\theta$ is computed by a single backward pass through the computational graph — at a cost roughly equal to the forward pass. This would be impossible with finite differences (cost $O(n)$ evaluations) or symbolic differentiation (intractable).

Beyond simple training, AD enables:

- **Meta-learning (MAML):** Differentiating through an inner gradient descent loop requires computing second-order derivatives — gradients of gradients. AD handles this cleanly via `create_graph=True`.
- **Differentiable physics and rendering:** Gradients of physical simulations allow optimising 3D scenes, fluid dynamics, or molecular configurations end-to-end.
- **Neural ODEs:** Treating the dynamics of a neural network as a continuous ODE and backpropagating through the ODE solver via the adjoint method.
- **Differentiable optimisation layers:** Embedding QP or LP solvers inside a network and differentiating through their solutions via implicit differentiation.
- **Gradient checkpointing:** Trading compute for memory — recomputing activations during the backward pass instead of storing them, enabling training of models that would otherwise overflow GPU memory.

**For AI (2026):** JAX's `jit`, `vmap`, and `grad` transformations compose AD with JIT compilation and vectorisation, enabling research at scales (and with flexibility) that were impractical with earlier frameworks. The `torch.compile` stack in PyTorch 2.x similarly relies on AD being a first-class algebraic transformation.

### 1.3 Historical Context

The mathematical foundations of AD stretch back further than most practitioners realise:

| Year | Contributor | Contribution |
|------|-------------|--------------|
| 1964 | R. E. Wengert | First published algorithm for automatic derivative evaluation using a "Wengert list" (tape) |
| 1970 | Linnainmaa | Forward and reverse accumulation for floating-point function evaluation |
| 1980 | Speelpenning | Efficient reverse-mode AD for composite functions (thesis) |
| 1981 | Griewank & Walther | Systematic development of AD theory; `ADIFOR` system |
| 1986 | Rumelhart, Hinton, Williams | Popularised reverse-mode AD as "backpropagation" in neural network context |
| 1992 | Griewank | `ADOL-C` — first widely-used AD tool for C/C++ |
| 2015 | Maclaurin, Duvenaud, Adams | `Autograd` — differentiable NumPy, enabling research experimentation |
| 2016 | PyTorch team (Paszke et al.) | Dynamic computational graphs ("define-by-run") revolutionise AD usability |
| 2018 | Google Brain (Bradbury et al.) | `JAX` — composable function transformations: `grad`, `jit`, `vmap`, `pmap` |
| 2023+ | PyTorch 2.x | `torch.compile` integrates AD with graph-level optimisation |

The key conceptual leap in PyTorch was switching from *static* (TensorFlow 1.x) to *dynamic* computational graphs: the graph is built on-the-fly as Python executes, making debugging, control flow, and research iteration far more natural.

---

## 2. Formal Definitions

### 2.1 Computational Graphs

**Definition 2.1 (Computational graph).** A *computational graph* for a function $f : \mathbb{R}^n \to \mathbb{R}^m$ is a directed acyclic graph $G = (V, E)$ where:

- Each *leaf node* $v_i$ for $i = 1, \ldots, n$ represents an input variable $x_i$
- Each *internal node* $v_k$ represents an intermediate computed value $v_k = \phi_k(v_{p_1}, \ldots, v_{p_r})$ where $\phi_k$ is an elementary operation and $p_1, \ldots, p_r$ are the indices of its parent nodes
- Each *output node* $v_j$ for $j$ in the output set represents a component $f_j$ of the output
- A directed edge $(v_i, v_k)$ exists if $v_i$ is a direct input to the computation of $v_k$

The graph is acyclic because a value cannot depend on itself (no cycles in time-ordered computation). The topological ordering of nodes corresponds to the order in which operations are executed.

**Example.** For $f(x_1, x_2) = (x_1 + x_2) \cdot \sin(x_1)$:

```
COMPUTATIONAL GRAPH EXAMPLE: f(x₁, x₂) = (x₁ + x₂) · sin(x₁)
════════════════════════════════════════════════════════════════════════

  Inputs:   v₁ = x₁,  v₂ = x₂

  v₃ = v₁ + v₂          (addition)
  v₄ = sin(v₁)          (unary sine)
  v₅ = v₃ × v₄          (multiplication)  ← output

  Graph edges:
    v₁ ──→ v₃,  v₂ ──→ v₃
    v₁ ──→ v₄
    v₃ ──→ v₅,  v₄ ──→ v₅

  Topological order: v₁, v₂, v₃, v₄, v₅   (or v₁, v₂, v₄, v₃, v₅)

════════════════════════════════════════════════════════════════════════
```

### 2.2 The Wengert List

The *Wengert list* (also called the *tape*) is a linearisation of the computational graph — a sequential record of all elementary operations performed during a forward evaluation, along with the intermediate values needed to compute derivatives.

**Definition 2.2 (Wengert list).** For a forward evaluation of $f$ at $\mathbf{x}$, the Wengert list $W$ is the sequence:

$$W = \bigl[(v_1, x_1),\ (v_2, x_2),\ \ldots,\ (v_k, \phi_k(v_{p_1}, \ldots, v_{p_r}),\ \text{inputs}_{p_1,\ldots,p_r}),\ \ldots\bigr]$$

Each entry records: the variable index, the operation performed, the indices of input variables, and (for reverse-mode) the values needed to compute the local Jacobian of $\phi_k$.

In PyTorch, this tape is implicit in the `grad_fn` chain attached to each tensor with `requires_grad=True`. In JAX, it is made explicit through the function transformation machinery (`jax.make_jaxpr`).

**Example Wengert list** for $f(x_1, x_2) = (x_1 + x_2) \cdot \sin(x_1)$ at $(x_1, x_2) = (1, 2)$:

| Step | Variable | Operation | Value | Saved for backward |
|------|----------|-----------|-------|--------------------|
| 1 | $v_1$ | input | 1.0 | — |
| 2 | $v_2$ | input | 2.0 | — |
| 3 | $v_3$ | $v_1 + v_2$ | 3.0 | — (linear) |
| 4 | $v_4$ | $\sin(v_1)$ | 0.8415 | $v_1 = 1.0$ (to compute $\cos v_1$) |
| 5 | $v_5$ | $v_3 \times v_4$ | 2.524 | $v_3 = 3.0,\ v_4 = 0.8415$ |

### 2.3 Elementary Operations

AD works by decomposing every computation into *elementary operations* $\phi : \mathbb{R}^k \to \mathbb{R}$ whose derivatives are known analytically. The chain rule then assembles these into derivatives of the full computation.

**Standard elementary operations and their derivatives:**

| Operation $\phi(u, v)$ | Value | $\partial\phi/\partial u$ | $\partial\phi/\partial v$ |
|---|---|---|---|
| $u + v$ | $u + v$ | $1$ | $1$ |
| $u - v$ | $u - v$ | $1$ | $-1$ |
| $u \cdot v$ | $uv$ | $v$ | $u$ |
| $u / v$ | $u/v$ | $1/v$ | $-u/v^2$ |
| $u^a$ (const $a$) | $u^a$ | $au^{a-1}$ | — |
| $\exp(u)$ | $e^u$ | $e^u$ | — |
| $\log(u)$ | $\ln u$ | $1/u$ | — |
| $\sin(u)$ | $\sin u$ | $\cos u$ | — |
| $\cos(u)$ | $\cos u$ | $-\sin u$ | — |
| $\max(u, 0)$ | ReLU | $\mathbf{1}[u > 0]$ | — |
| $\sigma(u)$ | $1/(1+e^{-u})$ | $\sigma(u)(1-\sigma(u))$ | — |

Any differentiable function implemented as a composition of these operations (and control flow that doesn't depend on the differentiated variable) can be automatically differentiated by AD.

**Non-smooth operations.** Functions like ReLU are not differentiable at $u = 0$, but they are differentiable almost everywhere. AD frameworks return a *subgradient* at non-smooth points (typically 0 for ReLU at 0, following Clarke's generalised gradient), which works well in practice because the probability of hitting the non-smooth point exactly in floating-point arithmetic is effectively zero.

---

## 3. Dual Numbers and Forward-Mode AD

### 3.1 Dual Number Algebra

Dual numbers provide the cleanest algebraic foundation for forward-mode automatic differentiation. They extend the reals in a way that carries derivative information alongside function values.

**Definition 3.1 (Dual numbers).** The ring of *dual numbers* is:
$$\mathbb{D} = \{a + b\varepsilon : a, b \in \mathbb{R},\ \varepsilon^2 = 0,\ \varepsilon \neq 0\}$$

The symbol $\varepsilon$ is the *dual unit* — an infinitesimal that satisfies $\varepsilon^2 = 0$ but $\varepsilon \neq 0$. This is not a real number; it is an algebraic extension analogous to how $i^2 = -1$ extends $\mathbb{R}$ to $\mathbb{C}$.

**Arithmetic rules.** For dual numbers $\mathbf{a} = a + a'\varepsilon$ and $\mathbf{b} = b + b'\varepsilon$:

$$\mathbf{a} + \mathbf{b} = (a + b) + (a' + b')\varepsilon$$
$$\mathbf{a} \cdot \mathbf{b} = ab + (ab' + a'b)\varepsilon \quad \text{(since } \varepsilon^2 = 0\text{)}$$
$$\mathbf{a} / \mathbf{b} = \frac{a}{b} + \frac{a'b - ab'}{b^2}\varepsilon \quad (b \neq 0)$$

These are the familiar sum, product, and quotient rules for derivatives! The dual part tracks the derivative automatically.

**Extending scalar functions.** For a smooth function $f : \mathbb{R} \to \mathbb{R}$, its dual number extension is:

$$f(a + a'\varepsilon) = f(a) + a' f'(a)\varepsilon$$

This follows from the Taylor expansion truncated at first order: $f(a + a'\varepsilon) = f(a) + f'(a)(a'\varepsilon) + \frac{1}{2}f''(a)(a'\varepsilon)^2 + \cdots = f(a) + a'f'(a)\varepsilon$ (since $\varepsilon^2 = 0$, all higher terms vanish).

**Examples:**
- $\exp(a + a'\varepsilon) = e^a + a' e^a \varepsilon = e^a(1 + a'\varepsilon)$
- $\sin(a + a'\varepsilon) = \sin(a) + a'\cos(a)\varepsilon$
- $\log(a + a'\varepsilon) = \log(a) + (a'/a)\varepsilon$

### 3.2 Derivative Extraction

**Theorem 3.1 (Derivative via dual numbers).** To compute $f'(a)$ for $f : \mathbb{R} \to \mathbb{R}$:

1. Evaluate $f$ at the dual number $a + 1 \cdot \varepsilon$ (setting $a' = 1$)
2. The dual part of the result is $f'(a)$

$$f(a + \varepsilon) = f(a) + f'(a)\varepsilon \implies f'(a) = \text{dual\_part}(f(a + \varepsilon))$$

**Example.** Compute the derivative of $f(x) = x^2 \sin(x)$ at $x = 1$:

$$f(1 + \varepsilon) = (1+\varepsilon)^2 \cdot \sin(1+\varepsilon)$$
$$= (1 + 2\varepsilon) \cdot (\sin 1 + \cos 1 \cdot \varepsilon)$$
$$= \sin 1 + (2\sin 1 + \cos 1)\varepsilon$$

So $f'(1) = 2\sin(1) + \cos(1) \approx 2(0.8415) + 0.5403 = 2.223$.

Verification: $f'(x) = 2x\sin x + x^2 \cos x$, so $f'(1) = 2\sin 1 + \cos 1$. ✓

**For $f : \mathbb{R}^n \to \mathbb{R}^m$, partial derivatives.** To compute $\partial f_i / \partial x_j$, evaluate $f$ at $\mathbf{x} + \mathbf{e}_j \varepsilon$ (seeding the $j$-th input with dual part 1, all others 0). The dual part of the $i$-th output gives $\partial f_i / \partial x_j$. This requires $n$ evaluations to fill the full Jacobian.

### 3.3 Forward-Mode Trace and JVP

Forward-mode AD computes the *Jacobian-vector product* (JVP):

$$\text{jvp}(f, \mathbf{x}, \dot{\mathbf{x}}) = J_f(\mathbf{x})\, \dot{\mathbf{x}} \in \mathbb{R}^m$$

where $\dot{\mathbf{x}} \in \mathbb{R}^n$ is a *tangent vector* (also called the *seed vector* or *perturbation direction*).

**Trace table for forward mode.** For each variable $v_k$ in the Wengert list, maintain a pair $(v_k, \dot{v}_k)$ where $\dot{v}_k = dv_k / d\text{seed}$ is the tangent propagated through the computation. The tangent $\dot{v}_k$ evolves according to the local Jacobian:

$$\dot{v}_k = \sum_{j : v_j \to v_k} \frac{\partial \phi_k}{\partial v_j} \cdot \dot{v}_j$$

**Worked example.** $f(x_1, x_2) = (x_1 + x_2) \cdot \sin(x_1)$, at $(1, 2)$ with tangent $\dot{\mathbf{x}} = (1, 0)^\top$ (computing $\partial f / \partial x_1$):

| Variable | Operation | Value | Tangent $\dot{v}$ | Rule |
|---|---|---|---|---|
| $v_1$ | input $x_1$ | 1.0 | 1.0 | seed = 1 |
| $v_2$ | input $x_2$ | 2.0 | 0.0 | seed = 0 |
| $v_3$ | $v_1 + v_2$ | 3.0 | 1.0 | $\dot{v}_1 + \dot{v}_2$ |
| $v_4$ | $\sin(v_1)$ | 0.8415 | $\cos(1) \cdot 1 = 0.5403$ | $\cos(v_1)\dot{v}_1$ |
| $v_5$ | $v_3 \times v_4$ | 2.524 | $v_3 \dot{v}_4 + v_4 \dot{v}_3 = 3(0.5403) + 0.8415(1) = 2.463$ | product rule |

So $\partial f / \partial x_1 = 2.463$. Using $\dot{\mathbf{x}} = (0, 1)^\top$ in a second pass gives $\partial f / \partial x_2$.

### 3.4 Forward-Mode Complexity

**Theorem 3.2 (Forward-mode cost).** For $f : \mathbb{R}^n \to \mathbb{R}^m$:
- One forward pass with a fixed tangent $\dot{\mathbf{x}}$ computes $J_f \dot{\mathbf{x}}$ in time $O(T(f))$
- Computing the full $m \times n$ Jacobian requires $n$ passes (one per basis vector $\mathbf{e}_j$): time $O(n \cdot T(f))$

**For AI:** Forward mode is efficient when $n \ll m$ — i.e., few inputs and many outputs. This applies in:
- Physics simulations with a few control parameters and many state outputs
- Sensitivity analysis: how does one input affect all outputs?
- Neural tangent kernel computations (batched JVPs)

For training neural networks where $n = |\theta| \gg 1$ and $m = 1$ (scalar loss), forward mode would cost $O(|\theta|)$ — completely impractical. Reverse mode handles this in $O(1)$.

### 3.5 Implementation from Scratch

A minimal dual number implementation in Python demonstrates the full forward-mode machinery:

```python
class Dual:
    """Dual number a + b*eps, eps^2 = 0."""
    def __init__(self, real, dual=0.0):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Dual(self.real + other, self.dual)
        return Dual(self.real + other.real, self.dual + other.dual)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Dual(self.real * other, self.dual * other)
        return Dual(self.real * other.real,
                    self.real * other.dual + self.dual * other.real)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Dual(self.real / other, self.dual / other)
        r = other.real
        return Dual(self.real / r,
                    (self.dual * r - self.real * other.dual) / (r * r))

    def __repr__(self):
        return f"Dual({self.real:.6f} + {self.dual:.6f}ε)"

import math

def d_sin(x):
    return Dual(math.sin(x.real), x.dual * math.cos(x.real))

def d_cos(x):
    return Dual(math.cos(x.real), -x.dual * math.sin(x.real))

def d_exp(x):
    e = math.exp(x.real)
    return Dual(e, x.dual * e)

def d_log(x):
    return Dual(math.log(x.real), x.dual / x.real)

# Compute df/dx₁ for f(x₁, x₂) = (x₁ + x₂) * sin(x₁)
def f(x1, x2):
    return (x1 + x2) * d_sin(x1)

x1 = Dual(1.0, 1.0)  # seed dx₁ = 1
x2 = Dual(2.0, 0.0)  # seed dx₂ = 0
result = f(x1, x2)
print(f"f(1,2) = {result.real:.6f}")
print(f"df/dx₁ = {result.dual:.6f}")  # → 2.463
```

This 40-line implementation is functionally equivalent to the forward-mode core of professional AD libraries. The key insight: by *overloading arithmetic operators*, we make any Python function automatically differentiable — no source code transformation required.

---

## 4. Reverse-Mode AD and Backpropagation

### 4.1 Adjoint Variables

Reverse-mode AD introduces *adjoint variables* (also called *cotangent variables* or *bar variables*) that propagate gradient information backward through the computational graph.

**Definition 4.1 (Adjoint variable).** For a function $f : \mathbb{R}^n \to \mathbb{R}$ (scalar output) and intermediate variable $v_k$ in its computation, the adjoint is:

$$\bar{v}_k = \frac{\partial f}{\partial v_k}$$

The adjoints satisfy a recursive relation that runs *backward* through the graph:

$$\bar{v}_k = \sum_{j : v_k \to v_j} \bar{v}_j \cdot \frac{\partial v_j}{\partial v_k}$$

This is the chain rule applied in reverse: the adjoint of $v_k$ accumulates contributions from all nodes $v_j$ that depend on $v_k$, weighted by the local Jacobian $\partial v_j / \partial v_k$.

**Initialisation:** Set $\bar{v}_{\text{output}} = 1$ (since $\partial f / \partial f = 1$). Then propagate backward through the Wengert list in reverse order.

**The adjoint accumulation algorithm:**

```
ADJOINT ACCUMULATION (Reverse-Mode AD)
════════════════════════════════════════════════════════════════════════

  Forward pass:
    For k = 1, 2, ..., K:
      Evaluate v_k = φ_k(v_{p₁}, ..., v_{pᵣ})
      Store v_k and the inputs {v_{pⱼ}} needed for backward

  Backward pass:
    Initialise v̄_K = 1 (output node)
    For k = K, K-1, ..., 1:
      For each parent j of v_k:
        v̄_j += v̄_k · (∂φ_k / ∂v_j)   [local VJP]

  Result: v̄₁, ..., v̄ₙ = ∂f/∂x₁, ..., ∂f/∂xₙ   (full gradient)

════════════════════════════════════════════════════════════════════════
```

### 4.2 Reverse-Mode Trace

**Worked example.** $f(x_1, x_2) = (x_1 + x_2) \cdot \sin(x_1)$ at $(1, 2)$.

**Forward pass** (same as before):

| Variable | Value |
|---|---|
| $v_1 = x_1$ | 1.0 |
| $v_2 = x_2$ | 2.0 |
| $v_3 = v_1 + v_2$ | 3.0 |
| $v_4 = \sin(v_1)$ | 0.8415 |
| $v_5 = v_3 \times v_4$ | 2.524 |

**Backward pass** (initialise $\bar{v}_5 = 1$):

| Variable | Adjoint update | Value |
|---|---|---|
| $\bar{v}_5$ | initialise | 1.0 |
| $\bar{v}_4$ | $\bar{v}_5 \cdot v_3 = 1.0 \times 3.0$ | 3.0 |
| $\bar{v}_3$ | $\bar{v}_5 \cdot v_4 = 1.0 \times 0.8415$ | 0.8415 |
| $\bar{v}_1$ from $v_4$: | $\bar{v}_4 \cdot \cos(v_1) = 3.0 \times 0.5403$ | += 1.621 |
| $\bar{v}_2$ from $v_3$: | $\bar{v}_3 \cdot 1 = 0.8415$ | 0.8415 |
| $\bar{v}_1$ from $v_3$: | $\bar{v}_3 \cdot 1 = 0.8415$ | += 0.8415 |

**Final adjoints:** $\bar{v}_1 = 1.621 + 0.8415 = 2.463 = \partial f/\partial x_1$, $\bar{v}_2 = 0.8415 = \partial f/\partial x_2$.

In a **single backward pass** we computed both partial derivatives simultaneously. Compare forward mode: two separate passes were needed.

### 4.3 VJP — Vector-Jacobian Product

The fundamental operation of reverse-mode AD is the **VJP (vector-Jacobian product)**:

$$\text{vjp}(f, \mathbf{x}, \mathbf{v}) = \mathbf{v}^\top J_f(\mathbf{x}) \in \mathbb{R}^{1 \times n}$$

where $\mathbf{v} \in \mathbb{R}^m$ is a *cotangent vector* (or *costate*). When $f$ is scalar and $\mathbf{v} = 1$, this reduces to $\nabla f(\mathbf{x})^\top$.

Each elementary operation $\phi_k : \mathbb{R}^{r} \to \mathbb{R}$ defines a local VJP:

$$\text{vjp}(\phi_k, \mathbf{u}, \bar{v}) = \bar{v} \cdot \nabla_\mathbf{u} \phi_k(\mathbf{u})$$

The reverse-mode pass chains these local VJPs together via the chain rule, accumulating cotangents into the input adjoints.

**VJP rules for elementary operations:**

| Operation $v = \phi(u_1, u_2)$ | VJP rule |
|---|---|
| $v = u_1 + u_2$ | $(\bar{u}_1, \bar{u}_2) \mathrel{+}= (\bar{v}, \bar{v})$ |
| $v = u_1 \cdot u_2$ | $(\bar{u}_1, \bar{u}_2) \mathrel{+}= (\bar{v} u_2, \bar{v} u_1)$ |
| $v = \exp(u)$ | $\bar{u} \mathrel{+}= \bar{v} \cdot e^u$ |
| $v = \log(u)$ | $\bar{u} \mathrel{+}= \bar{v} / u$ |
| $v = \sin(u)$ | $\bar{u} \mathrel{+}= \bar{v} \cdot \cos(u)$ |
| $v = Au$ (matrix-vector) | $\bar{u} \mathrel{+}= A^\top \bar{v}$ |
| $v = uA$ | $\bar{u} \mathrel{+}= \bar{v} A$ |
| $v = u^\top w$ | $\bar{u} \mathrel{+}= \bar{v} w,\ \bar{w} \mathrel{+}= \bar{v} u$ |
| $v = \text{ReLU}(u)$ | $\bar{u} \mathrel{+}= \bar{v} \cdot \mathbf{1}[u > 0]$ |

The $\mathrel{+}=$  notation is important: adjoints accumulate across all paths through the graph (multiple uses of the same variable contribute multiple terms).

### 4.4 Reverse-Mode Complexity

**Theorem 4.1 (Reverse-mode cost).** For $f : \mathbb{R}^n \to \mathbb{R}^m$:
- One reverse pass computes $\mathbf{v}^\top J_f(\mathbf{x})$ (a VJP) in time $O(T(f))$
- Computing the full $m \times n$ Jacobian requires $m$ reverse passes: time $O(m \cdot T(f))$
- For scalar $f$ ($m = 1$): the full gradient $\nabla f \in \mathbb{R}^n$ costs $O(T(f))$ — **independent of $n$**

This is the **cheap gradient principle**: reverse-mode AD computes gradients of scalar functions in constant overhead relative to the function itself. For a network with $10^{10}$ parameters computing a single scalar loss, the gradient costs roughly $3 \times T(f)$ — not $10^{10} \times T(f)$.

**Memory cost.** The backward pass requires access to the intermediate values stored during the forward pass. For a computation with $K$ intermediate variables, the memory cost is $O(K)$. For a deep network with $L$ layers each storing activations of size $d$, this is $O(Ld)$ — significant for large models, motivating gradient checkpointing (§8.1).

### 4.5 Forward vs Reverse — When to Use Each

The choice between forward and reverse mode depends on the shape of the Jacobian:

```
FORWARD vs REVERSE MODE — DECISION GUIDE
════════════════════════════════════════════════════════════════════════

  f : ℝⁿ → ℝᵐ,  Jacobian J ∈ ℝᵐˣⁿ

  Forward mode (JVP):                Reverse mode (VJP):
  ─ Cost: O(n · T(f))                ─ Cost: O(m · T(f))
  ─ Use when n << m                  ─ Use when m << n
  ─ Computes J·ṽ for a tangent ṽ     ─ Computes v̄ᵀ·J for a cotangent v̄

  Examples:                          Examples:
  ─ n=1, m=large: ODE sensitivity    ─ n=large, m=1: NN training
  ─ Physics: few params, many states ─ Any scalar loss minimisation
  ─ JVPs in NTK computation          ─ HVPs (see §6.4)

  Mixed mode: for fat Jacobians      Mixed mode: partition inputs/outputs
  (n < m) use forward; for tall      and interleave forward/reverse
  (n > m) use reverse; square: tie   passes optimally

════════════════════════════════════════════════════════════════════════
```

**The crossover point.** Forward mode is preferable when $n < m$ (Jacobian has fewer columns than rows — a "tall" Jacobian). Reverse mode is preferable when $m < n$ (fewer rows — "fat" Jacobian). For square Jacobians ($m = n$), either mode costs $O(n \cdot T(f))$, and the constant factor determines the winner.

**For AI:** Nearly all gradient computations in ML involve a scalar loss ($m = 1$) and a large parameter space ($n = |\theta| \gg 1$). Reverse mode is unambiguously correct for training. Forward mode appears in:
- Computing JVPs for the Neural Tangent Kernel
- Hessian-vector products via mixed forward/reverse (§6.4)
- Sensitivity analysis in differentiable simulators

---

## 5. Computational Graph Mechanics

### 5.1 DAG Construction

Modern AD frameworks build computational graphs *dynamically* as operations are executed — the "define-by-run" or "eager execution" model introduced by PyTorch (Chainer was earlier). This contrasts with the *static graph* approach of TensorFlow 1.x, where you first declare the computation graph symbolically and then execute it.

**PyTorch's dynamic graph construction.** When a tensor with `requires_grad=True` participates in an operation, PyTorch:
1. Allocates a new output tensor
2. Attaches a `grad_fn` object to the output that knows:
   - Which operation was performed
   - References to the input tensors (or their data, for memory efficiency)
   - The VJP rule for this operation
3. Increments reference counts to keep inputs alive until the backward pass completes

The `grad_fn` objects form an implicit linked list (the computational graph), rooted at the output tensor.

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = torch.tensor([3.0, 4.0], requires_grad=True)
z = x @ y          # dot product
w = torch.sin(z)   # sine

print(w.grad_fn)            # <SinBackward0 object>
print(w.grad_fn.next_functions)  # ((DotBackward0, 0),)
print(w.grad_fn.next_functions[0][0].next_functions)
# ((AccumulateGrad, 0), (AccumulateGrad, 0)) ← leaves
```

**JAX's functional approach.** JAX uses *function transformations* rather than mutable graph construction. `jax.grad(f)(x)` traces `f` to build a *jaxpr* (a functional IR), then executes the reverse-mode pass. This is more amenable to JIT compilation (`jit`) and vectorisation (`vmap`) but requires functions to be pure (no side effects).

### 5.2 Topological Sort

The backward pass must process nodes in *reverse topological order* — each node must be processed before any of its dependencies in the forward graph (equivalently, after all of its successors in the forward graph).

**Kahn's algorithm** for topological sort:

```python
from collections import defaultdict, deque

def topological_sort(graph):
    """
    graph: dict mapping node -> list of nodes it depends on
    Returns: list of nodes in topological order (dependencies first)
    """
    in_degree = defaultdict(int)
    for node, deps in graph.items():
        if node not in in_degree:
            in_degree[node] = 0
        for dep in deps:
            in_degree[node] += 1

    queue = deque(n for n, d in in_degree.items() if d == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for dependent in reverse_graph[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    return order
```

In practice, PyTorch uses a simpler approach: DFS post-order traversal from the output tensor, following `grad_fn` links. The `retain_graph=False` default means the graph is freed after one backward pass (the links are broken, releasing memory).

**Why topological order matters.** When a variable $v_k$ is used multiple times in the computation (e.g., a residual connection), its adjoint $\bar{v}_k$ must accumulate contributions from *all* downstream uses before being propagated to $v_k$'s own parents. Processing in reverse topological order guarantees this: all successors of $v_k$ are processed before $v_k$ itself.

### 5.3 Memory Layout — Tape vs Closure

Two main implementation strategies exist for storing the information needed by the backward pass:

**Tape-based (Wengert list).** The forward pass records each operation in a sequential list (the tape), storing all values needed for the backward VJP computation. The backward pass replays the tape in reverse. This is the approach used by early AD systems and underlies TensorFlow 2.x's `GradientTape`.

**Closure-based.** Each operation creates a closure (a function with captured local state) that computes the VJP. These closures are linked together via the `grad_fn` chain. The backward pass executes these closures in topological order. PyTorch uses this approach — it is more memory-efficient for non-sequential graphs and allows graph pruning.

**Comparison:**

| Aspect | Tape | Closure |
|---|---|---|
| Memory pattern | Sequential array | Linked list of closures |
| Graph structure | Linear | Arbitrary DAG |
| Memory freed | All at once | Per-node as backward proceeds |
| Good for | Simple loops | Arbitrary control flow |

### 5.4 In-place Operations and Graph Invalidation

**The problem.** In-place operations (e.g., `x += 1` in PyTorch, `x.add_(1)`) modify tensor data without creating a new tensor. This corrupts the computational graph: if an earlier `grad_fn` saved a reference to `x`'s data to compute a VJP, but `x` has been modified in-place, the stored value is stale.

PyTorch detects most in-place violations at runtime:

```python
x = torch.randn(3, requires_grad=True)
y = x * 2
x.add_(1)   # in-place modification of a leaf that requires grad
y.backward(torch.ones(3))   # RuntimeError: leaf variable modified in-place
```

**Rules:**
1. Never modify a leaf tensor with `requires_grad=True` in-place
2. Never modify any tensor in-place if it is needed for a backward computation that has not yet been executed
3. In-place operations are allowed for intermediate results *only if* they are not needed for any future VJP (not on the gradient tape)

**In practice,** this means avoiding `+=`, `-=`, `*=`, `[:] =` on any tensor that is part of a computation graph. Use out-of-place alternatives (`x = x + 1` instead of `x += 1`).

---

## 6. JVP, VJP, and the Jacobian

### 6.1 JVP Formal Definition

**Definition 6.1 (JVP — Jacobian-vector product).** For $f : \mathbb{R}^n \to \mathbb{R}^m$ and a tangent vector $\dot{\mathbf{x}} \in \mathbb{R}^n$, the JVP at $\mathbf{x}$ is:

$$\text{jvp}(f, \mathbf{x}, \dot{\mathbf{x}}) = J_f(\mathbf{x})\, \dot{\mathbf{x}} \in \mathbb{R}^m$$

This is the directional derivative of $f$ in the direction $\dot{\mathbf{x}}$: the first-order change in $f(\mathbf{x})$ as $\mathbf{x}$ moves in direction $\dot{\mathbf{x}}$.

**Geometric interpretation.** If $\mathbf{x}(t)$ is a curve in input space with $\mathbf{x}(0) = \mathbf{x}$ and $\dot{\mathbf{x}}(0) = \dot{\mathbf{x}}$, then $\text{jvp}(f, \mathbf{x}, \dot{\mathbf{x}}) = \frac{d}{dt}[f(\mathbf{x}(t))]|_{t=0}$ — the velocity of the output curve.

**In JAX,** JVPs are first-class:
```python
import jax
from jax import jvp
import jax.numpy as jnp

def f(x): return jnp.array([x[0]**2, jnp.sin(x[1])])

x = jnp.array([1.0, 2.0])
v = jnp.array([1.0, 0.0])   # tangent: seed x₁ direction
primals, tangents = jvp(f, (x,), (v,))
# tangents = J_f(x) @ v = [2*x[0]*1, cos(x[1])*0] = [2.0, 0.0]
```

### 6.2 VJP Formal Definition

**Definition 6.2 (VJP — vector-Jacobian product).** For $f : \mathbb{R}^n \to \mathbb{R}^m$ and a cotangent vector $\mathbf{v} \in \mathbb{R}^m$ (costate), the VJP at $\mathbf{x}$ is:

$$\text{vjp}(f, \mathbf{x}, \mathbf{v}) = J_f(\mathbf{x})^\top \mathbf{v} \in \mathbb{R}^n$$

**Duality between JVP and VJP.** The JVP maps tangent vectors (in the input's tangent space) to tangent vectors in the output's tangent space. The VJP maps cotangent vectors (in the output's cotangent space — dual to the tangent space) to the input's cotangent space. They are dual operations:

$$\langle \mathbf{v},\, J_f(\mathbf{x})\, \dot{\mathbf{x}} \rangle = \langle J_f(\mathbf{x})^\top \mathbf{v},\, \dot{\mathbf{x}} \rangle$$

for all $\mathbf{v} \in \mathbb{R}^m$, $\dot{\mathbf{x}} \in \mathbb{R}^n$. This duality is the reason the transpose appears: VJP is the adjoint of JVP.

**In JAX:**
```python
from jax import vjp
primals, vjp_fn = vjp(f, x)
cotangent = jnp.array([1.0, 1.0])   # v = [1, 1]
grad, = vjp_fn(cotangent)
# grad = J_f(x).T @ v
```

### 6.3 Full Jacobian via JVPs and VJPs

When the full $m \times n$ Jacobian is needed:
- **Via JVPs:** $n$ passes, each with $\dot{\mathbf{x}} = \mathbf{e}_j$, gives all columns: $J_{:,j} = \text{jvp}(f, \mathbf{x}, \mathbf{e}_j)$
- **Via VJPs:** $m$ passes, each with $\mathbf{v} = \mathbf{e}_i$, gives all rows: $J_{i,:}^\top = \text{vjp}(f, \mathbf{x}, \mathbf{e}_i)$

For the full Jacobian, choose the mode with smaller dimension ($\min(m, n)$ passes).

**In JAX,** `jax.jacobian` automatically selects the efficient mode:
```python
from jax import jacobian
J = jacobian(f)(x)   # shape (m, n)
# JAX selects forward mode if n < m, reverse if m < n
```

**Batched JVPs via `vmap`.** Computing multiple JVPs simultaneously:
```python
from jax import vmap
all_cols = vmap(lambda v: jvp(f, (x,), (v,))[1])(jnp.eye(n))
# shape (n, m) — equivalent to J.T, obtained without materialising J
```

### 6.4 Hessian-Vector Products

The **Hessian-vector product (HVP)** $H\mathbf{v} = \nabla^2 f(\mathbf{x}) \mathbf{v}$ is needed for:
- Newton's method and quasi-Newton methods
- Curvature analysis (sharpness-aware minimisation, SAM)
- Computing the Neural Tangent Kernel
- Second-order gradient clipping

**Naive approach:** materialise the full $n \times n$ Hessian (cost $O(n^2)$ memory). Impractical for $n = |\theta| \sim 10^{9}$.

**Efficient HVP via nested AD.** Observe that:

$$H(\mathbf{x})\mathbf{v} = \nabla_\mathbf{x}[\nabla_\mathbf{x} f(\mathbf{x}) \cdot \mathbf{v}]$$

This is a gradient of a scalar (the dot product of the gradient with $\mathbf{v}$), computable in $O(T(f))$:

```python
def hvp(f, x, v):
    """Hessian-vector product H(x)v using reverse-over-forward AD."""
    # Forward-over-reverse: compute gradient in forward mode
    return jax.grad(lambda x: jnp.dot(jax.grad(f)(x), v))(x)
    # Or equivalently: jvp(jax.grad(f), (x,), (v,))[1]
```

**In PyTorch:**
```python
def hvp_torch(loss, params, v):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.flatten() for g in grads])
    gv = (flat_grad * v).sum()
    hvp = torch.autograd.grad(gv, params)
    return torch.cat([h.flatten() for h in hvp])
```

Cost: $O(T(f))$ time, $O(T(f))$ memory — the same as a single forward-backward pass. This makes Newton-CG (Hessian-free optimisation) tractable for large models.

---

## 7. Higher-Order Derivatives

### 7.1 Second Derivatives via create_graph

The standard reverse-mode backward pass discards the computational graph after execution (memory efficiency). To differentiate *through* the backward pass — computing second-order derivatives — we must keep the graph:

**In PyTorch:** Pass `create_graph=True` to `backward()` or `autograd.grad()`. This builds a new computational graph for the gradient computation itself, enabling further differentiation.

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)

def f(x): return (x[0]**3 + x[1]**2).sum()

# First-order gradient
grad = torch.autograd.grad(f(x), x, create_graph=True)[0]
# grad = [3x₀², 2x₁] = [3, 4]

# Second-order: differentiate grad[0] w.r.t. x
d2f_dx0_dx0 = torch.autograd.grad(grad[0], x, retain_graph=True)[0][0]
# = 6x₀ = 6
d2f_dx0_dx1 = torch.autograd.grad(grad[0], x, retain_graph=True)[0][1]
# = 0
```

The key insight: `create_graph=True` makes the gradient a regular tensor connected to the computation graph — differentiating it gives second-order derivatives.

**In JAX:** Higher-order derivatives are natural via function composition:
```python
import jax.numpy as jnp
from jax import grad

f = lambda x: x**4
f_prime   = grad(f)        # 4x³
f_double  = grad(f_prime)  # 12x²
f_triple  = grad(f_double) # 24x

x = jnp.array(2.0)
print(f_prime(x), f_double(x), f_triple(x))  # 32, 48, 24
```

### 7.2 Hessian-Vector Products via Nested AD

As shown in §6.4, HVPs can be computed via a single reverse pass over the JVP computation (or forward pass over the reverse):

```
NESTED AD PATTERNS FOR SECOND-ORDER DERIVATIVES
════════════════════════════════════════════════════════════════════════

  Pattern                  | Cost      | Memory | Use case
  ──────────────────────────────────────────────────────────────────
  reverse-over-reverse     | O(T)      | O(T)²  | Full Hessian rows
  forward-over-reverse     | O(T)      | O(T)   | HVP (preferred)
  reverse-over-forward     | O(T)      | O(T)   | HVP (alternative)
  forward-over-forward     | O(T)      | O(T)   | HVP via dual numbers

  For HVP: H(x)v = ∂/∂x [∇f(x)·v]
    → forward-over-reverse: jvp(grad(f), x, v) [JAX idiomatic]
    → reverse-over-forward: grad(lambda x: jvp(f,x,v)[1])(x)

════════════════════════════════════════════════════════════════════════
```

**Full Hessian (when needed).** For small-scale problems or explicit Hessian computations:

```python
from jax import hessian
H = hessian(f)(x)   # shape (n, n)
# Equivalent to: vmap(jax.grad(jax.grad(f)))(jnp.eye(n))
```

Cost: $O(n)$ reverse passes, each of cost $O(T(f))$ → total $O(n \cdot T(f))$ time, $O(n^2)$ memory.

### 7.3 Taylor-Mode Forward AD

*Taylor-mode* (or *jet mode*) AD computes truncated Taylor series coefficients of a function. A *jet of order $k$* at $\mathbf{x}$ in direction $\mathbf{v}$ is the vector:

$$[f(\mathbf{x}),\ f'(\mathbf{x})\mathbf{v},\ \frac{1}{2}f''(\mathbf{x})[\mathbf{v},\mathbf{v}],\ \ldots,\ \frac{1}{k!}f^{(k)}(\mathbf{x})[\mathbf{v}^k]]$$

Dual numbers (§3) are jets of order 1. Hyper-dual numbers extend this to order 2 and beyond. The key property: all arithmetic operations on order-$k$ jets can be defined purely algebraically, without any additional passes through the original function.

**Applications in ML:**
- Computing higher-order gradient penalties (regularisation on Hessian norms)
- Taylor-mode is used in `jax.experimental.jet` for efficient higher-order expansion
- Stochastic Taylor expansion methods for uncertainty quantification in neural networks

### 7.4 Cost of Higher-Order AD

**Computational cost scaling.** For a function $f : \mathbb{R}^n \to \mathbb{R}$ and reverse-over-reverse AD to compute the full Hessian:
- Each row of $H$ requires one differentiation of $\nabla f$ w.r.t. one input: $O(n)$ passes
- Each pass costs $O(T(\nabla f)) = O(T(f))$
- Total: $O(n \cdot T(f))$ time, $O(n)$ memory for HVP; $O(n^2)$ memory for full Hessian

**Memory amplification.** With `create_graph=True`, the graph for the backward pass itself is stored. For a network with $L$ layers, this doubles the memory (storing both forward and backward activation graphs). For higher orders, memory grows multiplicatively. This is why:
- Full Hessian computations are impractical beyond $n \sim 10^5$ parameters
- HVPs (which avoid materialising the Hessian) scale to $n \sim 10^{10}$
- Gradient checkpointing (§8.1) is essential for large-scale higher-order AD

---

## 8. Advanced Implementation Topics

### 8.1 Gradient Checkpointing

**The problem.** Training a deep network requires storing all intermediate activations during the forward pass (to be used in the backward pass). For a transformer with $L$ layers and sequence length $T$, the activation memory scales as $O(LT)$, which at $L = 96$, $T = 8192$ can exceed GPU memory.

**Solution: Gradient checkpointing (activation rematerialisation).** Instead of storing all activations, store only *checkpoint* activations at selected layers. During the backward pass, when activations are needed, *recompute* the forward pass from the nearest checkpoint.

```
GRADIENT CHECKPOINTING: MEMORY vs COMPUTE TRADEOFF
════════════════════════════════════════════════════════════════════════

  Standard training:
    Forward:   store all L activations     → memory O(L)
    Backward:  use stored activations      → no extra compute

  With checkpointing (√L scheme):
    Divide L layers into √L segments.
    Forward: store only activations at √L segment boundaries.
    Backward: for each segment, recompute the √L activations within it.

  Cost analysis:
    Memory:  O(√L) instead of O(L)
    Compute: O(L) + O(√L · √L) = O(2L)  [one extra forward pass total]

  In practice: checkpointing roughly halves memory for ~33% extra compute.

════════════════════════════════════════════════════════════════════════
```

**In PyTorch:**
```python
from torch.utils.checkpoint import checkpoint

class CheckpointedLayer(nn.Module):
    def forward(self, x):
        # Recompute this block during backward, don't store activations
        return checkpoint(self.block, x, use_reentrant=False)
```

**Selective checkpointing.** Not all layers cost the same. Attention layers (cost $O(T^2 d)$) are more expensive to recompute than linear layers (cost $O(Td^2/T = O(d^2))$). Optimal checkpointing strategies (Chen et al., 2016; Kirisame et al., 2021) assign checkpoints to minimise memory subject to a compute budget.

**LLM training practice.** GPT-3 and subsequent large models use aggressive gradient checkpointing to fit within GPU memory. Megatron-LM checkpoints at every transformer layer; some implementations checkpoint at every attention head. The tradeoff is approximately 30-40% extra compute for an order of magnitude reduction in activation memory.

### 8.2 Custom Backward Functions

Sometimes the VJP computed by automatic differentiation is numerically unstable or less efficient than a manually derived one. Custom backward functions allow overriding the automatic VJP.

**PyTorch `torch.autograd.Function`:**

```python
class LogSumExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Numerically stable logsumexp
        m = x.max()
        log_z = m + (x - m).exp().sum().log()
        softmax = (x - log_z).exp()
        ctx.save_for_backward(softmax)  # save for backward
        return log_z

    @staticmethod
    def backward(ctx, grad_output):
        softmax, = ctx.saved_tensors
        # VJP: grad_output * softmax  (Jacobian of logsumexp is softmax)
        return grad_output * softmax

logsumexp = LogSumExpFunction.apply
```

**Use cases for custom backward:**
- **Numerical stability:** Computing log-softmax directly is more stable than `log(softmax(x))`
- **Efficiency:** The forward pass of `norm(x)` divides by `norm(x)`, but the backward is a simple vector normalisation — storing the normalised vector is cheaper
- **Non-differentiable operations with surrogate gradients:** The straight-through estimator for binary quantisation (§8.3)
- **External solvers:** Differentiating through an LP/QP/neural ODE solver via the implicit function theorem

**JAX `custom_vjp`:**
```python
from jax import custom_vjp

@custom_vjp
def safe_log(x):
    return jnp.log(x)

def safe_log_fwd(x):
    return safe_log(x), x   # residuals

def safe_log_bwd(x, g):
    return (g / (x + 1e-8),)  # stabilised gradient

safe_log.defvjp(safe_log_fwd, safe_log_bwd)
```

### 8.3 Non-Differentiable Operations

Many operations in ML models are technically non-differentiable:
- **Argmax, $\arg\max$:** Zero gradient almost everywhere, undefined at ties
- **Rounding / quantisation:** $\text{round}(x)$ has zero gradient everywhere except at integers
- **Sampling:** Drawing from a discrete distribution is non-differentiable w.r.t. the distribution parameters
- **Sorting:** Non-differentiable at ties; piecewise linear elsewhere

**The Straight-Through Estimator (STE).** For a non-differentiable function $q$ (e.g., rounding to 1-bit), use the identity function as a surrogate gradient:

$$\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial q(x)} \cdot 1 \quad [\text{pass gradient straight through } q]$$

```python
class STERound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output   # gradient passes through unchanged
```

STE is the foundation of quantisation-aware training (QAT) used in deploying transformers on edge hardware (e.g., GPTQ, AWQ, QuIP).

**Gumbel-Softmax (reparameterisation for discrete distributions).** The Gumbel-softmax trick provides a continuous approximation to categorical sampling that is differentiable:

$$y_i = \frac{\exp((\log \pi_i + g_i)/\tau)}{\sum_j \exp((\log \pi_j + g_j)/\tau)}, \quad g_i \sim \text{Gumbel}(0, 1)$$

As temperature $\tau \to 0$, $y$ approaches a one-hot sample from $\pi$. For $\tau > 0$, $y$ is a differentiable function of $\pi$ (and $g$, which is reparameterised from a uniform source).

**Subgradients.** For convex non-smooth functions (e.g., ReLU, $|x|$, $\max(x, 0)$), PyTorch and JAX return a subgradient (an element of the Clarke subdifferential). For ReLU, the convention is $\text{relu}'(0) = 0$; for $|x|$, $|x|'_0 = 0$.

### 8.4 Mixed-Precision and Numerical Stability

**Mixed-precision training (FP16/BF16/FP32).** Modern GPU training uses 16-bit floating point for activations and weights (halving memory and doubling throughput) while accumulating gradients in 32-bit. AD must handle the precision mismatch:

```
MIXED-PRECISION AD CONSIDERATIONS
════════════════════════════════════════════════════════════════════════

  FP16 range: ~6e-5 to 6.5e4  (dynamic range: 5 orders of magnitude)
  FP32 range: ~1.2e-38 to 3.4e38  (dynamic range: 77 orders of magnitude)

  Gradient underflow in FP16:
  ─ Gradients of deep networks often have values < 6e-5
  ─ These round to zero in FP16, killing learning (vanishing gradients)

  Solution: Loss scaling
  ─ Multiply loss by scale factor S (e.g., S = 2^15)
  ─ Gradients are S× larger → stay in FP16 range
  ─ Before weight update, divide by S to recover true gradient
  ─ Dynamic loss scaling: halve S on overflow, double on stability

  BF16 (bfloat16):
  ─ Same dynamic range as FP32 (8 exponent bits)
  ─ Smaller mantissa (7 bits vs 23 for FP32, 10 for FP16)
  ─ Less underflow risk than FP16; preferred for modern LLM training

════════════════════════════════════════════════════════════════════════
```

**Numerically stable gradient implementations.** Some gradients are analytically well-defined but numerically problematic:
- `log(softmax(x))`: implement as `log_softmax(x) = x - logsumexp(x)` directly
- `sigmoid(x)` at extreme $x$: use `log_sigmoid` and `softplus` primitives
- `norm(x)` gradient at $x = 0$: add $\epsilon$ regularisation or use `safe_norm`

PyTorch's and JAX's standard library operations are implemented with these numerical considerations in mind. When writing custom functions, always derive and implement the numerically stable backward formula explicitly.

---

## 9. Applications in Machine Learning

### 9.1 Training Neural Networks

The training loop of a neural network is the canonical application of reverse-mode AD. Given parameters $\theta$, inputs $\mathbf{x}$, targets $\mathbf{y}$, and loss $\mathcal{L} = \ell(f_\theta(\mathbf{x}), \mathbf{y})$:

1. **Forward pass:** Evaluate $f_\theta(\mathbf{x})$ and $\mathcal{L}$, building the computational graph
2. **Backward pass:** Call `loss.backward()` — runs reverse-mode AD, accumulating $\partial\mathcal{L}/\partial\theta_i$ into `param.grad` for each parameter
3. **Update:** Optimizer applies $\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}$ (or more complex update rules)

**The gradient descent step itself** is covered in §08/02. The AD machinery in step 2 is what this section provides.

**Layer-by-layer VJP structure.** For a 3-layer network $f = f_3 \circ f_2 \circ f_1$:

$$\nabla_\theta \mathcal{L} = J_{f_1}^\top \cdot J_{f_2}^\top \cdot J_{f_3}^\top \cdot \nabla_\mathbf{y} \ell$$

Reverse mode computes this right-to-left — starting from $\nabla_\mathbf{y}\ell$ and multiplying by each layer's transposed Jacobian in sequence. This is exactly backpropagation.

**For LLMs specifically.** Training GPT-4-class models involves:
- $|\theta| \sim 10^{12}$ parameters
- Sequence lengths $T \sim 32768$
- Activations dominating memory (not weights): $O(L \cdot T \cdot d_{\text{model}})$
- Gradient checkpointing (§8.1) at every layer is standard
- Mixed-precision (BF16) + loss scaling for numerical stability
- Tensor parallelism splits the graph across devices; AD must handle cross-device VJPs

### 9.2 Meta-Learning and MAML

**MAML (Model-Agnostic Meta-Learning, Finn et al., 2017)** explicitly requires second-order AD. The meta-objective is:

$$\min_\theta \sum_{\tau \sim p(\mathcal{T})} \mathcal{L}_\tau(U_\tau(\theta))$$

where $U_\tau(\theta) = \theta - \alpha \nabla_\theta \mathcal{L}_\tau(\theta)$ is one step of gradient descent on task $\tau$. The meta-gradient is:

$$\nabla_\theta \mathcal{L}_\tau(U_\tau(\theta)) = (I - \alpha \nabla_\theta^2 \mathcal{L}_\tau(\theta))^\top \nabla_{U_\tau(\theta)} \mathcal{L}_\tau(U_\tau(\theta))$$

This requires the Hessian $\nabla_\theta^2 \mathcal{L}_\tau(\theta)$ — a second-order derivative, computed via `create_graph=True`.

```python
# MAML inner loop with higher-order gradients (PyTorch)
def maml_step(model, task_data, inner_lr, outer_lr):
    # Inner loop: adapt to task
    support_x, support_y, query_x, query_y = task_data
    inner_loss = model(support_x, support_y)
    adapted_params = {}
    grads = torch.autograd.grad(inner_loss, model.parameters(),
                                 create_graph=True)  # keeps graph for meta-grad
    for (name, param), grad in zip(model.named_parameters(), grads):
        adapted_params[name] = param - inner_lr * grad

    # Outer loop: meta-objective on adapted model
    query_loss = functional_forward(model, adapted_params, query_x, query_y)
    meta_grads = torch.autograd.grad(query_loss, model.parameters())
    # meta_grads includes Hessian terms due to create_graph=True above
    return meta_grads
```

**First-order MAML (FOMAML).** Finn et al. show that dropping the Hessian term (using only first-order gradients) performs nearly as well in practice. This avoids `create_graph=True` and is significantly faster. The approximation is valid when the loss surface is locally flat (small Hessian).

### 9.3 Differentiable Programming

Differentiable programming extends the AD paradigm beyond neural networks to arbitrary programs. Any program $f$ written in a differentiable programming language (JAX, Zygote.jl, Enzyme) can be differentiated with respect to its inputs.

**Applications:**
- **Differentiable rendering (NeRF, 3DGS):** Render an image from a 3D representation $\theta$. $\mathcal{L} = ||r(\theta) - r_{\text{target}}||^2$. AD gives $\nabla_\theta \mathcal{L}$, enabling reconstruction of 3D scenes from 2D images.
- **Differentiable physics (Brax, DiffTaichi):** Simulate rigid body or fluid dynamics, then optimise control inputs or material parameters via AD.
- **Differentiable molecular dynamics:** Optimise molecular configurations for drug design by differentiating through force-field evaluations.
- **Neural Architecture Search (DARTS):** Differentiable NAS makes architecture parameters continuous, enabling gradient-based optimisation of the architecture itself.

**Neural ODEs (Chen et al., 2018).** Model the hidden state of a neural network as a continuous dynamical system $\frac{d\mathbf{h}}{dt} = f_\theta(\mathbf{h}(t), t)$. The ODE is solved numerically; the backward pass uses the *adjoint method* — solving a second ODE backward in time for the adjoint variables, avoiding storing the full trajectory. Memory cost: $O(1)$ (only the final state, plus the ODE solver's working memory).

### 9.4 Implicit Differentiation

**The implicit function theorem in AD.** Many ML procedures solve an inner optimisation problem $\mathbf{z}^*(\theta) = \arg\min_\mathbf{z} F(\mathbf{z}, \theta)$. We want $\nabla_\theta g(\mathbf{z}^*(\theta))$ — gradients through the solver.

At the solution, the first-order condition gives $\nabla_\mathbf{z} F(\mathbf{z}^*, \theta) = 0$. Differentiating this identity implicitly:

$$\nabla_\mathbf{z}^2 F \cdot \frac{d\mathbf{z}^*}{d\theta} + \nabla_\theta \nabla_\mathbf{z} F = 0$$

$$\implies \frac{d\mathbf{z}^*}{d\theta} = -(\nabla_\mathbf{z}^2 F)^{-1} \nabla_\theta \nabla_\mathbf{z} F$$

This gives the exact gradient without unrolling the inner solver. Cost: one linear system solve at the solution.

**Applications:**
- **DEQs (Deep Equilibrium Models):** Fixed-point iterations $\mathbf{z}^* = f_\theta(\mathbf{z}^*)$. The backward pass solves one linear system instead of backpropagating through all iterations.
- **Differentiable optimisation layers (cvxpylayers):** QP/LP/SDP solvers inside neural networks, differentiated via KKT implicit differentiation (the optimality conditions are the implicit equations).
- **Differentiable sorting (NeuralSort):** Sort via a relaxed permutation matrix; gradient via implicit differentiation of the sorting criterion.

**In JAX,** the `jax.lax.custom_linear_solve` and the `optimistix` library implement implicit differentiation patterns cleanly.

### 9.5 Gradient-Based Hyperparameter Optimisation

**Hypergradients.** Let $\theta^*(λ)$ be the optimal model parameters for hyperparameter $\lambda$ (e.g., learning rate, regularisation weight). The *hypergradient* $\nabla_\lambda g(\theta^*(λ))$ tells us how to adjust $\lambda$ to improve the validation loss $g$.

**Truncated backpropagation through time (TBPTT).** Approximate $\theta^*(λ)$ by running $K$ gradient steps (instead of converging), then backpropagate through these $K$ steps. Cost: $K \times$ the forward training cost.

**IFT-based hyperparameter optimisation (DrMAD, T1-T2).** Use the implicit function theorem:

$$\nabla_\lambda g(\theta^*(λ)) = -\nabla_\lambda \nabla_\theta \mathcal{L}_{\text{train}} \cdot (\nabla_\theta^2 \mathcal{L}_{\text{train}})^{-1} \cdot \nabla_\theta g(\theta^*(λ))$$

Efficient computation: apply CG to solve the linear system, then backpropagate through the gradient-vector product.

**DARTS (Differentiable Architecture Search).** Relaxes the discrete architecture choices to continuous mixture weights $\alpha$. At each training step:
1. Update model weights $\theta$ by gradient descent on training loss
2. Update architecture weights $\alpha$ by gradient descent on validation loss using hypergradients

This jointly learns the architecture and weights in a single differentiable optimisation.

---

## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Using finite differences as the primary gradient in training | $O(n)$ evaluations; error $\sim 10^{-8}$ accumulates; impractical for $n > 10^3$ | Use AD; reserve finite differences for gradient checking only |
| 2 | Calling `.backward()` twice without `retain_graph=True` | The graph is freed after the first backward pass; second call raises an error | Add `retain_graph=True` if you need multiple backward passes through the same graph |
| 3 | In-place operations on tensors with `requires_grad=True` | Corrupts saved values in the computational graph needed for the backward pass | Use out-of-place operations (`x = x + 1` not `x += 1`) |
| 4 | Forgetting `zero_grad()` before each training step | Gradients accumulate (are added) across steps by default; old gradients corrupt new ones | Call `optimizer.zero_grad()` (or `model.zero_grad()`) at the start of each training iteration |
| 5 | Confusing `.detach()` with `.item()` | `.item()` extracts a Python scalar (breaks the graph by design); `.detach()` creates a tensor that stops gradient flow | Use `.detach()` to stop gradient flow; `.item()` for logging scalar values |
| 6 | Not using `create_graph=True` for meta-learning | Without `create_graph=True`, the gradient is a leaf tensor — differentiating it again gives zero | Always use `create_graph=True` when you need second-order derivatives |
| 7 | Materialising the full Hessian for HVPs | $O(n^2)$ memory is infeasible for $n > 10^5$ | Use the HVP trick: $H\mathbf{v} = \nabla[\nabla f \cdot \mathbf{v}]$; costs $O(T(f))$ |
| 8 | Applying the straight-through estimator carelessly | STE works for binary/ternary quantisation; using it for arbitrary non-differentiable functions can give misleading gradients | Verify that the STE approximation is valid for your specific operation and loss landscape |
| 9 | Expecting exact gradients from mixed-precision AD | FP16 arithmetic introduces rounding errors that accumulate in long backward passes | Use BF16 for better range, or FP32 accumulation; enable loss scaling |
| 10 | Comparing gradient norms without accounting for parameter count | Gradient norms grow with parameter count; a large raw norm does not indicate instability | Normalise by $\sqrt{n}$ or monitor per-layer gradient norms separately |
| 11 | Assuming control flow doesn't affect gradients | `if` and `while` statements that depend on tensor values interact with AD in subtle ways (traced vs dynamic control flow) | In JAX use `lax.cond`/`lax.while_loop` for differentiable control flow; in PyTorch, dynamic control flow works in eager mode but may cause issues under `torch.compile` |
| 12 | Reusing intermediate tensors after `backward()` without `retain_graph` | The backward pass frees all intermediate activations; accessing them afterward gives stale or invalid data | Use `retain_graph=True` or recompute the forward pass if needed after backward |

---

## 11. Exercises

**Exercise 1 ★ — Dual Number Implementation**
Implement a complete dual number class supporting `+`, `-`, `*`, `/`, and the elementary functions `exp`, `log`, `sin`, `cos`. Use your implementation to compute the gradient of $f(x) = \log(\sin(x^2) + e^x)$ at $x = 1.5$ and verify against finite differences.
(a) Implement the `Dual` class with operator overloading.
(b) Implement the elementary dual functions.
(c) Compute $f'(1.5)$ via dual numbers.
(d) Compare to the central finite difference approximation.
(e) What is the accuracy difference? Why?

**Exercise 2 ★ — Manual Reverse-Mode Trace**
For $f(x_1, x_2, x_3) = (x_1 \cdot x_2 + x_3) \cdot \exp(-x_1)$ at $(1, 2, 3)$:
(a) Write the Wengert list (forward trace table) with all intermediate values.
(b) Initialise the backward pass with $\bar{v}_{\text{output}} = 1$ and compute all adjoints.
(c) State the full gradient $\nabla f(1, 2, 3)$.
(d) Verify your result against `numpy`-based finite differences.
(e) Count the number of multiply-add operations in forward vs backward — what is the overhead ratio?

**Exercise 3 ★ — Topological Sort**
Implement the Kahn's algorithm topological sort for a computational graph. Apply it to the graph defined by the function $f(x, y, z) = \sin(x + y) \cdot \log(y \cdot z)$ and determine the correct backward pass order. Verify that your ordering respects all dependencies.

**Exercise 4 ★★ — JVP vs VJP Complexity Crossover**
For a function $f : \mathbb{R}^n \to \mathbb{R}^m$, the full Jacobian costs $O(n \cdot T(f))$ with forward mode and $O(m \cdot T(f))$ with reverse mode.
(a) Implement both approaches for a random linear map $f(\mathbf{x}) = A\mathbf{x}$, $A \in \mathbb{R}^{m \times n}$.
(b) Measure runtime for $(n,m) \in \{(10, 100), (100, 10), (50, 50), (1000, 1)\}$.
(c) Plot the runtime ratio (forward/reverse) as a function of $n/m$.
(d) Verify the crossover prediction at $n = m$.
(e) What does this imply for choosing forward vs reverse mode in a physics simulation with $n = 5$ control parameters and $m = 1000$ state outputs?

**Exercise 5 ★★ — Hessian-Vector Products**
Implement three methods for computing $H(\mathbf{x})\mathbf{v}$ where $H = \nabla^2 f$ and $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top A^\top A \mathbf{x} + \mathbf{b}^\top \mathbf{x}$ (so $H = A^\top A$):
(a) Direct: materialise $H = A^\top A$ and multiply by $\mathbf{v}$.
(b) Via nested AD: $H\mathbf{v} = \nabla[\nabla f \cdot \mathbf{v}]$ using PyTorch or JAX.
(c) Verify all three methods give the same result.
(d) Profile memory usage for $n \in \{100, 1000, 10000\}$.
(e) At what $n$ does materialising the Hessian become impractical (exceeds 1GB)?

**Exercise 6 ★★ — Custom Backward Function**
Implement a numerically stable `log1p_exp` function (which computes $\log(1 + e^x)$, a.k.a. softplus) with a custom VJP.
(a) Implement the naive version: `log(1 + exp(x))` — observe overflow at $x > 88$.
(b) Implement the stable version: `x + log(1 + exp(-x))` for $x > 0$; `log(1 + exp(x))` for $x \leq 0`.
(c) Define a custom backward function with VJP $\partial\text{softplus}/\partial x = \sigma(x)$.
(d) Verify the VJP matches finite differences at $x \in \{-10, 0, 10, 100\}$.
(e) Show that the naive implementation fails at large $x$ while your custom implementation does not.

**Exercise 7 ★★★ — Gradient Checkpointing Analysis**
Analyse gradient checkpointing for a depth-$L$ sequential network.
(a) Implement a sequential computation chain $v_k = \sigma(W_k v_{k-1})$ for $k = 1, \ldots, L$.
(b) Measure peak memory usage (number of activations stored) for $L \in \{10, 50, 100\}$, both with and without checkpointing.
(c) Implement the $\sqrt{L}$-checkpointing scheme: store activations at layers $\sqrt{L}, 2\sqrt{L}, \ldots$
(d) Verify the memory reduction ratio approaches $\sqrt{L}$ as $L$ grows.
(e) Measure the runtime overhead of checkpointing (extra forward computations) and confirm it is approximately one extra forward pass.

**Exercise 8 ★★★ — Implicit Differentiation through a Linear Solver**
Given $\mathbf{z}^*(\theta) = \arg\min_\mathbf{z} \frac{1}{2}||A\mathbf{z} - \mathbf{b}||^2 + \frac{1}{2}\theta||\mathbf{z}||^2$ (ridge regression), compute $d\mathbf{z}^*/d\theta$ via:
(a) Unrolling: differentiate through $K=50$ steps of gradient descent; compare to exact.
(b) Implicit differentiation: derive and implement the IFT formula $d\mathbf{z}^*/d\theta = -(A^\top A + \theta I)^{-1} \mathbf{z}^*$.
(c) Verify both approaches give the same answer.
(d) Compare their computational cost (number of operations and memory).
(e) Explain how this pattern generalises to differentiating through DEQ (Deep Equilibrium Model) fixed-point iterations.

---

## 12. Why This Matters for AI (2026 Perspective)

| Concept | Impact on AI/LLMs |
|---|---|
| Reverse-mode AD | The engine of all neural network training. Without it, training a GPT-class model would require $10^{11}$ function evaluations per gradient step instead of 2. |
| JVP/VJP distinction | JAX's `jit(vmap(grad(f)))` chains transforms on JVPs/VJPs, enabling $100\times$ hardware utilisation via automatic batching and parallelism. |
| Gradient checkpointing | Enables training 100B+ parameter models on finite GPU memory. GPT-3 training used aggressive checkpointing; LLaMA uses it at every transformer block. |
| Custom VJPs | PyTorch's fused attention kernels (FlashAttention) implement custom VJPs for numerical stability and memory efficiency — cutting attention memory from $O(T^2)$ to $O(T)$. |
| Higher-order AD | MAML and MAML-adjacent meta-learning methods rely on differentiating through the inner gradient update. Used in few-shot learning and rapid domain adaptation. |
| Implicit differentiation | DEQ models backpropagate through fixed-point iterations in $O(1)$ memory via the adjoint; differentiable optimisation layers (cvxpy layers) enable constraint-aware neural architectures. |
| Differentiable rendering | NeRF and 3DGS reconstruct 3D scenes from 2D images via AD through a volumetric renderer. Foundation of many multimodal understanding systems. |
| Straight-through estimator | Enables end-to-end training of quantised networks (QAT). GPTQ, AWQ, and QuIP use STE-based quantisation to deploy 70B models at 4-bit precision. |
| HVP without materialising Hessian | SAM (Sharpness-Aware Minimisation) computes the perturbation direction via an HVP approximation, improving generalisation without full Hessian materialisation. |
| Mixed-precision AD | BF16 + loss scaling is the default for all major LLM training runs. Understanding the numerical AD implications is essential for training stability. |

---

## Conceptual Bridge

Automatic differentiation sits at the interface of mathematics and software engineering. The dual number construction (§3) shows that AD has deep algebraic roots — it is not a numerical trick but an exact algebraic computation carried out in an extended number system. The adjoint method (§4) connects to the theory of Lagrange multipliers and optimal control: the adjoints $\bar{v}_k$ are exactly the co-state variables in the Pontryagin maximum principle.

This section builds directly on the chain rule derivation from §05/03-Chain-Rule-and-Backpropagation, which established *why* the backward pass computes correct gradients. Here we have established *how* to implement it efficiently for arbitrary programs, including the data structures (Wengert list, closure graph), the algorithms (topological sort, adjoint accumulation), and the engineering considerations (memory, precision, non-differentiable operations).

Looking forward, the gradients computed by AD feed directly into optimisation algorithms (§08/02-Gradient-Descent and beyond). The AD machinery also underlies the optimality condition analysis from §05/04 — numerical KKT verification uses AD to compute gradients of constraints, and gradient-based hyperparameter optimisation (§9.5) connects back to the Lagrangian duality theory developed there.

```
POSITION IN CURRICULUM
════════════════════════════════════════════════════════════════════════

  §04 Calculus Fundamentals
  ├── §05 Multivariate Calculus
  │   ├── 01-Partial-Derivatives         ← partial derivs, gradients
  │   ├── 02-Jacobians-and-Hessians      ← Jacobian structure, Hessians
  │   ├── 03-Chain-Rule-Backpropagation  ← WHY backprop works
  │   ├── 04-Optimality-Conditions       ← KKT, duality, constrained opt
  │   └── 05-Automatic-Differentiation  ◄══ YOU ARE HERE
  │         (HOW to compute grads efficiently)
  │
  └── §08 Optimization
      ├── 02-Gradient-Descent            ← uses AD-computed gradients
      └── ...

  AD is the implementation layer between
  mathematical calculus and practical optimisation.

════════════════════════════════════════════════════════════════════════
```

---

## Appendix A: Gradient Checking in Practice

Gradient checking is the standard technique for verifying that an analytical VJP implementation is correct. Despite AD being "exact", custom backward functions (`torch.autograd.Function`) can contain bugs — gradient checking catches them.

### A.1 The Central Finite Difference Test

For each parameter $\theta_i$, compute:

$$\frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\theta + h \mathbf{e}_i) - \mathcal{L}(\theta - h \mathbf{e}_i)}{2h}$$

and compare to the analytic gradient from `backward()`. With $h = 10^{-5}$, the central difference has $O(h^2) = O(10^{-10})$ truncation error and $O(\epsilon_{\text{machine}}/h) = O(10^{-11})$ cancellation error — giving a sweet spot accuracy of roughly $10^{-7}$ to $10^{-10}$.

**Rule of thumb:** If the relative error

$$\text{rel\_err} = \frac{\|\mathbf{g}_{\text{analytic}} - \mathbf{g}_{\text{FD}}\|}{\|\mathbf{g}_{\text{analytic}}\| + \|\mathbf{g}_{\text{FD}}\|}$$

is below $10^{-5}$, the gradient is almost certainly correct. Above $10^{-3}$, there is likely a bug.

**Python gradient checker:**

```python
def gradient_check(f, x, eps=1e-5, tol=1e-4):
    """
    Compare analytic gradient to central FD.
    f: function returning (loss, grad) tuple.
    x: numpy array, the point at which to check.
    """
    loss, grad_analytic = f(x)
    grad_fd = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        grad_fd[i] = (f(xp)[0] - f(xm)[0]) / (2*eps)

    num = np.linalg.norm(grad_analytic - grad_fd)
    den = np.linalg.norm(grad_analytic) + np.linalg.norm(grad_fd) + 1e-12
    rel_err = num / den
    print(f'Gradient check: relative error = {rel_err:.2e}',
          'PASS' if rel_err < tol else 'FAIL')
    return rel_err < tol
```

**Common gradient check failures:**

| Symptom | Likely cause |
|---|---|
| Relative error > 1e-2 | Wrong sign in VJP (e.g., forgot negative in sin/cos backward) |
| Relative error ~ 1e-7 to 1e-3 | FP32 precision limitation; try FP64 or reduce network depth |
| Specific parameter has large error | Missing contribution from a fan-out node (forgot +=) |
| Error only at boundaries | Non-smooth operation (ReLU at 0, max at ties) — expected |

**For AI:** PyTorch's `torch.autograd.gradcheck` is the production-quality version. It uses complex-step differentiation (evaluating at $x + ih$ in the complex plane) instead of real finite differences, giving machine-precision gradient checks without the cancellation problem.

```python
import torch
from torch.autograd import gradcheck

# Verify a custom autograd function
x = torch.randn(5, dtype=torch.float64, requires_grad=True)
ok = gradcheck(my_custom_function, (x,), eps=1e-6, atol=1e-4)
print(f'gradcheck: {"PASS" if ok else "FAIL"}')
```

Note: `gradcheck` requires `float64` — the extra precision is necessary to separate AD errors from FP32 rounding noise.

### A.2 When Gradient Checking Fails at Non-Smooth Points

Functions like ReLU, absolute value, and max/min have non-smooth points where the finite difference straddles a kink. At $x = 0$ for ReLU:

- For $h > 0$: $(\text{relu}(h) - \text{relu}(-h))/(2h) = h/(2h) = 0.5$
- But the subgradient convention is `relu'(0) = 0`

This causes gradient check failures that are *not bugs* — they are expected behaviour. Filter them by checking that failures only occur at known non-smooth points.

---

## Appendix B: AD in Modern Frameworks — Design Comparison

Understanding the design choices in PyTorch, JAX, and TensorFlow 2.x helps explain why certain patterns (like `create_graph`, `detach`, and functional transforms) exist.

### B.1 PyTorch — Dynamic Graph with Closures

PyTorch's autograd is **define-by-run**: the graph is constructed imperatively as Python executes. Each tensor with `requires_grad=True` carries a `grad_fn` — a closure that captures the local VJP rule and references to its input tensors.

**Key design decisions:**
- **Mutable tensors:** PyTorch tensors are mutable (unlike JAX arrays), enabling in-place operations but requiring careful graph tracking
- **`retain_graph=False` by default:** The backward pass frees the graph after execution — important for memory efficiency in the typical single-backward-pass training loop
- **`no_grad()` context:** Disables graph construction entirely, halving memory and time for inference
- **`detach()`:** Creates a new tensor sharing data but not connected to the graph — used to stop gradient flow (e.g., in target networks for RL, in EMA teacher models)
- **`torch.compile` (PyTorch 2.x):** Traces the Python program to a graph IR, then applies backend-specific optimisations (fusing operations, reducing memory allocations) while preserving AD correctness

### B.2 JAX — Functional Transforms

JAX takes a fundamentally different approach: **pure functional transformations** over programs. There are no mutable tensors; `jax.grad(f)(x)` returns a new function (the gradient function) by tracing `f` to a JAX expression (jaxpr).

**Key transforms:**
- `jax.grad(f)`: reverse-mode AD, returns gradient function
- `jax.jvp(f, x, v)`: forward-mode JVP
- `jax.vjp(f, x)`: returns (primals, vjp_fn) for manual VJP composition
- `jax.jit(f)`: JIT-compile to XLA (fuses ops, eliminates Python overhead)
- `jax.vmap(f)`: vectorise over a batch dimension without explicit looping
- `jax.pmap(f)`: parallelise over multiple devices

**Composability:** `jax.jit(jax.vmap(jax.grad(f)))` works as expected — the transforms compose algebraically. This is possible because they all operate on the same functional IR (jaxpr).

**Limitations:** JAX requires pure functions (no side effects, no in-place mutation). Control flow that depends on values must use `jax.lax.cond` / `jax.lax.while_loop` to be JIT-compatible.

### B.3 TensorFlow 2.x — GradientTape

TensorFlow 2.x uses a **tape-based** approach via `tf.GradientTape`. Operations on `tf.Variable` objects within the tape context are recorded, and `tape.gradient(loss, vars)` executes the backward pass.

```python
import tensorflow as tf

x = tf.Variable([1.0, 2.0])
with tf.GradientTape() as tape:
    y = tf.reduce_sum(x**2)
grad = tape.gradient(y, x)   # [2x_0, 2x_1] = [2, 4]
```

The tape is consumed after `gradient()` — analogous to PyTorch's `retain_graph=False`. For higher-order derivatives, nest tapes:

```python
with tf.GradientTape() as outer:
    with tf.GradientTape() as inner:
        y = tf.reduce_sum(x**2)
    dy = inner.gradient(y, x)   # first-order
d2y = outer.gradient(dy, x)    # second-order
```

### B.4 The Anatomy of `backward()` in PyTorch

When `loss.backward()` is called:

1. **Topological sort:** Starting from the loss tensor, DFS traverses `grad_fn` links to build the reverse processing order
2. **Gradient initialisation:** `loss.grad = tensor(1.0)` (the incoming cotangent is 1)
3. **Backward pass loop:** For each node in reverse order:
   a. Retrieve the incoming `grad_output` (accumulated cotangent)
   b. Call `grad_fn.apply(grad_output)` → executes the local VJP closure
   c. Distribute the resulting cotangents to the `next_functions` (parent nodes)
   d. Accumulate into `leaf.grad` for leaf tensors
4. **Graph cleanup:** Links are broken (graph freed) unless `retain_graph=True`
5. **Hook execution:** Any registered backward hooks fire at step 3c

Understanding this loop explains why:
- The `.grad` attribute accumulates (+=) across multiple `.backward()` calls — always call `zero_grad()` between training steps
- Calling `.backward()` on a non-scalar requires passing a `gradient` argument (the initial cotangent vector matching the tensor shape)
- `torch.autograd.grad()` is a lower-level alternative that gives finer control over which nodes' gradients to return

---

## Appendix C: AD for Sparse and Structured Operations

Large ML models frequently use sparse operations (sparse attention, embedding lookups) and structured operations (grouped convolution, low-rank decompositions). AD handles these with specialised VJP rules.

### C.1 Embedding Lookups (Sparse Gradients)

The embedding layer $E : \mathbb{Z}^T \to \mathbb{R}^{T \times d}$ maps integer token indices to dense vectors. Its VJP is sparse: only the rows of the embedding matrix corresponding to the input tokens receive non-zero gradient.

In PyTorch, `nn.Embedding` produces sparse gradients when `sparse=True`:

```python
embed = nn.Embedding(vocab_size, d_model, sparse=True)
x = embed(input_ids)   # shape (T, d_model)
x.sum().backward()
# embed.weight.grad is a SparseTensor, non-zero only at indices in input_ids
```

Sparse gradients enable significant memory savings for large vocabularies ($V \sim 10^5$) when most tokens are absent from a batch.

### C.2 LoRA and Low-Rank Gradient Approximation

Low-Rank Adaptation (LoRA, Hu et al., 2021) freezes the pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and adds a trainable low-rank perturbation $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$.

The forward pass: $h = (W_0 + BA)x$. The gradients are:
$$\frac{\partial \mathcal{L}}{\partial A} = B^\top \frac{\partial \mathcal{L}}{\partial h} x^\top, \quad \frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial h} x^\top A^\top$$

AD computes these automatically, but the key insight is that by freezing $W_0$ (setting `requires_grad=False`), we eliminate $O(dk)$ gradient storage and restrict the trainable parameter space to $O(r(d+k)) \ll O(dk)$.

**For AI:** LoRA reduces fine-tuning memory from $O(dk)$ to $O(r(d+k))$ per layer. For GPT-3-scale models ($d = k \sim 4096$, $r \sim 16$), this is a $\sim 250\times$ reduction in gradient storage per weight matrix.

### C.3 FlashAttention — Custom Backward for I/O Efficiency

Standard attention computes:
$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

The naive AD backward stores the $T \times T$ attention matrix $P = \text{softmax}(QK^\top/\sqrt{d})$ during the forward pass, requiring $O(T^2)$ memory.

**FlashAttention's custom backward (Dao et al., 2022):** The attention matrix $P$ is *not stored*. Instead, only the $O(T)$ log-sum-exp statistics $\ell_i = \log\sum_j e^{S_{ij}}$ are saved. During the backward pass, the attention matrix is *recomputed* on-the-fly in GPU SRAM (a form of gradient checkpointing applied to the attention computation).

The VJP derivation uses the identity:
$$\frac{\partial \mathcal{L}}{\partial Q} = \left(\frac{\partial \mathcal{L}}{\partial O} V^\top - D \cdot \mathbf{1}^\top\right) \odot P \cdot K / \sqrt{d}$$

where $D_i = \sum_j P_{ij} \frac{\partial \mathcal{L}}{\partial O_{ij}}$ is a scalar per row, computable in $O(T)$.

This custom VJP reduces attention memory from $O(T^2)$ to $O(T)$ — enabling context lengths that would be impossible with standard AD.


---

## Appendix D: Differentiating Through Stochastic Operations

### D.1 The Reparameterisation Trick

To differentiate through sampling from a distribution $\mathbf{z} \sim p_\phi(\mathbf{z})$, we need $\nabla_\phi \mathbb{E}_{p_\phi}[\mathcal{L}(\mathbf{z})]$. Direct differentiation of the expectation requires the score function estimator (REINFORCE), which has high variance.

The **reparameterisation trick** (Kingma & Welling, 2014) rewrites $\mathbf{z} = g_\phi(\boldsymbol{\epsilon})$ where $\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$ is a fixed base distribution independent of $\phi$. Then:

$$\nabla_\phi \mathbb{E}_{p_\phi}[\mathcal{L}(\mathbf{z})] = \nabla_\phi \mathbb{E}_p[\mathcal{L}(g_\phi(\boldsymbol{\epsilon}))] = \mathbb{E}_p\left[\nabla_\phi \mathcal{L}(g_\phi(\boldsymbol{\epsilon}))\right]$$

The gradient now passes through $g_\phi$ via standard AD.

**Standard example — Gaussian:** $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ is reparameterised as $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$ with $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$.

```python
# Reparameterised sampling (PyTorch)
mu = encoder_mu(x)       # mean from network
log_var = encoder_logv(x)  # log variance from network
sigma = torch.exp(0.5 * log_var)
eps = torch.randn_like(sigma)
z = mu + sigma * eps     # reparameterised sample — differentiable in mu, sigma

# Loss: reconstruction + KL divergence
loss = recon_loss(decoder(z), x) + kl_loss(mu, log_var)
loss.backward()  # gradients flow through z to mu and sigma via eps
```

The reparameterisation trick is the key operation in Variational Autoencoders (VAE) and diffusion models — any model that samples from a learned distribution and needs to backpropagate through the sampling operation.

### D.2 Score Function Estimator (REINFORCE)

When reparameterisation is not possible (discrete distributions, black-box simulators), the score function estimator provides unbiased but high-variance gradients:

$$\nabla_\phi \mathbb{E}_{p_\phi}[\mathcal{L}(\mathbf{z})] = \mathbb{E}_{p_\phi}\left[\mathcal{L}(\mathbf{z}) \nabla_\phi \log p_\phi(\mathbf{z})\right]$$

This is estimated by sampling $\mathbf{z}_1, \ldots, \mathbf{z}_K \sim p_\phi$ and computing:

$$\frac{1}{K} \sum_{k=1}^K \mathcal{L}(\mathbf{z}_k) \nabla_\phi \log p_\phi(\mathbf{z}_k)$$

**For AI:** Policy gradient in reinforcement learning (REINFORCE, PPO, SAC) uses this estimator — the policy gradient theorem is exactly the score function estimator applied to the expected cumulative reward.

### D.3 Straight-Through Estimator — Detailed Derivation

For binary quantisation $q(x) = \text{sign}(x)$ (which has zero gradient everywhere), the STE uses:

$$\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial q(x)} \cdot \mathbf{1}[|x| \leq 1]$$

The clipping condition $|x| \leq 1$ is often added to avoid gradient explosion at saturated neurons. This is the version used in binary neural network training (BinaryConnect, XNOR-Net).

**Theoretical justification:** The STE can be understood as the derivative of a piecewise-linear soft approximation to sign($x$) — specifically, the derivative of the clipped linear function $\text{clip}(x, -1, 1)$ evaluated at $q(x)$. As quantisation precision increases (2-bit, 4-bit, 8-bit), the STE approximation improves.

**In quantisation-aware training (QAT):**

```python
class FakeQuantise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, zero_point, bits):
        # Quantise to integer grid
        q_min, q_max = 0, 2**bits - 1
        x_int = torch.round(x / scale + zero_point).clamp(q_min, q_max)
        ctx.save_for_backward(x, scale, torch.tensor([q_min, q_max]))
        return (x_int - zero_point) * scale  # dequantise

    @staticmethod
    def backward(ctx, grad_output):
        x, scale, bounds = ctx.saved_tensors
        q_min, q_max = bounds[0].item(), bounds[1].item()
        x_int = torch.round(x / scale)
        # STE: pass gradient through where x was in quantisation range
        mask = (x_int >= q_min) & (x_int <= q_max)
        return grad_output * mask.float(), None, None, None
```

This is the core of GPTQ, AWQ, and other post-training quantisation methods.

---

## Appendix E: Higher-Order AD in Research

### E.1 Newton's Method with Exact Hessians

For small-scale problems (e.g., $n < 10^4$), Newton's method $\theta \leftarrow \theta - H^{-1}\nabla\mathcal{L}$ converges quadratically. The full Hessian is computed via reverse-over-reverse AD:

```python
# Full Hessian via nested AD (JAX)
from jax import hessian
H = hessian(loss_fn)(params)  # shape (n, n)
update = jnp.linalg.solve(H, grad)
```

**For AI:** Full Hessian computations appear in:
- **Fisher Information Matrix (FIM):** The FIM is the expected Hessian of the log-likelihood, used in natural gradient methods (KFAC, EKFAC, Shampoo) and Bayesian neural networks
- **Influence functions:** $\partial \mathcal{L}_{\text{test}} / \partial \mathbf{x}_{\text{train}} = -H_\theta^{-1} \nabla_\theta \mathcal{L}_{\text{test}} \cdot \nabla_\theta \mathcal{L}_{\text{train}}$, used in interpretability to trace predictions to training examples (Koh & Liang, 2017)
- **Second-order fine-tuning:** LORA+, DARE, and other parameter-efficient methods analyse Hessian structure to identify which parameters are worth fine-tuning

### E.2 Neural Tangent Kernel (NTK)

The NTK $\Theta(\mathbf{x}, \mathbf{x}') = \nabla_\theta f(\mathbf{x}) \cdot \nabla_\theta f(\mathbf{x}')$ describes the linearisation of a neural network around its initialisation. Each entry is an inner product of Jacobians.

**Efficient NTK computation via forward-mode AD:**

```python
# NTK(x, x') = J(x).T @ J(x') where J = df/dtheta
# Using JAX vmap + jvp for efficiency:
from jax import jvp, vmap

def jvp_fn(params, x, v):
    """Forward-mode JVP: J(x)@v."""
    f = lambda p: model(p, x)
    return jvp(f, (params,), (v,))[1]

# Compute one row of the NTK: Theta(x, x')[i, :] for all x'
def ntk_row(params, x, X):
    v = grad(model, params, x)  # gradient = one JVP direction
    return vmap(lambda xp: jvp_fn(params, xp, v))(X)
```

**For AI:** The NTK is central to:
- Lazy training regime analysis (infinite-width networks)
- Understanding double descent and grokking phenomena
- Knowledge distillation theory (student learns in the NTK space of the teacher)

### E.3 Differentiable Simulation and Neural ODEs

**Neural ODEs** (Chen et al., 2018) parameterise the hidden state dynamics as $\frac{d\mathbf{h}}{dt} = f_\theta(\mathbf{h}(t), t)$. The forward pass integrates this ODE; the backward pass uses the *continuous adjoint method*:

$$\frac{d\mathbf{a}(t)}{dt} = -\mathbf{a}(t)^\top \frac{\partial f_\theta}{\partial \mathbf{h}}, \quad \frac{d\mathbf{g}}{dt} = -\mathbf{a}(t)^\top \frac{\partial f_\theta}{\partial \theta}$$

where $\mathbf{a}(t) = \partial \mathcal{L} / \partial \mathbf{h}(t)$ is the adjoint (continuous counterpart of the discrete adjoint in standard AD). This is solved by *reverse-time integration* — starting from the terminal condition $\mathbf{a}(T) = \partial \mathcal{L} / \partial \mathbf{h}(T)$ and integrating backward.

Memory cost: $O(1)$ activations (only the current state) — as opposed to $O(L)$ for a discrete $L$-layer ResNet. This makes Neural ODEs attractive for very deep networks and continuous-time dynamics.


---

## Appendix F: Debugging AD Issues — A Field Guide

This appendix collects the most common AD debugging scenarios encountered in practice, with systematic diagnostic approaches.

### F.1 Gradient is Zero (Everywhere)

**Symptom:** Loss decreases by zero at every training step; `param.grad` is all zeros.

**Diagnostic checklist:**

1. **Is `requires_grad` set correctly?**
   ```python
   print(any(p.requires_grad for p in model.parameters()))  # should be True
   ```

2. **Was `zero_grad()` called before `backward()` but after the forward pass?**
   ```python
   # Wrong order:
   optimizer.zero_grad()
   loss = model(x)    # gradients accumulated from here
   loss.backward()
   optimizer.step()
   # Correct order: (above is actually correct)
   ```

3. **Is a `detach()` or `no_grad()` breaking the graph?**
   ```python
   h = encoder(x).detach()  # detach breaks gradient flow!
   loss = decoder(h)
   loss.backward()  # encoder gets no gradient
   ```

4. **Are you using `.item()` or converting to numpy mid-graph?**
   ```python
   scale = loss.item()  # extracts Python float, breaks graph
   new_loss = scale * other_loss  # other_loss.grad exists, but scale doesn't
   ```

### F.2 Loss is NaN

**Symptom:** `loss.item()` returns `nan`; training diverges immediately.

**Diagnostic checklist:**

1. **Check for log(0) or division by zero:**
   ```python
   # Add epsilon:
   torch.log(x + 1e-8)       # instead of torch.log(x)
   x / (norm + 1e-8)          # instead of x / norm
   ```

2. **Check for exp() overflow:**
   ```python
   # Attention scores before softmax can be large:
   torch.exp(1000.)  # inf
   # Fix: use stable softmax with max subtraction
   ```

3. **Check for FP16 underflow in gradients:**
   ```python
   # Enable loss scaling for mixed precision:
   scaler = torch.cuda.amp.GradScaler()
   with torch.cuda.amp.autocast():
       loss = model(x)
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

4. **Check learning rate — too large causes NaN in first step:**
   ```python
   # Rule of thumb: lr < 1 / (max eigenvalue of Hessian)
   # For Adam: lr = 1e-4 to 3e-4 for most transformer architectures
   ```

### F.3 Exploding Gradients

**Symptom:** Gradient norms grow exponentially across layers (RNNs, very deep networks without skip connections).

**Diagnosis:**
```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f'{name}: grad norm = {param.grad.norm():.4f}')
```

**Solutions:**
- **Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
- **Residual connections:** Ensure skip connections are present (ResNet/Transformer design)
- **Layer normalisation:** Normalises activations before each layer, preventing gradient signal amplification
- **Better initialisation:** Xavier/He initialisation keeps activation variances consistent across layers

### F.4 RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation

**Cause:** An in-place operation modified a tensor that is needed by a `grad_fn` that has not yet been executed.

**Fix:** Replace in-place operations with out-of-place equivalents in any tensor participating in gradient computation:

```python
# Wrong:
x += 1        # in-place
x.relu_()     # in-place ReLU

# Correct:
x = x + 1    # out-of-place
x = x.relu() # out-of-place
```

---

## Appendix G: Practical AD Patterns for LLM Research

### G.1 Gradient Accumulation

Training with effective batch size $B$ when GPU memory only allows batch size $b < B$:

```python
accumulation_steps = B // b
optimizer.zero_grad()

for i, (x, y) in enumerate(dataloader):
    loss = model(x, y) / accumulation_steps  # scale loss
    loss.backward()  # accumulate gradients

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

The key insight: dividing loss by `accumulation_steps` ensures the effective gradient is the average over the full batch of size $B$, not the sum.

### G.2 EMA (Exponential Moving Average) of Weights

EMA models (used in DDPM diffusion, SimCLR, Bootstrap Your Own Latent) maintain a copy of the model with exponentially averaged weights. The EMA update must NOT be included in gradient computation:

```python
@torch.no_grad()  # prevent EMA update from entering the compute graph
def ema_update(model, ema_model, decay=0.9999):
    for p, p_ema in zip(model.parameters(), ema_model.parameters()):
        p_ema.data = decay * p_ema.data + (1 - decay) * p.data
```

Without `@torch.no_grad()`, the EMA update would be recorded on the tape, consuming memory and potentially causing incorrect gradients.

### G.3 Automatic Mixed Precision (AMP) with AD

PyTorch AMP automates the mixed-precision training recipe:

```python
scaler = torch.cuda.amp.GradScaler(init_scale=2**15)

for x, y in loader:
    optimizer.zero_grad()

    # Forward pass in FP16/BF16
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = model(x)
        loss = criterion(logits, y)

    # Backward pass with loss scaling (for FP16; not needed for BF16)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)    # unscale before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

The `GradScaler` dynamically adjusts the loss scale: doubles it every 2000 steps (building up headroom), halves it immediately on NaN/inf gradient detection.

### G.4 Distributed Training and AD

In data-parallel training (DDP), gradients are all-reduced across devices after each `backward()`:

```python
# PyTorch DDP: gradient sync is automatic
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

for x, y in loader:
    loss = model(x, y)
    loss.backward()      # AD runs locally; DDP all-reduces grads across devices
    optimizer.step()
    optimizer.zero_grad()
```

In tensor-parallel training (Megatron-LM, DeepSpeed), the model is split across devices. AD must handle cross-device VJPs: the gradient of an all-reduce operation is another all-reduce (by symmetry of the reduction). This is automatically handled by NCCL-aware AD backends.


---

## Appendix H: Mathematical Foundations — Cotangent Spaces and Functors

For readers with a differential geometry background, AD has a clean categorical formulation that explains why JVP and VJP are duals, and why they compose correctly.

### H.1 Tangent and Cotangent Spaces

For a smooth manifold $M$ and a point $p \in M$, the **tangent space** $T_p M$ is the vector space of tangent vectors — directions in which one can move from $p$. The **cotangent space** $T_p^* M$ is its dual — linear functionals on $T_p M$.

For $M = \mathbb{R}^n$, $T_p \mathbb{R}^n \cong \mathbb{R}^n$ (tangent vectors are just vectors). The cotangent space $T_p^* \mathbb{R}^n \cong (\mathbb{R}^n)^*$ consists of row vectors (linear maps $\mathbb{R}^n \to \mathbb{R}$).

**The differential of $f$:** For $f : \mathbb{R}^n \to \mathbb{R}^m$, the differential $df_p : T_p \mathbb{R}^n \to T_{f(p)} \mathbb{R}^m$ is the Jacobian:
$$df_p(\dot{\mathbf{x}}) = J_f(p) \dot{\mathbf{x}} \quad [\text{this is the JVP}]$$

The **pullback** (adjoint/transpose) $df_p^* : T_{f(p)}^* \mathbb{R}^m \to T_p^* \mathbb{R}^n$ maps cotangent vectors backward:
$$df_p^*(\mathbf{v}) = \mathbf{v}^\top J_f(p) \quad [\text{this is the VJP}]$$

### H.2 The Chain Rule as Functor Composition

The chain rule for $g \circ f$ at the level of tangent maps:

$$d(g \circ f)_p = dg_{f(p)} \circ df_p \quad [\text{JVP: apply left-to-right}]$$

For cotangent maps (VJPs):

$$d(g \circ f)_p^* = df_p^* \circ dg_{f(p)}^* \quad [\text{VJP: apply right-to-left}]$$

This explains why:
- **Forward mode** applies Jacobians left-to-right (following the computation)
- **Reverse mode** applies transposed Jacobians right-to-left (running backward through the tape)

Both are instances of the chain rule for function composition, expressed in dual vector spaces.

### H.3 AD as a Functor

In the language of category theory, a **functor** is a structure-preserving map between categories. AD defines two functors on the category of smooth functions:

- **Forward AD (JVP functor):** Maps each function $f : \mathbb{R}^n \to \mathbb{R}^m$ to a function $\hat{f} : \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}^m \times \mathbb{R}^m$ that computes $(f(x), J_f(x)v)$ simultaneously
- **Reverse AD (VJP functor):** Maps each function $f$ to a pair $(f^{\text{fwd}}, f^{\text{bwd}})$ where the backward component $f^{\text{bwd}}$ computes $J_f(x)^\top \bar{y}$ given the forward primals

The correctness of AD — specifically that chaining VJPs gives the correct gradient — follows from this functor being a *homomorphism*: it preserves composition. This is a formal statement of the chain rule.

**For AI:** JAX's design makes this categorical structure explicit. `jax.linear_util.wrap_init` and the `Trace`/`Tracer` abstraction are direct implementations of this functor machinery. Understanding this structure explains why `jax.grad(jax.jit(f)) == jax.jit(jax.grad(f))` — both are applying the same mathematical functor, and functors respect composition.


---

## Appendix I: Reference Summary Tables

### I.1 AD Mode Selection Guide

```
AD MODE SELECTION
════════════════════════════════════════════════════════════════════════

  Scenario                        Mode      Reason
  ──────────────────────────────────────────────────────────────────────
  Scalar loss, many params        Reverse   O(1) passes for scalar output
  Many outputs, few inputs        Forward   n << m, n passes suffices
  Full Jacobian, n~m              Either    Pick by implementation ease
  HVP H*v (for SAM, Newton)       Mixed     jvp(grad(f), x, v)
  Second-order MAML               Reverse   create_graph=True
  ODE sensitivity analysis        Forward   Few params, many state vars
  NTK computation                 Forward   vmap over JVPs
  Physics simulation gradient     Forward   Few controls, many outputs
  Hyperparameter gradient (IFT)   Reverse   + linear solve for implicit

════════════════════════════════════════════════════════════════════════
```

### I.2 VJP Rules Quick Reference

| Operation | Forward $y = f(x)$ | VJP $\bar{x} \mathrel{+}=$ |
|---|---|---|
| $y = ax + b$ | affine | $\bar{y} \cdot a$ |
| $y = x_1 + x_2$ | add | $\bar{y}$ to both inputs |
| $y = x_1 \cdot x_2$ | multiply | $\bar{y} x_2$ to $x_1$; $\bar{y} x_1$ to $x_2$ |
| $y = Ax$ | matmul | $A^\top \bar{y}$ |
| $y = x^\top A$ | row-matmul | $A \bar{y}$ |
| $y = \exp(x)$ | exp | $\bar{y} e^x$ |
| $y = \log(x)$ | log | $\bar{y}/x$ |
| $y = \sin(x)$ | sin | $\bar{y}\cos(x)$ |
| $y = \cos(x)$ | cos | $-\bar{y}\sin(x)$ |
| $y = \sigma(x)$ | sigmoid | $\bar{y}\sigma(x)(1-\sigma(x))$ |
| $y = \text{relu}(x)$ | relu | $\bar{y}\mathbf{1}[x > 0]$ |
| $y = \text{softmax}(x)$ | softmax | $\bar{y} \odot y - (\bar{y} \cdot y) y$ |
| $y = \text{LSE}(x)$ | logsumexp | $\bar{y} \cdot \text{softmax}(x)$ |
| $y = \|x\|$ | norm | $\bar{y} \cdot x/\|x\|$ |
| $y = x/\|x\|$ | normalize | $\bar{y}/\|x\| - (\bar{y}\cdot x/\|x\|^3) x$ |

### I.3 AD Framework Feature Comparison

| Feature | PyTorch | JAX | TensorFlow 2 |
|---|---|---|---|
| Graph style | Dynamic (eager) | Functional (traced) | Eager + `tf.function` |
| VJP API | `loss.backward()` | `jax.vjp` | `tape.gradient` |
| JVP API | `torch.func.jvp` | `jax.jvp` | Not built-in |
| Higher-order | `create_graph=True` | `jax.grad(jax.grad(...))` | Nested tapes |
| JIT | `torch.compile` | `jax.jit` | `@tf.function` |
| Vectorisation | `torch.vmap` | `jax.vmap` | `tf.vectorized_map` |
| Sparse grads | `sparse=True` | Limited | `tf.IndexedSlices` |
| Custom VJP | `autograd.Function` | `@custom_vjp` | `@tf.custom_gradient` |
| Mixed precision | `torch.autocast` | Built-in | `tf.keras.mixed_precision` |
| Checkpointing | `torch.utils.checkpoint` | `jax.checkpoint` | `tf.recompute_grad` |


### I.4 Complexity Summary

| Operation | Time | Memory | Notes |
|---|---|---|---|
| Forward-mode JVP | $O(T(f))$ per direction | $O(n)$ | One tangent vector propagated |
| Reverse-mode VJP | $O(T(f))$ per cotangent | $O(K)$ activations | $K$ = tape length |
| Full Jacobian (fwd) | $O(n \cdot T(f))$ | $O(n)$ | $n$ JVP passes |
| Full Jacobian (rev) | $O(m \cdot T(f))$ | $O(K)$ | $m$ VJP passes |
| HVP $Hv$ | $O(T(f))$ | $O(K)$ | Mixed forward-over-reverse |
| Full Hessian | $O(n \cdot T(f))$ | $O(n^2)$ | $n$ HVP passes |
| $K$-step unroll | $O(K \cdot T(\nabla f))$ | $O(K \cdot K_\text{activ})$ | Biased if not converged |
| IFT gradient | $O(T(f) + n^\omega)$ | $O(n^2)$ | $\omega \approx 2.37$ for matrix solve |
| Gradient checkpointing | $O(2T(f))$ | $O(\sqrt{K})$ | $\sqrt{L}$ scheme |


---

## Further Reading

1. **Griewank & Walther** — *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation* (2008). The canonical reference. Covers forward and reverse mode, complexity theory, and higher-order AD.

2. **Baydin, Pearlmutter, Radul & Siskind** — *Automatic Differentiation in Machine Learning: a Survey* (JMLR 2018). Accessible survey connecting AD theory to ML practice.

3. **Paszke, Gross, Massa et al.** — *PyTorch: An Imperative Style, High-Performance Deep Learning Library* (NeurIPS 2019). Describes the design of PyTorch's autograd engine.

4. **Bradbury, Frostig, Hawkins et al.** — *JAX: composable transformations of Python+NumPy programs* (2018). Describes JAX's functional AD and transform composition.

5. **Maclaurin, Duvenaud & Adams** — *Gradient-based Hyperparameter Optimization through Reversible Learning* (ICML 2015). Original `autograd` paper; derives hyperparameter gradients via AD through training.

6. **Dao, Fu, Ermon, Rudra & Re** — *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (NeurIPS 2022). Custom backward for attention; IO complexity analysis.

7. **Chen, Rubanova, Bettencourt & Duvenaud** — *Neural Ordinary Differential Equations* (NeurIPS 2018). Adjoint method for continuous-depth networks.

8. **Blondel, Berthet, Cuturi et al.** — *Efficient and Modular Implicit Differentiation* (NeurIPS 2022). Unifies IFT, DEQ, differentiable optimisation layers.

9. **Karpathy** — *micrograd* (2020). Minimal 100-line autograd engine; pedagogically invaluable.

10. **Hu, Shen, Wallis et al.** — *LoRA: Low-Rank Adaptation of Large Language Models* (ICLR 2022). AD through low-rank parameterisation for parameter-efficient fine-tuning.


---

## Appendix J: Notation Reference

| Symbol | Meaning |
|---|---|
| $f : \mathbb{R}^n \to \mathbb{R}^m$ | Function mapping $n$ inputs to $m$ outputs |
| $J_f(\mathbf{x}) \in \mathbb{R}^{m \times n}$ | Jacobian matrix of $f$ at $\mathbf{x}$ |
| $\nabla f(\mathbf{x}) \in \mathbb{R}^n$ | Gradient (for scalar $f$); column vector of partials |
| $\dot{\mathbf{x}} \in T_\mathbf{x}\mathbb{R}^n$ | Tangent vector (JVP seed direction) |
| $\bar{\mathbf{y}} \in T^*_{f(\mathbf{x})}\mathbb{R}^m$ | Cotangent vector (VJP seed / adjoint input) |
| $\bar{v}_k = \partial f / \partial v_k$ | Adjoint variable for intermediate $v_k$ |
| $\varepsilon$ | Dual unit ($\varepsilon^2 = 0$, $\varepsilon \neq 0$) |
| $a + b\varepsilon$ | Dual number (real part $a$, dual part $b$) |
| $T(f)$ | Runtime of a single evaluation of $f$ |
| $K$ | Number of intermediate variables in the Wengert list |
| $L$ | Depth of a sequential network (number of layers) |
| $\|\theta\|$ or $n$ | Number of model parameters |
| $W$ | Wengert list (tape) |
| $\mathcal{G} = (V, E)$ | Computational graph (DAG) |
| $\phi_k$ | Elementary operation at node $k$ |
| $\text{jvp}(f, \mathbf{x}, \dot{\mathbf{x}})$ | Jacobian-vector product $J_f(\mathbf{x})\dot{\mathbf{x}}$ |
| $\text{vjp}(f, \mathbf{x}, \bar{\mathbf{y}})$ | Vector-Jacobian product $J_f(\mathbf{x})^\top \bar{\mathbf{y}}$ |
| $H = \nabla^2 f$ | Hessian matrix (second-order partials) |
| $H\mathbf{v}$ | Hessian-vector product |
| $\mathbf{z}^*(\theta)$ | Optimal solution as a function of hyperparameter $\theta$ |
| $\text{STE}$ | Straight-through estimator |
| $\text{LSE}(\mathbf{x})$ | Log-sum-exp: $\log \sum_i e^{x_i}$ |
| $\sigma(x)$ | Sigmoid function: $1/(1+e^{-x})$ |
| $\text{softmax}(\mathbf{x})_i$ | $e^{x_i}/\sum_j e^{x_j}$ |
| $\mathbf{1}[P]$ | Indicator: 1 if $P$ is true, 0 otherwise |


---

*End of §05/05-Automatic-Differentiation.*

*Next section: §06-Probability-Theory introduces the probabilistic foundations — random variables, distributions, expectation — that underpin the statistical learning theory of modern neural networks.*
