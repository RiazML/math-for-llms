[← Sets and Logic](../02-Sets-and-Logic/notes.md) | [Home](../../README.md) | [Summation and Product Notation →](../04-Summation-and-Product-Notation/notes.md)

---

# Functions and Mappings

## Introduction

Functions are the mathematical objects that model relationships and transformations. In ML, everything from activation functions to loss functions to entire neural networks are functions. Understanding function properties is essential for analysis, optimization, and model design.

## Prerequisites

- Sets and set notation
- Basic algebra
- Coordinate geometry

## Learning Objectives

1. Understand function definition and notation
2. Master domain, codomain, and range concepts
3. Identify function properties (injective, surjective, bijective)
4. Work with composition and inverse functions
5. Recognize common function types in ML

---

## Table of Contents

1. [Function Definition](#1-function-definition)
2. [Function Properties](#2-function-properties)
3. [Function Composition](#3-function-composition)
4. [Inverse Functions](#4-inverse-functions)
5. [Common Function Types](#5-common-function-types)
6. [Multivariate Functions](#6-multivariate-functions)
7. [AI/ML Domain Connections](#7-aiml-domain-connections)
8. [Real-World Code Examples](#8-real-world-code-examples)
9. [Function Spaces](#9-function-spaces--where-functions-live)
10. [Common Pitfalls & Interview Questions](#10-common-pitfalls--interview-questions)
11. [Summary](#11-summary)
12. [Further Reading](#12-further-reading)

---

## 1. Function Definition

### Formal Definition

A **function** f from set A to set B, written f: A → B, assigns to each element a ∈ A exactly one element f(a) ∈ B.

```
Function f: A → B

  A (Domain)              B (Codomain)
  ┌─────────┐             ┌─────────┐
  │    a₁ ──┼─────────────┼──▶ b₁   │
  │    a₂ ──┼─────────────┼──▶ b₂   │
  │    a₃ ──┼─────────────┼──▶ b₁   │  (two inputs can map to same output)
  │    a₄ ──┼─────────────┼──▶ b₃   │
  └─────────┘             │    b₄   │  (not all outputs need be hit)
                          └─────────┘

Valid function:
- Every element in A maps to exactly one element in B

NOT a function:
- Some element in A maps to multiple elements in B
- Some element in A has no mapping
```

### Notation

$$f: A \to B$$
$$a \mapsto f(a)$$

- **A**: Domain (input set)
- **B**: Codomain (potential output set)
- **f(a)**: Image of a under f
- **Range** or **Image**: {f(a) : a ∈ A} ⊆ B

### Examples

```
f: ℝ → ℝ, f(x) = x²
- Domain: ℝ
- Codomain: ℝ
- Range: [0, ∞)

g: ℝ → ℝ, g(x) = eˣ
- Domain: ℝ
- Codomain: ℝ
- Range: (0, ∞)

h: ℝ² → ℝ, h(x, y) = x² + y²
- Domain: ℝ²
- Codomain: ℝ
- Range: [0, ∞)
```

---

## 2. Function Properties

### Injective (One-to-One)

A function f: A → B is **injective** if different inputs give different outputs:

$$f(a_1) = f(a_2) \implies a_1 = a_2$$

Equivalently: $a_1 \neq a_2 \implies f(a_1) \neq f(a_2)$

```
Injective:                      Not Injective:
┌───┐       ┌───┐              ┌───┐       ┌───┐
│ 1 │───────│ a │              │ 1 │───────│ a │
│ 2 │───────│ b │              │ 2 │──┐    │   │
│ 3 │───────│ c │              │ 3 │──┴────│ b │
└───┘       │ d │              └───┘       │ c │
            └───┘                          └───┘
(each output has ≤1 input)    (b has 2 inputs → not injective)
```

**Test**: Horizontal line test - no horizontal line crosses graph more than once.

### Surjective (Onto)

A function f: A → B is **surjective** if every element of B is hit:

$$\forall b \in B, \exists a \in A : f(a) = b$$

```
Surjective:                    Not Surjective:
┌───┐       ┌───┐              ┌───┐       ┌───┐
│ 1 │───┬───│ a │              │ 1 │───────│ a │
│ 2 │───┘   │   │              │ 2 │───────│ b │
│ 3 │───────│ b │              │ 3 │───────│ c │
└───┘       └───┘              └───┘       │ d │
                                           └───┘
(every output is hit)          (d is never hit → not surjective)
```

**Test**: Range = Codomain

### Bijective (One-to-One and Onto)

A function is **bijective** if it is both injective AND surjective.

$$\text{Bijective} \iff \text{Injective} \land \text{Surjective}$$

```
Bijective:
┌───┐       ┌───┐
│ 1 │───────│ a │
│ 2 │───────│ b │
│ 3 │───────│ c │
└───┘       └───┘

Properties:
- Perfect one-to-one correspondence
- |A| = |B|
- Has an inverse function
```

### Summary Table

| Property   | Condition               | Test                 |
| ---------- | ----------------------- | -------------------- |
| Injective  | f(a₁) = f(a₂) ⟹ a₁ = a₂ | Horizontal line test |
| Surjective | Range = Codomain        | Every output is hit  |
| Bijective  | Both above              | Has inverse          |

---

## 3. Function Composition

### Definition

For f: A → B and g: B → C, the **composition** g ∘ f: A → C is:

$$(g \circ f)(x) = g(f(x))$$

```
Composition g ∘ f:

       f           g
  A ─────▶ B ─────▶ C

  x ──▶ f(x) ──▶ g(f(x))

  └────────────────────┘
           g ∘ f
```

### Properties

1. **Associativity**: (h ∘ g) ∘ f = h ∘ (g ∘ f)
2. **Not commutative**: g ∘ f ≠ f ∘ g (in general)
3. **Identity**: f ∘ id = id ∘ f = f

### Example

```
f(x) = x + 1
g(x) = x²

(g ∘ f)(x) = g(f(x)) = g(x + 1) = (x + 1)²
(f ∘ g)(x) = f(g(x)) = f(x²) = x² + 1

Note: (g ∘ f)(x) ≠ (f ∘ g)(x)
```

---

## 4. Inverse Functions

### Definition

For a bijective function f: A → B, the **inverse** f⁻¹: B → A satisfies:

$$f^{-1}(f(a)) = a \quad \text{and} \quad f(f^{-1}(b)) = b$$

Equivalently:
$$f^{-1} \circ f = id_A \quad \text{and} \quad f \circ f^{-1} = id_B$$

```
Function and Inverse:

  A ──────f──────▶ B
    ◀─────f⁻¹─────

  f(a) = b  ⟺  f⁻¹(b) = a
```

### Finding Inverses

1. Write y = f(x)
2. Solve for x in terms of y
3. Swap x and y

**Example:**

```
f(x) = 2x + 3

1. y = 2x + 3
2. x = (y - 3)/2
3. f⁻¹(x) = (x - 3)/2

Verify: f(f⁻¹(x)) = f((x-3)/2) = 2·(x-3)/2 + 3 = x ✓
```

### Existence

A function has an inverse if and only if it is **bijective**.

- Not injective → can't determine which input gave output
- Not surjective → some outputs have no preimage

---

## 5. Common Function Types

### Linear Functions

$$f(x) = mx + b$$

- Domain: ℝ
- Range: ℝ
- Always bijective (for m ≠ 0)
- Inverse: f⁻¹(x) = (x - b)/m

### Polynomial Functions

$$f(x) = a_n x^n + a_{n-1} x^{n-1} + \cdots + a_1 x + a_0$$

- Domain: ℝ
- Range: depends on degree and leading coefficient

### Exponential Functions

$$f(x) = a^x \quad (a > 0, a \neq 1)$$

- Domain: ℝ
- Range: (0, ∞)
- Injective, not surjective onto ℝ
- Inverse: logarithm

### Logarithmic Functions

$$f(x) = \log_a(x)$$

- Domain: (0, ∞)
- Range: ℝ
- Inverse of exponential

### Trigonometric Functions

```
sin: ℝ → [-1, 1]    (periodic, not injective)
cos: ℝ → [-1, 1]    (periodic, not injective)
tan: ℝ\{π/2 + nπ} → ℝ

Restricted domains give bijective versions:
sin: [-π/2, π/2] → [-1, 1]  (bijective, has inverse arcsin)
```

---

## 6. Multivariate Functions

### Vector-Valued Functions

$$\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$$

$$\mathbf{f}(\mathbf{x}) = \begin{bmatrix} f_1(\mathbf{x}) \\ f_2(\mathbf{x}) \\ \vdots \\ f_m(\mathbf{x}) \end{bmatrix}$$

### Examples in ML

```
Linear Transformation: f(x) = Ax + b
- A ∈ ℝᵐˣⁿ, b ∈ ℝᵐ
- Maps ℝⁿ → ℝᵐ

Neural Network Layer:
f(x) = σ(Wx + b)
- σ: activation function (applied element-wise)
- W: weight matrix
- b: bias vector
```

### Jacobian Matrix

For f: ℝⁿ → ℝᵐ, the **Jacobian** is the matrix of partial derivatives:

$$
J_f = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$

### Lipschitz Continuity

A function f is **Lipschitz continuous** with constant L if:

$$\|f(x) - f(y)\| \leq L \cdot \|x - y\| \quad \forall x, y$$

This bounds how fast the function can change — critical for neural network stability.

```
LIPSCHITZ CONTINUITY IN ML
═══════════════════════════════════════════════════════════════════════

┌──────────────────────────┬────────────┬──────────────────────────────┐
│ Function                 │ Lipschitz L│ ML Implication               │
├──────────────────────────┼────────────┼──────────────────────────────┤
│ ReLU: max(0, x)          │     1      │ Gradient ≤ 1, stable         │
│ Sigmoid: σ(x)            │    0.25    │ Gradients shrink (vanishing) │
│ Tanh: tanh(x)            │     1      │ Better than sigmoid          │
│ Linear: Wx               │   ‖W‖      │ Depends on weight norm       │
│ Layer norm               │    ~1      │ Stabilizes per-layer         │
│ Softmax                  │     1      │ Bounded output change        │
└──────────────────────────┴────────────┴──────────────────────────────┘

WHY IT MATTERS:
  • Spectral normalization: forces L = 1 for discriminator (GAN stability)
  • Gradient clipping: ensures ‖∇L‖ ≤ max_norm (bounded update step)
  • Wasserstein GAN: critic MUST be 1-Lipschitz for valid distance
  • Robustness: small input perturbation → bounded output change
```

#### Code Example

```python
import numpy as np

# Compute empirical Lipschitz constant
def estimate_lipschitz(f, x_range, n_pairs=10000):
    """Estimate Lipschitz constant by sampling."""
    x1 = np.random.uniform(*x_range, n_pairs)
    x2 = np.random.uniform(*x_range, n_pairs)
    ratios = np.abs(f(x1) - f(x2)) / (np.abs(x1 - x2) + 1e-10)
    return np.max(ratios)

# Test different activations
sigmoid = lambda x: 1 / (1 + np.exp(-x))
relu = lambda x: np.maximum(0, x)

print(f"ReLU Lipschitz:    {estimate_lipschitz(relu, (-5, 5)):.2f}")    # ~1.0
print(f"Sigmoid Lipschitz: {estimate_lipschitz(sigmoid, (-5, 5)):.2f}")  # ~0.25
print(f"x² Lipschitz on [-5,5]: {estimate_lipschitz(lambda x: x**2, (-5, 5)):.1f}")  # ~10
```

### Fixed Points and Iterative Algorithms

A **fixed point** of function f is a value x* where f(x*) = x\*.

$$f(x^*) = x^*$$

Many ML algorithms are fixed-point iterations:

```
FIXED POINTS IN ML
═══════════════════════════════════════════════════════════════════════

┌────────────────────────────┬──────────────────────────────────────────┐
│ Algorithm                  │ Fixed Point Formulation                  │
├────────────────────────────┼──────────────────────────────────────────┤
│ Gradient descent           │ θ* = θ* - α∇L(θ*) ⟹ ∇L(θ*) = 0       │
│ K-means                    │ centroids = mean(assigned points)        │
│ EM algorithm               │ θ* = argmax E[log L | θ*]               │
│ Power iteration (eigvec)   │ v* = Av* / ‖Av*‖                        │
│ PageRank                   │ r* = M·r* (stationary distribution)     │
│ Self-attention (deep eq.)  │ z* = f_attn(z*) (deep equilibrium model)│
│ Batch norm running stats   │ μ* = (1-α)μ* + α·batch_mean             │
└────────────────────────────┴──────────────────────────────────────────┘

BANACH FIXED POINT THEOREM:
  If f is a contraction (Lipschitz L < 1), then:
  1. There exists a UNIQUE fixed point x*
  2. Iterating x_{n+1} = f(x_n) ALWAYS converges to x*
  3. Convergence rate is geometric: ‖x_n - x*‖ ≤ L^n · ‖x_0 - x*‖

  → This is why gradient descent with small enough lr converges!
  → This is why contraction mappings guarantee EM convergence!
```

---

## 7. AI/ML Domain Connections

### 1. Activation Functions as Mappings (Phase 1)

Activation functions are non-linear transformations $f: \mathbb{R} \to \mathbb{R}$ or $f: \mathbb{R}^n \to \mathbb{R}^n$ that determine the output of a neuron.

```
Sigmoid: σ(x) = 1/(1 + e⁻ˣ)
- Domain: ℝ
- Range: (0, 1)
- Property: Bijective onto (0, 1), Squashing function
- Use: Binary classification probability

ReLU: f(x) = max(0, x)
- Domain: ℝ
- Range: [0, ∞)
- Property: Non-injective (all x < 0 map to 0), Non-saturating gradient
- Use: Hidden layers (prevents vanishing gradient)

Softmax: σ(z)ᵢ = eᶻⁱ / Σⱼ eᶻʲ
- Domain: ℝⁿ
- Range: Δⁿ⁻¹ (Probability Simplex)
- Property: Maps logits to probability distribution
- Use: Multi-class classification
```

### 2. Modern Activation Function Zoo

Beyond classic sigmoid/ReLU, modern LLMs and vision models use specialized activations:

| Activation     | Formula               | Where Used      | Key Property            |
| -------------- | --------------------- | --------------- | ----------------------- |
| **ReLU**       | max(0, x)             | Classic default | Dead neuron problem     |
| **GELU**       | x · Φ(x)              | GPT, BERT, ViT  | Smooth, stochastic      |
| **SwiGLU**     | Swish(xW) ⊙ (xV)      | LLaMA, PaLM     | Gated, best for LLMs    |
| **Mish**       | x · tanh(softplus(x)) | YOLOv4          | Smooth, non-monotonic   |
| **Swish/SiLU** | x · σ(x)              | EfficientNet    | Smooth ReLU alternative |
| **Leaky ReLU** | max(αx, x)            | GANs            | No dead neurons         |

```
ACTIVATION EVOLUTION
═══════════════════════════════════════════════════════════════════════

                  Range:     Smooth?  Dead neurons?  Used in:
  Sigmoid         (0, 1)     ✓        N/A           Binary output
  Tanh            (-1, 1)    ✓        N/A           RNN hidden
  ReLU            [0, ∞)     ✗        YES           CNN, old default
  GELU            (-0.17,∞)  ✓        Rare          Transformers
  SwiGLU          varies     ✓        No            Modern LLMs
  Mish            (-0.31,∞)  ✓        No            Vision models

  GELU(x) = x · Φ(x) where Φ = standard normal CDF
           ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])

  SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊙ (xV + c)
  where Swish(x) = x · sigmoid(x)
```

#### Code Example

```python
import numpy as np

def relu(x): return np.maximum(0, x)
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))
def swish(x): return x * (1 / (1 + np.exp(-x)))
def mish(x): return x * np.tanh(np.log1p(np.exp(x)))

x = np.linspace(-3, 3, 7)
print(f"{'x':>6} {'ReLU':>8} {'GELU':>8} {'Swish':>8} {'Mish':>8}")
for xi in x:
    print(f"{xi:6.1f} {relu(xi):8.3f} {gelu(xi):8.3f} {swish(xi):8.3f} {mish(xi):8.3f}")
print("\nKey difference: GELU/Swish/Mish allow small negative outputs")
print("→ prevents dead neurons, smoother optimization landscape")
```

### 2. Loss Landscapes & Optimization (Phase 2)

Loss functions $L(\theta)$ map parameters $\theta \in \mathbb{R}^d$ to a scalar error $E \in \mathbb{R}$.

- **Global Minima**: Points $\theta^*$ where $L(\theta^*) \leq L(\theta)$ for all $\theta$.
- **Saddle Points**: Points where $\nabla L = 0$ but are minima in some directions and maxima in others (critical in high-dim optimization).
- **Convexity**: If $L$ is convex, any local minimum is global.

```python
# Example: MSE Loss as a functional mapping
def mse_loss(y_true, y_pred):
    # Maps domain (ℝⁿ × ℝⁿ) → Range [0, ∞)
    return ((y_true - y_pred) ** 2).mean()
```

### 3. High-Dimensional Mappings (Phase 3)

In Deep Learning, we often map data between spaces of different dimensions ($f: \mathbb{R}^n \to \mathbb{R}^m$).

- **Embeddings**: Mapping discrete tokens to continuous vector space ($f: V \to \mathbb{R}^d$).
  - Example: Word2Vec maps words to 300-dim vectors where semantic proximity $\approx$ Euclidean distance.
- **Manifold Hypothesis**: High-dimensional data (images) lie on low-dimensional manifolds. Encoders learn the mapping $f: \mathbb{R}^{pixel} \to \mathbb{R}^{latent}$.

### 4. Kernels & Feature Maps (Phase 4)

Kernel methods rely on an implicit mapping $\phi: \mathcal{X} \to \mathcal{H}$ into a higher (or infinite) dimensional Hilbert space to make data linearly separable.

```
Kernel Trick: K(x, y) = ⟨φ(x), φ(y)⟩
- Avoids computing φ(x) explicitly.
- RBF Kernel: K(x, y) = exp(-γ||x - y||²)
  - corresponds to an infinite-dimensional feature map.
```

### 5. Functional Programming in ML (Phase 5)

Modern ML frameworks (JAX, PyTorch) embrace functional paradigms.

- **Pure Functions**: Outputs depend only on inputs (no side effects). Essential for `vmap` (vectorization) and `grad` (differentiation).
- **Function Transformations**: Higher-order functions that take functions as input.
  - `grad(f)`: Returns a function computing $\nabla f$.
  - `jit(f)`: Returns a compiled version of $f$.

```python
# JAX-style functional transformation
import jax.numpy as jnp
from jax import grad

def f(x):
    return 3 * x**2 + 2

# df is a new function generated by transforming f
df = grad(f)
print(df(2.0))  # Output: 12.0 (derivative 6x at x=2)
```

### 6. Universal Approximation Theorem (UAT)

The UAT is the theoretical justification for why neural networks work:

> A feedforward network with **one hidden layer** and a non-polynomial activation can **approximate any continuous function** on a compact set to arbitrary accuracy.

$$\forall \epsilon > 0, \exists N, W, b: \left\| f(x) - \sum_{i=1}^{N} w_i \sigma(a_i x + b_i) \right\| < \epsilon$$

```
UNIVERSAL APPROXIMATION THEOREM
═══════════════════════════════════════════════════════════════════════

WHAT IT SAYS:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  ANY continuous function f: [a,b]ⁿ → ℝ can be approximated by:     │
│                                                                     │
│  g(x) = Σᵢ wᵢ · σ(aᵢᵀx + bᵢ)                                     │
│                                                                     │
│  where σ is any non-polynomial activation (sigmoid, ReLU, etc.)    │
│                                                                     │
│  → The network just needs to be WIDE ENOUGH (large N)              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

WHAT IT DOES NOT SAY:
  ✗ How wide the network needs to be (could be astronomically large)
  ✗ That gradient descent will FIND the right weights
  ✗ That the approximation will GENERALIZE to unseen data
  ✗ That one hidden layer is EFFICIENT (depth helps enormously)

WHY DEPTH MATTERS (in practice):
  1-layer:  Width grows EXPONENTIALLY with input dimension
  L-layer:  Width grows POLYNOMIALLY — much more efficient
  → This is why we use DEEP networks, not just wide ones
  → Why composition of simple functions > one complex function
```

> **Key insight for LLMs**: Transformers approximate functions over **sequences**, not just vectors. The attention mechanism provides an adaptive, input-dependent function composition, which is more powerful than fixed architectures.

### 7. Normalizing Flows and Change of Variables

Normalizing flows require **bijective** functions with tractable Jacobians to model exact probability densities.

$$p_X(x) = p_Z(f(x)) \cdot \left| \det \frac{\partial f}{\partial x} \right|$$

```
CHANGE OF VARIABLES — WHY BIJECTIVITY MATTERS
═══════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Given:  z ~ p_Z(z) (simple distribution, e.g., N(0,1))           │
│  Want:   x ~ p_X(x) (complex distribution, e.g., images)          │
│                                                                     │
│  If x = g(z) where g is BIJECTIVE and differentiable:              │
│                                                                     │
│  p_X(x) = p_Z(g⁻¹(x)) · |det J_{g⁻¹}(x)|                        │
│                                                                     │
│  f = g⁻¹ (encoder)                                                 │
│  g = f⁻¹ (decoder/generator)                                       │
│                                                                     │
│  KEY REQUIREMENTS for f:                                            │
│  ✓ Bijective (for the formula to work)                              │
│  ✓ Differentiable (for the Jacobian to exist)                      │
│  ✓ Easy-to-compute Jacobian determinant (for efficiency)           │
│                                                                     │
│  ARCHITECTURES:                                                     │
│  • Coupling layers (RealNVP): triangular Jacobian → O(n) det      │
│  • Autoregressive (MAF/IAF): triangular by construction           │
│  • Residual flows: f(x) = x + g(x), det via trace estimation     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

USED IN:
  • Image generation (Glow)
  • Variational inference (more expressive posteriors)
  • Density estimation (exact log-likelihood training)
  • Anomaly detection (low-likelihood = anomaly)
```

---

## 8. Real-World Code Examples

### 1. PyTorch Activation Maps (Phase 6)

Visualizing how functions transform the input space.

```python
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-5, 5, 100)
relu = torch.nn.ReLU()
sigmoid = torch.nn.Sigmoid()

plt.plot(x.numpy(), relu(x).numpy(), label='ReLU')
plt.plot(x.numpy(), sigmoid(x).numpy(), label='Sigmoid')
plt.legend()
# Observe: ReLU maps negative inputs to a single point (0) -> Loss of information
# Sigmoid maps infinite range to (0,1) -> Squashing
```

### 2. Optimization Trajectories (Phase 7)

Visualizing the path of parameters $\theta$ on the loss surface $L(\theta)$.

```python
import numpy as np

def loss_surface(x, y):
    return x**2 + 2*y**2  # Convex paraboloid

theta = torch.tensor([2.0, 2.0], requires_grad=True)
optimizer = torch.optim.SGD([theta], lr=0.1)

path = []
for _ in range(50):
    path.append(theta.detach().numpy().copy())
    loss = loss_surface(theta[0], theta[1])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# The array 'path' represents the sequence of mappings:
# f_opt: Initial θ -> Final θ^*
```

### 3. Functional API vs Sequential (Phase 8)

Defining models as explicit function compositions.

```python
# Sequential: Implicit composition f = f_3 ∘ f_2 ∘ f_1
model = torch.nn.Sequential(
    torch.nn.Linear(10, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 1)
)

# Functional: Explicit data flow y = f_3(f_2(f_1(x)))
import torch.nn.functional as F

class Net(torch.nn.Module):
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
```

### 4. Custom Autograd Functions (Phase 9)

Defining a custom differentiable mapping $f: \mathbb{R} \to \mathbb{R}$ by specifying $f(x)$ and $f'(x)$.

```python
class MySquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input ** 2

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 2 * input  # Chain rule: dL/dx = dL/dy * dy/dx
```

### 5. Invertible Neural Networks (Phase 10)

Normalizing Flows require bijective functions $f: \mathcal{X} \to \mathcal{Z}$ to model exact probability densities using the change of variables formula.

$$p_X(x) = p_Z(f(x)) \left| \det \frac{\partial f(x)}{\partial x} \right|$$

Key Requirement: Jacobian determinant must be easy to compute.

---

## 9. Function Spaces — Where Functions Live

Just as vectors live in vector spaces, **functions live in function spaces**. This perspective is essential for understanding generalization.

```
FUNCTION SPACES IN ML
═══════════════════════════════════════════════════════════════════════

┌───────────────────────┬──────────────────────────────────────────────┐
│ Space                 │ ML Connection                                │
├───────────────────────┼──────────────────────────────────────────────┤
│ Hypothesis space H    │ Set of all functions your model CAN learn    │
│                       │ (determined by architecture + parameters)    │
│ Lᵖ spaces             │ L² loss = MSE, L¹ loss = MAE               │
│ RKHS (kernel space)   │ Functions in SVM's feature space            │
│ Sobolev spaces        │ Functions with bounded derivatives          │
│                       │ (smoothness regularization)                  │
│ Banach spaces         │ General framework for function analysis     │
└───────────────────────┴──────────────────────────────────────────────┘

BIAS-VARIANCE THROUGH FUNCTION SPACES:
  • Small H (few parameters):  high bias, low variance → underfitting
  • Large H (many parameters): low bias, high variance → overfitting
  • Regularization restricts H to "smoother" functions
  • A neural network's H is determined by its architecture
```

> **Key insight**: When you choose a model architecture, you're choosing a **function space**. A 2-layer MLP with 100 neurons can represent a different set of functions than a Transformer with 12 layers. Understanding this helps reason about model capacity.

---

## 10. Common Pitfalls & Interview Questions

### Common Pitfalls

1. **Domain Mismatch Errors**:
   - _Issue_: Applying `log(x)` when `x <= 0` (e.g., in Cross-Entropy loss).
   - _Fix_: Use `log(x + epsilon)` or `log_softmax` (numerically stable).

2. **Vanishing Gradients due to Composition**:
   - _Issue_: Deep composition $f_n \circ \dots \circ f_1$ leads to chain rule product $\prod f'_i$. If $|f'_i| < 1$, gradient vanishes.
   - _Fix_: Use ReLU/GELU (derivative near 1) instead of Sigmoid (max derivative 0.25). Use residual connections.

3. **Assuming Bijectivity**:
   - _Issue_: Trying to invert a non-injective function (e.g., recovering input from a ReLU layer output).
   - _Fix_: Use Invertible Neural Networks (INNs) if perfect reconstruction is needed.

4. **Ignoring Lipschitz Constant**:
   - _Issue_: Unconstrained discriminator in GAN → mode collapse.
   - _Fix_: Spectral normalization, gradient penalty (WGAN-GP).

### Interview Questions

1. **Is ReLU invertible?**
   - _Answer_: No. It is not injective (maps all negative numbers to 0). Thus, information is lost and input cannot be uniquely recovered.

2. **Why do we need non-linear activation functions?**
   - _Answer_: Composition of linear functions is just another linear function ($W_2(W_1x) = (W_2W_1)x$). Non-linearity allows approximation of complex functions (Universal Approximation Theorem).

3. **What is the range of Softmax?**
   - _Answer_: The open interval $(0, 1)$ for each component, such that their sum is exactly 1. It maps $\mathbb{R}^n$ to the probability simplex $\Delta^{n-1}$.

4. **Why does SwiGLU outperform ReLU in LLMs?**
   - _Answer_: SwiGLU is a gated activation: Swish(xW) ⊙ (xV). The gating mechanism allows the network to learn which features to pass through, and Swish's smooth non-linearity avoids dead neurons. Empirically 3-5% better on language modeling.

5. **What does the Universal Approximation Theorem actually guarantee?**
   - _Answer_: That a single-hidden-layer network CAN approximate any continuous function given enough neurons. It does NOT guarantee that gradient descent will find the solution, that the required width is practical, or that the result will generalize.

6. **What is a Lipschitz constraint and why is it important in GANs?**
   - _Answer_: A function f is K-Lipschitz if |f(x)-f(y)| ≤ K|x-y|. In Wasserstein GANs, the critic must be 1-Lipschitz for the loss to be a valid distance metric. Enforced via spectral normalization or gradient penalty.

---

## Companion Notebooks

| Notebook                           | Description                                                                                  |
| ---------------------------------- | -------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Interactive examples: function properties, composition, activation functions, loss functions |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions                                                             |

---

## 11. Summary

### Function Properties Checklist

| Property     | Check                             | Example                  |
| ------------ | --------------------------------- | ------------------------ |
| Well-defined | Each input has exactly one output | f(x) = x² ✓              |
| Injective    | f(a) = f(b) ⟹ a = b               | f(x) = 2x ✓, f(x) = x² ✗ |
| Surjective   | Range = Codomain                  | f: ℝ → ℝ, f(x) = x³ ✓    |
| Bijective    | Injective + Surjective            | Has inverse              |

### Key Relationships

```
Composition: (g ∘ f)(x) = g(f(x))

Inverse: f⁻¹(f(x)) = x (only if bijective)

Chain Rule: (g ∘ f)'(x) = g'(f(x)) · f'(x)
```

### ML Function Types

| Type      | Domain | Range   | Bijective?       |
| --------- | ------ | ------- | ---------------- |
| Sigmoid   | ℝ      | (0,1)   | Yes (onto range) |
| ReLU      | ℝ      | [0,∞)   | No               |
| Softmax   | ℝⁿ     | Simplex | No               |
| Linear Wx | ℝⁿ     | ℝᵐ      | Depends on W     |

---

## Exercises

1. Determine if f(x) = x³ - x is injective, surjective, bijective on ℝ
2. Find the inverse of f(x) = (2x + 1)/(x - 3)
3. Given f(x) = x² and g(x) = sin(x), find (f ∘ g)(x) and (g ∘ f)(x)
4. Show that sigmoid is bijective onto (0, 1) and find its inverse
5. Prove that composition of bijections is bijective

---

## 12. Further Reading

### Famous Courses

1. **Stanford CS229: Machine Learning** (Andrew Ng)
2. **MIT 6.042J: Mathematics for Computer Science** (Leighton & Lehman)
3. **Coursera: Mathematics for Machine Learning** (Imperial College London)
4. **Fast.ai: Practical Deep Learning** (Jeremy Howard)

### Best Books

1. **"Mathematics for Machine Learning"** (Deisenroth, Faisal, Ong) — Ch.2: functions and mappings for ML
2. **"Analysis I"** (Terence Tao) — Build intuition for continuity, differentiability
3. **"Deep Learning"** (Goodfellow et al.) — Ch.6: NNs as function approximation
4. **"Pattern Recognition and Machine Learning"** (Bishop) — Probabilistic view of functions

### Papers

- 📄 [Universal Approximation (Hornik 1991)](<https://doi.org/10.1016/0893-6080(91)90009-T>)
- 📄 [GLU Variants Improve Transformer (Shazeer 2020)](https://arxiv.org/abs/2002.05202) — SwiGLU origin
- 📄 [GELU (Hendrycks & Gimpel 2016)](https://arxiv.org/abs/1606.08415)
- 📄 [Spectral Normalization for GANs (Miyato 2018)](https://arxiv.org/abs/1802.05957)
- 📄 [Normalizing Flows (Rezende & Mohamed 2015)](https://arxiv.org/abs/1505.05770)

---

[← Sets and Logic](../02-Sets-and-Logic/notes.md) | [Home](../../README.md) | [Summation and Product Notation →](../04-Summation-and-Product-Notation/notes.md)
