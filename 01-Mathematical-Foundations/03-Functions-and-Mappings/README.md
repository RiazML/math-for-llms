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
9. [Common Pitfalls & Interview Questions](#9-common-pitfalls)
10. [Summary](#10-summary)
11. [Further Reading](#11-further-reading)

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

## 9. Common Pitfalls & Interview Questions

### Common Pitfalls (Phase 19)

1. **Domain Mismatch Errors**:
   - _Issue_: Applying `log(x)` when `x <= 0` (e.g., in Cross-Entropy loss).
   - _Fix_: Use `log(x + epsilon)` or `log_softmax` (numerically stable).

2. **Vanishing Gradients due to Composition**:
   - _Issue_: Deep composition $f_n \circ \dots \circ f_1$ leads to chain rule product $\prod f'_i$. If $|f'_i| < 1$, gradient vanishes.
   - _Fix_: Use ReLU (derivative is 1) instead of Sigmoid (max derivative 0.25).

3. **Assuming Bijectivity**:
   - _Issue_: Trying to invert a non-injective function (e.g., recovering input from a ReLU layer output).
   - _Fix_: Use Invertible Neural Networks (INNs) if perfect reconstruction is needed.

### Data Science Interview Questions (Phase 20)

1. **Is ReLU invertible?**
   - _Answer_: No. It is not injective (maps all negative numbers to 0). Thus, information is lost and input cannot be uniquely recovered.

2. **Why do we need non-linear activation functions?**
   - _Answer_: Composition of linear functions is just another linear function ($W_2(W_1x) = (W_2W_1)x$). Non-linearity allows approximation of complex functions (Universal Approximation Theorem).

3. **What is the range of Softmax?**
   - _Answer_: The open interval $(0, 1)$ for each component, such that their sum is exactly 1. It maps $\mathbb{R}^n$ to the probability simplex $\Delta^{n-1}$.

---

## 10. Summary

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

## 11. Further Reading

### Famous Courses

1. **Stanford CS229: Machine Learning** (Andrew Ng)
   - _Why_: The gold standard for ML foundations. Covers functional view of ML deeply.
2. **MIT 6.042J: Mathematics for Computer Science** (Leighton & Lehman)
   - _Why_: Rigorous treatment of sets, functions, and logic.
3. **Coursera: Mathematics for Machine Learning** (Imperial College London)
   - _Why_: Great visualizations of mappings and basis transformations.
4. **Fast.ai: Practical Deep Learning** (Jeremy Howard)
   - _Why_: Code-first approach to functional deep learning.

### Best Books

#### Mathematical Foundations

1. **"Mathematics for Machine Learning"** (Deisenroth, Faisal, Ong)
   - _Phase 15_: Chapter 2 covers functions and mappings specifically for ML context.
2. **"Analysis I"** (Terence Tao)
   - _Phase 17_: Build intuition for "well-behaved" functions (continuity, differentiability).

#### Deep Learning & AI

3. **"Deep Learning"** (Goodfellow, Bengio, Courville)
   - _Phase 16_: Chapter 6 (Deep Feedforward Networks) treats NNs purely as function approximation machines.
4. **"Pattern Recognition and Machine Learning"** (Bishop)
   - _Phase 18_: Probabilistic view of functions as mappings between distributions.
