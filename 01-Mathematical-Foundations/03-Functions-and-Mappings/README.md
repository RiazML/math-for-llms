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

## 7. Applications in ML/AI

### 1. Activation Functions

```
Sigmoid: σ(x) = 1/(1 + e⁻ˣ)
- Domain: ℝ
- Range: (0, 1)
- Bijective onto (0, 1)

ReLU: f(x) = max(0, x)
- Domain: ℝ
- Range: [0, ∞)
- Not injective (all negatives map to 0)

Softmax: σ(z)ᵢ = eᶻⁱ / Σⱼ eᶻʲ
- Domain: ℝⁿ
- Range: probability simplex
- Not injective (shift invariant)
```

### 2. Loss Functions

```
MSE: L(y, ŷ) = (1/n) Σ(yᵢ - ŷᵢ)²
- Domain: ℝⁿ × ℝⁿ
- Range: [0, ∞)

Cross-Entropy: L(y, p) = -Σ yᵢ log(pᵢ)
- Domain: probability vectors
- Range: [0, ∞)
```

### 3. Feature Transformations

```
Polynomial Features:
φ: ℝ → ℝ³
φ(x) = [1, x, x²]

Kernel Functions:
K: ℝⁿ × ℝⁿ → ℝ
K(x, y) = φ(x)ᵀφ(y)

RBF Kernel:
K(x, y) = exp(-γ||x - y||²)
```

### 4. Neural Networks as Functions

```
Single Layer:
f(x) = σ(Wx + b)

Deep Network (composition):
f(x) = fₗ ∘ fₗ₋₁ ∘ ⋯ ∘ f₁(x)

Where each fᵢ(x) = σᵢ(Wᵢx + bᵢ)
```

### 5. Encoder-Decoder Architecture

```
Encoder: f: 𝒳 → 𝒵 (map to latent space)
Decoder: g: 𝒵 → 𝒳 (reconstruct)

Goal: g ∘ f ≈ id (autoencoder)

If f is bijective: g = f⁻¹
```

---

## 8. Summary

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

## References

1. Apostol, T. - "Mathematical Analysis"
2. Lang, S. - "Undergraduate Analysis"
3. Goodfellow et al. - "Deep Learning" (Chapter 6: Deep Feedforward Networks)
