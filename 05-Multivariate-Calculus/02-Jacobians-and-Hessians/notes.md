# Jacobians and Hessians

> **Navigation**: [01-Partial-Derivatives-and-Gradients](../01-Partial-Derivatives-and-Gradients/) | [03-Chain-Rule-and-Backpropagation](../03-Chain-Rule-and-Backpropagation/) | [04-Optimization-Theory](../04-Optimization-Theory/)

## Overview

The **Jacobian** generalizes gradients to vector-valued functions. The **Hessian** captures second-order derivative information. Together, they're fundamental to understanding neural network backpropagation, optimization curvature, and the geometry of loss landscapes.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  JACOBIAN & HESSIAN IN ML                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  JACOBIAN (m×n matrix)              HESSIAN (n×n matrix)                │
│  ─────────────────────              ────────────────────                │
│                                                                          │
│  f: ℝⁿ → ℝᵐ                         f: ℝⁿ → ℝ                           │
│                                                                          │
│  ┌ ∂f₁/∂x₁  ...  ∂f₁/∂xₙ ┐         ┌ ∂²f/∂x₁²    ...  ∂²f/∂x₁∂xₙ ┐     │
│  │   ⋮       ⋱      ⋮     │         │    ⋮         ⋱        ⋮      │     │
│  └ ∂fₘ/∂x₁  ...  ∂fₘ/∂xₙ ┘         └ ∂²f/∂xₙ∂x₁  ... ∂²f/∂xₙ²   ┘     │
│                                                                          │
│  • Backpropagation                  • Loss curvature                    │
│  • Chain rule as matrices           • Newton's method                   │
│  • Normalizing flows                • Saddle point detection            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- [01-Partial-Derivatives-and-Gradients](../01-Partial-Derivatives-and-Gradients/)
- Matrix operations
- Chain rule basics

## Learning Objectives

1. Understand and compute Jacobian matrices
2. Understand and compute Hessian matrices
3. Apply these to ML optimization
4. Understand their role in backpropagation

---

## 1. The Jacobian Matrix

### Definition

For a function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$:

$$\mathbf{f}(\mathbf{x}) = \begin{pmatrix} f_1(x_1, \ldots, x_n) \\ f_2(x_1, \ldots, x_n) \\ \vdots \\ f_m(x_1, \ldots, x_n) \end{pmatrix}$$

The **Jacobian** is an $m \times n$ matrix:

$$
\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}
$$

> **💡 Key Insight**: Each **row** of J is the gradient of one output component: $\text{Row}_i = (\nabla f_i)^T$

```
Jacobian Structure:

                    Inputs: x₁  x₂  x₃  ...  xₙ
                           ↓   ↓   ↓       ↓
              ┌─────────────────────────────────┐
Output f₁ →   │  ∂f₁/∂x₁  ∂f₁/∂x₂  ...  ∂f₁/∂xₙ │  ← ∇f₁ᵀ
Output f₂ →   │  ∂f₂/∂x₁  ∂f₂/∂x₂  ...  ∂f₂/∂xₙ │  ← ∇f₂ᵀ
    ⋮         │     ⋮         ⋮     ⋱      ⋮     │
Output fₘ →   │  ∂fₘ/∂x₁  ∂fₘ/∂x₂  ...  ∂fₘ/∂xₙ │  ← ∇fₘᵀ
              └─────────────────────────────────┘
                       Jacobian J (m × n)
```

### Example

For $\mathbf{f}(x, y) = \begin{pmatrix} x^2 + y \\ xy \end{pmatrix}$:

$$
\mathbf{J} = \begin{pmatrix}
2x & 1 \\
y & x
\end{pmatrix}
$$

At point $(1, 2)$:
$$\mathbf{J}(1, 2) = \begin{pmatrix} 2 & 1 \\ 2 & 1 \end{pmatrix}$$

---

## 2. Jacobian Properties

### Linear Approximation

$$\mathbf{f}(\mathbf{x} + \Delta\mathbf{x}) \approx \mathbf{f}(\mathbf{x}) + \mathbf{J}\Delta\mathbf{x}$$

This is the multivariate analog of $f(x + h) \approx f(x) + f'(x)h$.

> **💡 Interpretation**: The Jacobian tells you how small input changes affect outputs!

### Chain Rule = Matrix Multiplication!

For $\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^m$ and $\mathbf{f}: \mathbb{R}^m \to \mathbb{R}^p$:

$$\mathbf{J}_{\mathbf{f} \circ \mathbf{g}} = \mathbf{J}_{\mathbf{f}} \cdot \mathbf{J}_{\mathbf{g}}$$

```
Chain Rule for Jacobians:

Input          Layer 1        Layer 2        Output
  x      →      g(x)     →    f(g(x))    →     y
  │              │               │              │
 ℝⁿ            ℝᵐ              ℝᵖ             ℝᵖ
  │              │               │
  └──── J_g ────┴──── J_f ──────┘
        (m×n)        (p×m)

Total Jacobian:  J_total = J_f × J_g  =  (p×n) matrix

CHAIN RULE IS JUST MATRIX MULTIPLICATION!
```

> **🔑 This is the mathematical foundation of backpropagation!**

### Jacobian Determinant

The **Jacobian determinant** measures volume change:

$$\det(\mathbf{J}) = \text{ratio of output volume to input volume}$$

```
Jacobian Determinant:

Input Space              Output Space
┌───────────┐            ┌─────────────────┐
│  ░░░░░░   │            │                 │
│  ░░░░░░   │   f(x)     │  ░░░░░░░░░░░░   │
│  ░░░░░░   │  ────→     │  ░░░░░░░░░░░░   │
│           │            │  ░░░░░░░░░░░░   │
└───────────┘            └─────────────────┘
   Area = A              Area = |det(J)| × A

|det(J)| < 1: Compression
|det(J)| > 1: Expansion
|det(J)| = 1: Volume preserving
```

**ML Application**: Normalizing flows use the Jacobian determinant to track probability density under transformations!

---

## 3. The Hessian Matrix

### Definition

For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the **Hessian** is the matrix of second partial derivatives:

$$
\mathbf{H} = \nabla^2 f = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}
$$

### Key Properties

1. **Symmetric**: $H_{ij} = H_{ji}$ (by Clairaut's theorem)
2. **Jacobian of gradient**: $\mathbf{H} = \mathbf{J}(\nabla f)$
3. **Curvature information**: Eigenvalues = curvatures along principal directions

```
Hessian Structure:

              ┌─────────────────────────────────────┐
              │  ∂²f/∂x₁²    ∂²f/∂x₁∂x₂  ...  ...   │
              │  ∂²f/∂x₂∂x₁  ∂²f/∂x₂²    ...  ...   │
     H =      │     ⋮           ⋮         ⋱    ⋮    │
              │  ∂²f/∂xₙ∂x₁  ...        ... ∂²f/∂xₙ² │
              └─────────────────────────────────────┘
                      Symmetric! H = Hᵀ
```

### Example

For $f(x, y) = x^3 + 2xy^2 - y^3$:

$$\nabla f = \begin{pmatrix} 3x^2 + 2y^2 \\ 4xy - 3y^2 \end{pmatrix}$$

$$
\mathbf{H} = \begin{pmatrix}
6x & 4y \\
4y & 4x - 6y
\end{pmatrix}
$$

---

## 4. Hessian and Optimization

### Second-Order Taylor Expansion

$$f(\mathbf{x} + \Delta\mathbf{x}) \approx f(\mathbf{x}) + \nabla f^T \Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^T \mathbf{H} \Delta\mathbf{x}$$

```
Taylor Expansion:

f(x + Δx) ≈  f(x)     +    ∇fᵀΔx      +   ½ΔxᵀHΔx
             ───            ─────          ──────────
            value      linear term     quadratic term
            at x      (slope info)     (curvature info)
```

### Critical Point Classification

At a critical point where $\nabla f = 0$:

| Hessian Property | Eigenvalues | Point Type |
|-----------------|-------------|------------|
| $\mathbf{H} \succ 0$ (positive definite) | All $> 0$ | **Local minimum** |
| $\mathbf{H} \prec 0$ (negative definite) | All $< 0$ | **Local maximum** |
| $\mathbf{H}$ indefinite | Mixed signs | **Saddle point** |
| $\mathbf{H}$ singular | Some zero | Inconclusive |

```
Critical Point Classification:

         Minimum            Maximum          Saddle Point
           ╲ ╱                ╱ ╲              ↗     ↘
            ●                ●                   ●
           ╱ ╲              ╲ ╱              ↙     ↖
        (bowl up)         (bowl down)       (horse saddle)

     H positive def.     H negative def.    H indefinite
     All λᵢ > 0          All λᵢ < 0         Mixed signs
```

### 2×2 Hessian Test

For $\mathbf{H} = \begin{pmatrix} a & b \\ b & c \end{pmatrix}$:

| Condition | Classification |
|-----------|---------------|
| $a > 0$ and $ac - b^2 > 0$ | Positive definite (minimum) |
| $a < 0$ and $ac - b^2 > 0$ | Negative definite (maximum) |
| $ac - b^2 < 0$ | Indefinite (saddle point) |

> **💡 Remember**: $ac - b^2 = \det(\mathbf{H})$

---

## 5. Newton's Method (Second-Order Optimization)

### Algorithm

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \mathbf{H}^{-1} \nabla f(\mathbf{x}_k)$$

### Derivation

From Taylor expansion, minimize quadratic approximation:
$$q(\Delta\mathbf{x}) = f + \nabla f^T \Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^T \mathbf{H} \Delta\mathbf{x}$$

Setting $\nabla_{\Delta\mathbf{x}} q = 0$:
$$\nabla f + \mathbf{H}\Delta\mathbf{x} = 0$$
$$\Delta\mathbf{x} = -\mathbf{H}^{-1}\nabla f$$

```
Newton vs Gradient Descent:

Gradient Descent              Newton's Method
─────────────────             ───────────────

Step = -η∇f                   Step = -H⁻¹∇f

Uses only gradient            Uses gradient AND curvature
(first-order info)            (second-order info)

     │                             │
     │  ●→→→→→→●                   │  ●─────→●
     │       ↓                     │
     │       ●→→→●                 Jumps directly to
     │           ↓                 the minimum!
     │           ●→●
     │
Takes many small steps        Takes fewer, smarter steps
```

### Comparison

| Method | Update | Convergence Rate |
|--------|--------|------------------|
| Gradient Descent | $-\eta \nabla f$ | Linear (slow) |
| Newton's Method | $-\mathbf{H}^{-1} \nabla f$ | Quadratic (fast!) |

> **⚠️ Trade-off**: Newton is faster but requires computing $\mathbf{H}^{-1}$ which is $O(n^3)$ — expensive for large $n$!

---

## 6. Applications in ML

### 1. Backpropagation and Jacobians

For a neural network layer $\mathbf{y} = \mathbf{f}(\mathbf{x})$:

$$\frac{\partial L}{\partial \mathbf{x}} = \mathbf{J}^T \frac{\partial L}{\partial \mathbf{y}}$$

> **🔑 Backpropagation is repeated Jacobian-transpose-vector products!**

```
Backprop as JᵀV Products:

Forward:  x → Layer 1 → a₁ → Layer 2 → a₂ → Loss L
                ↑              ↑              ↑
             (J₁)ᵀ          (J₂)ᵀ          ∂L/∂a₂

Backward: ∂L/∂x ← J₁ᵀ(∂L/∂a₁) ← J₂ᵀ(∂L/∂a₂) ← from loss

Each layer: multiply incoming gradient by Jᵀ
```

### 2. Softmax Jacobian

For softmax $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$:

$$J_{ij} = \frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

$$\mathbf{J} = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$$

```
Softmax Jacobian:

J = diag(p) - ppᵀ

    ┌ p₁(1-p₁)   -p₁p₂    -p₁p₃   ┐
  = │  -p₂p₁   p₂(1-p₂)  -p₂p₃   │
    └  -p₃p₁    -p₃p₂   p₃(1-p₃) ┘

• Diagonal: pᵢ(1-pᵢ)  (variance-like)
• Off-diagonal: -pᵢpⱼ  (negative, sums constrained!)
```

### 3. Loss Landscape Curvature

The Hessian describes the **curvature** of the loss surface:

- **Large eigenvalues** → steep directions (fast learning)
- **Small eigenvalues** → flat directions (slow learning)
- **Negative eigenvalues** → saddle points (need to escape!)

```
Loss Landscape and Hessian:

   Steep (large λ)
         │
         │  ╲     ╱
         │   ╲   ╱
         │    ● ←── saddle
         │   ╱   ╲
         │  ╱     ╲
         └──────────── Flat (small λ)

Hessian eigenvalue λ = curvature in that direction
```

### 4. Natural Gradient

Uses **Fisher Information Matrix** (expected Hessian of log-likelihood):
$$\mathbf{F} = \mathbb{E}[\nabla \log p \cdot (\nabla \log p)^T]$$

Natural gradient: $\mathbf{F}^{-1} \nabla L$

Accounts for the geometry of probability distributions!

---

## 7. Jacobian-Vector Products (JVP) and VJP

### JVP (Forward Mode)

Compute $\mathbf{J}\mathbf{v}$ without forming full $\mathbf{J}$:
$$\mathbf{J}\mathbf{v} = \lim_{\epsilon \to 0} \frac{\mathbf{f}(\mathbf{x} + \epsilon\mathbf{v}) - \mathbf{f}(\mathbf{x})}{\epsilon}$$

### VJP (Reverse Mode / Backprop)

Compute $\mathbf{J}^T\mathbf{v}$ without forming full $\mathbf{J}$:

This is what backpropagation computes!

```
Forward vs Reverse Mode:

Forward (JVP):                 Reverse (VJP):
──────────────                 ──────────────

Input ───→ Layer ───→ Output   Input ←─── Layer ←─── Output
  │                              ↑                      ↑
  v                              │                      │
  ↓                            Jᵀv                      v
  Jv propagates forward        propagates backward

Efficient when:                Efficient when:
  outputs >> inputs              outputs << inputs
  (computing full Jacobian)      (neural nets: 1 scalar loss!)
```

> **💡 Why Reverse Mode Wins for Neural Nets**: We have millions of parameters but only 1 scalar loss. Reverse mode (backprop) needs just ONE backward pass!

---

## 8. Computing Jacobians and Hessians

### Numerical Jacobian

```python
def numerical_jacobian(f, x, h=1e-7):
    n = len(x)
    f_x = f(x)
    m = len(f_x)
    J = np.zeros((m, n))
    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += h
        J[:, j] = (f(x_plus) - f_x) / h
    return J
```

### Numerical Hessian

```python
def numerical_hessian(f, x, h=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
            x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h
            x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
            x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h
            H[i,j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4*h*h)
    return H
```

---

## 9. Summary

### Key Matrices

| Matrix | Dimension | Definition |
|--------|-----------|------------|
| Jacobian | $m \times n$ | $J_{ij} = \frac{\partial f_i}{\partial x_j}$ |
| Hessian | $n \times n$ | $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ |

### Key Results Cheat Sheet

```
┌───────────────────────────────────────────────────────────────┐
│                    KEY FORMULAS                               │
├───────────────────────────────────────────────────────────────┤
│  Chain rule:  J_{f∘g} = J_f × J_g  (matrix multiplication!)  │
│                                                               │
│  Taylor:  f(x+Δ) ≈ f(x) + ∇fᵀΔ + ½ΔᵀHΔ                       │
│                                                               │
│  Newton:  x_{k+1} = x_k - H⁻¹∇f                              │
│                                                               │
│  Backprop: ∂L/∂x = Jᵀ × (∂L/∂y)                              │
└───────────────────────────────────────────────────────────────┘
```

### ML Applications Summary

```
Jacobian and Hessian in ML:
│
├── Jacobian
│   ├── Backpropagation: Jᵀ × (∂L/∂y)
│   ├── Softmax gradient
│   └── Change of variables (normalizing flows)
│
├── Hessian
│   ├── Curvature of loss landscape
│   ├── Newton's method / quasi-Newton
│   ├── Second-order optimization
│   └── Saddle point detection (negative eigenvalues)
│
└── Efficient Computation
    ├── JVP: Forward mode autodiff
    └── VJP: Reverse mode (backprop)
```

---

## Exercises

1. Compute the Jacobian of $\mathbf{f}(x, y) = (x^2 - y, xy, e^x)$
2. Find the Hessian of $f(x, y) = x^4 + y^4 - 2x^2y^2$ and classify critical points
3. Derive the Jacobian of the softmax function
4. Show that the Hessian of $f(\mathbf{x}) = \mathbf{x}^T\mathbf{A}\mathbf{x}$ is $\mathbf{A} + \mathbf{A}^T$
5. Implement Newton's method to minimize $f(x, y) = (x-1)^2 + 10(y-x^2)^2$ (Rosenbrock)

---

## References

1. Magnus & Neudecker - "Matrix Differential Calculus"
2. Goodfellow et al. - "Deep Learning"
3. Boyd & Vandenberghe - "Convex Optimization"

---

> **Next**: [03-Chain-Rule-and-Backpropagation](../03-Chain-Rule-and-Backpropagation/) — Derivatives through compositions
