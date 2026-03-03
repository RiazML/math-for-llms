# Automatic Differentiation

> **Navigation**: [← 04-Optimization-Theory](../04-Optimization-Theory/) | [Multivariate Calculus](../) | [06-Probability-Theory →](../../06-Probability-Theory/)

**Files in this section:**

- [theory.ipynb](theory.ipynb) - 10 worked examples with NumPy + PyTorch
- [exercises.ipynb](exercises.ipynb) - 10 practice problems with solutions

---

## Introduction

Automatic differentiation (autodiff) is the **actual mechanism** that computes gradients in PyTorch, JAX, and TensorFlow. It is distinct from both symbolic differentiation (what Mathematica does) and numerical differentiation (finite differences). Understanding autodiff is essential because it explains **how** `loss.backward()` works, why computational graphs exist, and where gradient computation can fail.

> **Why this matters:** Karpathy's `micrograd` is a from-scratch implementation of reverse-mode autodiff. Every call to `.backward()` in PyTorch is running the algorithm described in this section.

## Prerequisites

- [03-Chain-Rule-and-Backpropagation](../03-Chain-Rule-and-Backpropagation/)
- [01-Partial-Derivatives-and-Gradients](../01-Partial-Derivatives-and-Gradients/)
- Basic Python (classes, operator overloading)

## Learning Objectives

1. Distinguish symbolic, numerical, and automatic differentiation
2. Implement forward-mode autodiff with dual numbers
3. Implement reverse-mode autodiff (backpropagation)
4. Understand computational graphs and topological sorting
5. Reason about memory/compute tradeoffs between forward and reverse mode

---

## 1. Three Ways to Compute Derivatives

### 1.1 Symbolic Differentiation

Apply calculus rules symbolically to produce a derivative expression.

**Example:** $f(x) = x^2 \sin(x)$

$$f'(x) = 2x\sin(x) + x^2\cos(x)$$

**Problem:** Expression swell — derivative expressions grow exponentially for deep compositions.

### 1.2 Numerical Differentiation

Approximate with finite differences:

$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$

**Problem:** Truncation error (large $h$) vs. round-off error (small $h$). Costs $O(n)$ function evaluations for $n$ parameters.

### 1.3 Automatic Differentiation

Decompose computation into elementary operations, apply chain rule to each.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 THREE DIFFERENTIATION METHODS                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Symbolic        Numerical         Automatic                           │
│  ─────────       ──────────        ──────────                          │
│  Exact formula   Approximate       Exact values                       │
│  Expression       f(x+h)-f(x-h)   Trace computation                  │
│  swell             ───────────     Apply chain rule                   │
│                       2h           per operation                       │
│                                                                         │
│  Used by:        Used by:          Used by:                            │
│  Mathematica     Gradient check    PyTorch, JAX,                      │
│  SymPy           Finite elements   TensorFlow                         │
│                                                                         │
│  Cost: varies    Cost: O(n) evals  Cost: O(1) × forward               │
│  Exact? Yes      Exact? No         Exact? Yes (to float precision)    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Forward-Mode Autodiff (Dual Numbers)

### 2.1 Dual Numbers

Define a number system $a + b\epsilon$ where $\epsilon^2 = 0$ (but $\epsilon \neq 0$):

$$(a + b\epsilon) + (c + d\epsilon) = (a+c) + (b+d)\epsilon$$
$$(a + b\epsilon) \cdot (c + d\epsilon) = ac + (ad + bc)\epsilon$$

The key property — evaluate $f$ at $x + \epsilon$:

$$f(x + \epsilon) = f(x) + f'(x)\epsilon$$

> **💡 Intuition**: Dual numbers carry a value AND its derivative through every computation automatically.

### 2.2 Elementary Operations

| Operation             | Primal | Tangent (derivative)                        |
| --------------------- | ------ | ------------------------------------------- |
| $v_i = v_j + v_k$     | $v_i$  | $\dot{v}_i = \dot{v}_j + \dot{v}_k$         |
| $v_i = v_j \cdot v_k$ | $v_i$  | $\dot{v}_i = \dot{v}_j v_k + v_j \dot{v}_k$ |
| $v_i = \sin(v_j)$     | $v_i$  | $\dot{v}_i = \cos(v_j) \dot{v}_j$           |
| $v_i = \exp(v_j)$     | $v_i$  | $\dot{v}_i = \exp(v_j) \dot{v}_j$           |
| $v_i = \log(v_j)$     | $v_i$  | $\dot{v}_i = \dot{v}_j / v_j$               |

### 2.3 Example: Forward Trace

For $f(x_1, x_2) = x_1 x_2 + \sin(x_1)$ at $(x_1, x_2) = (2, 3)$:

Seed $\dot{x}_1 = 1, \dot{x}_2 = 0$ (computing $\partial f / \partial x_1$):

| Step                  | Expression  | Value | Derivative                              |
| --------------------- | ----------- | ----- | --------------------------------------- |
| $v_1 = x_1$           | input       | 2     | $\dot{v}_1 = 1$                         |
| $v_2 = x_2$           | input       | 3     | $\dot{v}_2 = 0$                         |
| $v_3 = v_1 \cdot v_2$ | $x_1 x_2$   | 6     | $\dot{v}_3 = 1 \cdot 3 + 2 \cdot 0 = 3$ |
| $v_4 = \sin(v_1)$     | $\sin(x_1)$ | 0.909 | $\dot{v}_4 = \cos(2) \cdot 1 = -0.416$  |
| $v_5 = v_3 + v_4$     | $f$         | 6.909 | $\dot{v}_5 = 3 + (-0.416) = 2.584$      |

Result: $\partial f / \partial x_1 = 2.584$

> **⚠️ Forward mode computes one directional derivative per pass.** For $n$ inputs, need $n$ passes. Bad for neural networks ($n$ = millions of parameters).

---

## 3. Reverse-Mode Autodiff (Backpropagation)

### 3.1 The Key Idea

Instead of propagating derivatives forward, accumulate **adjoints** backward:

$$\bar{v}_i = \frac{\partial L}{\partial v_i}$$

where $L$ is the final scalar output (loss).

### 3.2 Reverse Trace

Same function $f(x_1, x_2) = x_1 x_2 + \sin(x_1)$:

**Forward pass** (compute values):
$$v_1 = 2, \quad v_2 = 3, \quad v_3 = 6, \quad v_4 = 0.909, \quad v_5 = 6.909$$

**Backward pass** (compute adjoints, starting from $\bar{v}_5 = 1$):

| Step                                     | Adjoint Rule          | Value                              |
| ---------------------------------------- | --------------------- | ---------------------------------- |
| $\bar{v}_5 = 1$                          | seed                  | 1                                  |
| $\bar{v}_3 += \bar{v}_5 \cdot 1$         | $v_5 = v_3 + v_4$     | $\bar{v}_3 = 1$                    |
| $\bar{v}_4 += \bar{v}_5 \cdot 1$         | $v_5 = v_3 + v_4$     | $\bar{v}_4 = 1$                    |
| $\bar{v}_1 += \bar{v}_3 \cdot v_2$       | $v_3 = v_1 \cdot v_2$ | $\bar{v}_1 = 3$                    |
| $\bar{v}_2 += \bar{v}_3 \cdot v_1$       | $v_3 = v_1 \cdot v_2$ | $\bar{v}_2 = 2$                    |
| $\bar{v}_1 += \bar{v}_4 \cdot \cos(v_1)$ | $v_4 = \sin(v_1)$     | $\bar{v}_1 = 3 + (-0.416) = 2.584$ |

**One pass gives ALL gradients:** $\partial f/\partial x_1 = 2.584$, $\partial f/\partial x_2 = 2$

> **🔑 THIS IS WHY BACKPROP WORKS**: Reverse mode computes gradients w.r.t. ALL inputs in a single backward pass. Cost = O(1) × forward pass, regardless of number of parameters.

### 3.3 Forward vs. Reverse Mode Complexity

| Property      | Forward Mode                                     | Reverse Mode                                    |
| ------------- | ------------------------------------------------ | ----------------------------------------------- |
| Computes      | $\partial \mathbf{f} / \partial x_i$ (one input) | $\partial L / \partial \mathbf{x}$ (all inputs) |
| Cost per pass | O(forward pass)                                  | O(forward pass)                                 |
| Passes needed | $n$ (num inputs)                                 | $m$ (num outputs)                               |
| Memory        | O(1) extra                                       | O(ops) — must store graph                       |
| Best when     | Few inputs, many outputs                         | Few outputs, many inputs                        |
| ML case       | Rarely used                                      | **Always** ($m=1$ scalar loss)                  |

```
Forward Mode (n passes):          Reverse Mode (1 pass):

  ∂f/∂x₁ ← pass 1                  ∂L/∂x₁ ┐
  ∂f/∂x₂ ← pass 2                  ∂L/∂x₂ ├── ALL from 1 pass!
  ∂f/∂x₃ ← pass 3                  ∂L/∂x₃ ┘
  ...
  ∂f/∂xₙ ← pass n                 Cost: O(forward) not O(n × forward)
```

---

## 4. Computational Graphs

### 4.1 Structure

Every computation builds a DAG (directed acyclic graph):

- **Leaves**: input variables, parameters
- **Internal nodes**: operations (+, ×, sin, ReLU, ...)
- **Root**: scalar loss

```
        loss
       /    \
      +
     / \
    *   sin
   / \   |
  x₁  x₂  x₁
```

### 4.2 Topological Sort

Backward pass processes nodes in **reverse topological order** — guaranteeing all downstream gradients are accumulated before propagating upstream.

### 4.3 In-Place Operations and Graph Invalidation

> **⚠️ Common Mistake:** In-place operations (`x += 1`, `x.relu_()`) break autograd because they modify values that the graph needs for backward.

```python
# WRONG — will error on backward
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
x += 1  # modifies x in-place, graph is now invalid
y.backward()  # RuntimeError!

# RIGHT — create new tensor
x = torch.tensor([1.0], requires_grad=True)
y = x ** 2
x = x + 1  # new tensor, original x in graph is preserved
```

---

## 5. Autodiff in PyTorch

### 5.1 Core API

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + torch.sin(x)
y.backward()          # reverse-mode autodiff
print(x.grad)         # dy/dx = 2x + cos(x) = 4 + cos(2) ≈ 3.584
```

### 5.2 Gradient Accumulation

Gradients **accumulate** by default (important for RNNs, gradient accumulation over mini-batches):

```python
x.grad.zero_()  # Must zero before each backward pass
```

### 5.3 Detach and No-Grad

```python
y = x.detach()           # Removes from graph (for frozen weights)
with torch.no_grad():    # Disables tracking (for inference)
    pred = model(x)
```

### 5.4 Custom Autograd Functions

```python
class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
```

---

## 6. Gradient Checking

Numerical differentiation verifies autodiff correctness:

$$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta + h\mathbf{e}_i) - L(\theta - h\mathbf{e}_i)}{2h}$$

Relative error check:

$$\text{rel\_error} = \frac{|\text{analytic} - \text{numerical}|}{|\text{analytic}| + |\text{numerical}| + \epsilon}$$

| Relative Error         | Meaning                            |
| ---------------------- | ---------------------------------- |
| $< 10^{-7}$            | Correct                            |
| $10^{-5}$ to $10^{-7}$ | Possibly correct, check edge cases |
| $> 10^{-3}$            | Bug in gradient computation        |

> **💡 Always gradient-check custom autograd functions.** This is how Karpathy validates `micrograd`.

---

## 7. Advanced Topics

### 7.1 Higher-Order Derivatives

PyTorch supports `create_graph=True` to differentiate through the backward pass:

$$\nabla^2 L \cdot \mathbf{v} = \frac{\partial}{\partial \mathbf{\theta}} (\nabla L \cdot \mathbf{v})$$

Used for: Hessian-vector products, MAML (meta-learning), natural gradient.

### 7.2 Jacobian-Vector Products (JVP) and Vector-Jacobian Products (VJP)

| Operation | Mode    | Computes                  | API                             |
| --------- | ------- | ------------------------- | ------------------------------- |
| JVP       | Forward | $\mathbf{J} \mathbf{v}$   | `torch.autograd.functional.jvp` |
| VJP       | Reverse | $\mathbf{v}^T \mathbf{J}$ | `torch.autograd.functional.vjp` |

### 7.3 Checkpointing (Memory-Compute Tradeoff)

For very deep networks, store only some intermediate activations and recompute others during backward:

$$\text{Memory: } O(\sqrt{n}) \quad \text{instead of} \quad O(n)$$

```
Standard:      Store all activations → high memory
Checkpointing: Store every √n-th → recompute in between → low memory
```

---

## ❌ Common Mistakes → ✅ Fixes

- ❌ **Forgetting `zero_grad()`** → accumulated gradients give wrong updates
  ✅ Always call `optimizer.zero_grad()` before `loss.backward()`

- ❌ **In-place operations** break the graph silently
  ✅ Use `x = x + 1` not `x += 1` for tensors with `requires_grad=True`

- ❌ **Detaching when you shouldn't** → kills gradient flow
  ✅ Only detach for frozen components (pretrained encoders, target networks in RL)

- ❌ **Not using `torch.no_grad()` during inference** → wastes memory on graph
  ✅ Wrap inference in `with torch.no_grad():`

- ❌ **Numerical gradient check with too-small $h$** → round-off error dominates
  ✅ Use $h \approx 10^{-5}$ for float32

---

## Where This Appears in Real Models

| Model / System         | How Autodiff Is Used                                     |
| ---------------------- | -------------------------------------------------------- |
| PyTorch `.backward()`  | Reverse-mode autodiff over dynamic computational graph   |
| JAX `grad()`           | Functional transforms: `jax.grad`, `jax.jvp`, `jax.vjp`  |
| Karpathy's `micrograd` | Minimal reverse-mode autodiff engine (~100 lines)        |
| MAML (meta-learning)   | Differentiates through the optimization loop (2nd order) |
| Neural ODEs            | Adjoint method — continuous-time reverse-mode autodiff   |
| Physics-informed NNs   | Autodiff computes PDE residuals                          |

---

## Key Takeaways

1. **Autodiff ≠ finite differences.** It computes exact derivatives (to floating-point precision) by tracing operations and applying the chain rule.
2. **Reverse mode** is what makes training neural networks with millions of parameters feasible — one backward pass gives all gradients.
3. **Computational graphs** are not just data structures — they ARE the autodiff algorithm's working memory.
4. **`micrograd` is the minimal proof** that you understand this: a Value class with `__add__`, `__mul__`, and `backward()`.
5. **Gradient checking** with finite differences is the gold standard for verifying correctness.

---

## References

- Baydin et al., "Automatic Differentiation in Machine Learning: a Survey" (2018)
- Karpathy, `micrograd` — https://github.com/karpathy/micrograd
- Griewank & Walther, _Evaluating Derivatives_ (2008)
- PyTorch Autograd docs — https://pytorch.org/docs/stable/autograd.html
