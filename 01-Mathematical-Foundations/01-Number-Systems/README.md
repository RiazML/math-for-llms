# Number Systems

## Overview

Understanding number systems is the bedrock of all mathematical concepts in machine learning. From representing data to performing computations, every ML algorithm relies on different types of numbers.

## Prerequisites

- Basic arithmetic operations
- Understanding of fractions and decimals

## Learning Objectives

After completing this section, you will:

- Understand the hierarchy of number systems
- Know how different number types are used in ML
- Recognize limitations of computer number representation
- Understand complex numbers and their role in ML

---

## Theory

### The Number System Hierarchy

```
                    ┌─────────────────┐
                    │    Complex ℂ    │  a + bi
                    │  (includes i)   │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │     Real ℝ      │  -∞ to +∞
                    │ (number line)   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────┴───────┐      │     ┌────────┴───────┐
     │  Irrational    │      │     │   Rational ℚ   │
     │   π, e, √2     │      │     │    p/q form    │
     └────────────────┘      │     └────────┬───────┘
                             │              │
                    ┌────────┴───────┐      │
                    │   Integer ℤ    │◄─────┘
                    │  ...-2,-1,0,1..│
                    └────────┬───────┘
                             │
                    ┌────────┴───────┐
                    │   Natural ℕ    │
                    │    0,1,2,3...  │
                    └────────────────┘
```

### 1. Natural Numbers (ℕ)

**Definition:** The counting numbers, typically {0, 1, 2, 3, ...} or {1, 2, 3, ...}

$$\mathbb{N} = \{0, 1, 2, 3, 4, ...\}$$

**ML Applications:**

- Counting: number of samples, features, classes
- Indices: array positions, batch numbers
- Discrete outputs: classification labels

**Example:**

```python
n_samples = 1000      # Number of training examples
n_features = 784      # Dimensions in MNIST
n_classes = 10        # Output classes
```

### 2. Integers (ℤ)

**Definition:** All positive and negative whole numbers including zero.

$$\mathbb{Z} = \{..., -3, -2, -1, 0, 1, 2, 3, ...\}$$

**ML Applications:**

- Indexing with negative indices (Python)
- Integer quantization for model compression
- Discrete action spaces in RL

### 3. Rational Numbers (ℚ)

**Definition:** Numbers expressible as a fraction p/q where p, q ∈ ℤ and q ≠ 0.

$$\mathbb{Q} = \left\{\frac{p}{q} : p, q \in \mathbb{Z}, q \neq 0\right\}$$

**Properties:**

- Dense: between any two rationals, there's another rational
- Countable: can be put in one-to-one correspondence with ℕ

### 4. Real Numbers (ℝ)

**Definition:** All numbers on the continuous number line.

$$\mathbb{R} = \mathbb{Q} \cup \text{Irrationals}$$

**Includes irrational numbers:**

- $\pi \approx 3.14159...$
- $e \approx 2.71828...$
- $\sqrt{2} \approx 1.41421...$

**ML Applications:**

- Model weights and biases
- Continuous features
- Loss values
- Probabilities (restricted to [0,1])

### 5. Complex Numbers (ℂ)

**Definition:** Numbers of the form a + bi where i² = -1.

$$\mathbb{C} = \{a + bi : a, b \in \mathbb{R}, i^2 = -1\}$$

**Euler's Formula:**
$$e^{i\theta} = \cos\theta + i\sin\theta$$

**ML Applications:**

- Fourier transforms for signal processing
- Eigenvalues of non-symmetric matrices
- Quantum machine learning

---

## Computer Number Representation

### Floating Point Numbers (IEEE 754)

```
32-bit float (single precision):
┌───┬──────────┬───────────────────────┐
│ S │ Exponent │       Mantissa        │
│ 1 │  8 bits  │       23 bits         │
└───┴──────────┴───────────────────────┘

64-bit float (double precision):
┌───┬──────────┬────────────────────────────────────────────┐
│ S │ Exponent │                  Mantissa                  │
│ 1 │ 11 bits  │                  52 bits                   │
└───┴──────────┴────────────────────────────────────────────┘

Value = (-1)^S × 2^(Exponent - bias) × (1 + Mantissa)
```

### Important Limits

| Type    | Min Positive | Max        | Precision (decimal digits) |
| ------- | ------------ | ---------- | -------------------------- |
| float16 | ~6×10⁻⁸      | ~65504     | ~3                         |
| float32 | ~1.2×10⁻³⁸   | ~3.4×10³⁸  | ~7                         |
| float64 | ~2.2×10⁻³⁰⁸  | ~1.8×10³⁰⁸ | ~16                        |

### Machine Epsilon

The smallest ε such that 1.0 + ε ≠ 1.0

| Type    | Machine Epsilon |
| ------- | --------------- |
| float32 | ~1.19×10⁻⁷      |
| float64 | ~2.22×10⁻¹⁶     |

**Why it matters for ML:**

- Numerical stability in gradient computation
- Avoiding division by zero (adding epsilon)
- Understanding precision limits in optimization

---

## Why This Matters for ML

### 1. Data Types Affect Memory and Speed

| Operation           | float32 | float64  |
| ------------------- | ------- | -------- |
| Memory per value    | 4 bytes | 8 bytes  |
| GPU optimization    | Better  | Standard |
| Training speed      | Faster  | Slower   |
| Numerical precision | Lower   | Higher   |

### 2. Numerical Overflow and Underflow

**Overflow (too large):**

```python
# Softmax without numerical stability
exp(1000)  # inf!

# Solution: subtract max
exp(x - max(x))
```

**Underflow (too small):**

```python
# Product of many probabilities
0.001 * 0.001 * ... * 0.001  # → 0

# Solution: use log probabilities
log(0.001) + log(0.001) + ...
```

### 3. Mixed Precision Training

Modern deep learning uses mixed precision:

- Forward pass: float16 (speed)
- Gradients: float16 (speed)
- Weight update: float32 (precision)
- Master weights: float32 (precision)

---

## Key Formulas

| Concept           | Formula                                  | Description          |
| ----------------- | ---------------------------------------- | -------------------- |
| Complex magnitude | $\|a + bi\| = \sqrt{a^2 + b^2}$          | Distance from origin |
| Complex conjugate | $\overline{a + bi} = a - bi$             | Flip imaginary part  |
| Euler's formula   | $e^{i\theta} = \cos\theta + i\sin\theta$ | Exponential to trig  |
| Machine epsilon   | $\epsilon: 1 + \epsilon \neq 1$          | Precision limit      |

---

## Common Pitfalls

### 1. Floating Point Comparison

```python
# Wrong
0.1 + 0.2 == 0.3  # False!

# Correct
import numpy as np
np.isclose(0.1 + 0.2, 0.3)  # True
```

### 2. Integer Division

```python
# Python 3
7 / 2   # 3.5 (float division)
7 // 2  # 3 (integer division)
```

### 3. Overflow in Intermediate Calculations

```python
# Can overflow
large = 1e308
result = large * large / large  # inf

# Better
result = large * (large / large)  # 1e308
```

---

## Interview Questions

1. **Q: Why do we add a small epsilon in many ML operations?**
   A: To prevent division by zero and improve numerical stability. For example, in batch normalization: $\frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$

2. **Q: What's the difference between float32 and float64 in deep learning?**
   A: float32 uses half the memory and is often faster on GPUs, while float64 provides more precision. Most deep learning uses float32 or even float16.

3. **Q: Why might training loss become NaN?**
   A: Usually due to numerical overflow (gradients too large), underflow (very small probabilities), or division by zero.

---

## Further Reading

- 📺 [3Blue1Brown - What is e?](https://www.youtube.com/watch?v=m2MIpDrF7Es)
- 📺 [Computerphile - Floating Point Numbers](https://www.youtube.com/watch?v=PZRI1IfStY0)
- 📖 [IEEE 754 Standard](https://en.wikipedia.org/wiki/IEEE_754)
- 📖 [Mixed Precision Training (NVIDIA)](https://developer.nvidia.com/automatic-mixed-precision)

---

## Next Steps

After mastering number systems, proceed to:
→ [Sets and Logic](../02-Sets-and-Logic/README.md)
