# Floating-Point Arithmetic

> **Navigation**: [← 04-Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/) | [Numerical Methods](../) | [02-Numerical-Linear-Algebra →](../02-Numerical-Linear-Algebra/)

**Files in this section:**
- [theory.ipynb](theory.ipynb) - 12 worked examples
- [exercises.ipynb](exercises.ipynb) - 10 practice problems with solutions

---

## Learning Objectives

- Understand IEEE 754 floating-point representation
- Recognize sources of numerical error
- Master techniques for numerical stability
- Apply best practices for ML implementations

## Prerequisites

- Binary number systems
- Basic calculus
- Linear algebra fundamentals

---

## 1. IEEE 754 Floating-Point Representation

### Structure

A floating-point number is represented as:

$$x = (-1)^s \times m \times 2^{e-\text{bias}}$$

Where:

- $s$ = sign bit (0 = positive, 1 = negative)
- $m$ = mantissa (significand)
- $e$ = exponent
- bias = $2^{k-1} - 1$ (k = exponent bits)

```
IEEE 754 Single Precision (32-bit):
┌─────┬──────────────┬───────────────────────────┐
│  S  │   Exponent   │        Mantissa           │
│ 1b  │     8 bits   │        23 bits            │
└─────┴──────────────┴───────────────────────────┘
  │         │                    │
  │         │                    └── Fractional part (implicit 1.)
  │         └── Biased exponent (bias = 127)
  └── Sign bit

IEEE 754 Double Precision (64-bit):
┌─────┬──────────────────┬────────────────────────────────────────┐
│  S  │    Exponent      │              Mantissa                  │
│ 1b  │     11 bits      │              52 bits                   │
└─────┴──────────────────┴────────────────────────────────────────┘
  │           │                           │
  │           │                           └── ~15-17 decimal digits
  │           └── Biased exponent (bias = 1023)
  └── Sign bit
```

### Special Values

```
Special Floating-Point Values:
┌─────────────────────┬─────────────┬─────────────┐
│ Value               │ Exponent    │ Mantissa    │
├─────────────────────┼─────────────┼─────────────┤
│ Zero (±0)           │ 0           │ 0           │
│ Denormalized        │ 0           │ ≠ 0         │
│ Infinity (±∞)       │ All 1s      │ 0           │
│ NaN                 │ All 1s      │ ≠ 0         │
│ Normal numbers      │ 1 to 2^k-2  │ any         │
└─────────────────────┴─────────────┴─────────────┘
```

### Precision Limits

| Type    | Bits | Significand | Decimal Digits | Range      |
| ------- | ---- | ----------- | -------------- | ---------- |
| float16 | 16   | 11          | ~3.3           | ±65504     |
| float32 | 32   | 24          | ~7.2           | ±3.4×10³⁸  |
| float64 | 64   | 53          | ~15.9          | ±1.8×10³⁰⁸ |

---

## 2. Machine Epsilon

### Definition

Machine epsilon ($\epsilon_{\text{mach}}$) is the smallest number such that:

$$1 + \epsilon_{\text{mach}} > 1$$

```
Machine Epsilon by Type:
┌─────────────┬─────────────────────┐
│ Type        │ Machine Epsilon     │
├─────────────┼─────────────────────┤
│ float16     │ ≈ 9.77 × 10⁻⁴       │
│ float32     │ ≈ 1.19 × 10⁻⁷       │
│ float64     │ ≈ 2.22 × 10⁻¹⁶      │
└─────────────┴─────────────────────┘
```

### Relative Error Bound

For any real number $x$ in the normal range, its floating-point representation $\text{fl}(x)$ satisfies:

$$\left|\frac{\text{fl}(x) - x}{x}\right| \leq \frac{\epsilon_{\text{mach}}}{2}$$

---

## 3. Sources of Numerical Error

### 3.1 Rounding Errors

```
Rounding Error Example:
┌─────────────────────────────────────────────────────┐
│                                                     │
│  0.1 in binary = 0.0001100110011... (repeating)     │
│                                                     │
│  Cannot be exactly represented!                     │
│                                                     │
│  float(0.1) + float(0.2) ≠ 0.3                     │
│  Result: 0.30000000000000004                        │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 3.2 Catastrophic Cancellation

When subtracting nearly equal numbers:

$$a = 1.234567890123456$$
$$b = 1.234567890123455$$
$$a - b = 1.0 \times 10^{-15}$$

```
Cancellation Error:
┌─────────────────────────────────────────────────────┐
│                                                     │
│  a =  1.234567890123456                            │
│  b =  1.234567890123455                            │
│  ─────────────────────────                          │
│  a-b = 0.000000000000001                           │
│                                                     │
│  Only 1 significant digit remains!                  │
│  Relative error can be huge                         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 3.3 Overflow and Underflow

```
Overflow/Underflow in Softmax:
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Naive softmax: exp(x_i) / Σ exp(x_j)              │
│                                                     │
│  If x = [1000, 1001, 1002]:                        │
│    exp(1000) = Inf  → NaN result!                  │
│                                                     │
│  Stable softmax: exp(x_i - max(x)) / Σ exp(...)    │
│                                                     │
│  If x = [-1000, -999, -998]:                       │
│    exp(-1000) ≈ 0  → 0/0 = NaN                     │
│                                                     │
│  Solution: Subtract max before exp                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 3.4 Accumulation Error

```
Accumulation Error Example:
┌─────────────────────────────────────────────────────┐
│                                                     │
│  sum = 0.0                                         │
│  for i in range(1_000_000):                        │
│      sum += 0.1                                    │
│                                                     │
│  Expected: 100000.0                                │
│  Actual:   100000.00000133288                      │
│                                                     │
│  Error accumulates with each addition!              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 4. Numerical Stability Techniques

### 4.1 Log-Sum-Exp Trick

For computing $\log\sum_i \exp(x_i)$:

$$\text{logsumexp}(x) = m + \log\sum_i \exp(x_i - m)$$

where $m = \max(x)$

```python
# Unstable
def logsumexp_unstable(x):
    return np.log(np.sum(np.exp(x)))  # Overflow!

# Stable
def logsumexp_stable(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))
```

### 4.2 Numerically Stable Softmax

$$\text{softmax}(x_i) = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}$$

where $m = \max(x)$

### 4.3 Numerically Stable Sigmoid

$$
\sigma(x) = \begin{cases}
\frac{1}{1 + e^{-x}} & \text{if } x \geq 0 \\
\frac{e^x}{1 + e^x} & \text{if } x < 0
\end{cases}
$$

### 4.4 Log Probabilities

Instead of multiplying probabilities (underflow):
$$P(A,B,C) = P(A) \cdot P(B) \cdot P(C)$$

Use log probabilities (addition):
$$\log P(A,B,C) = \log P(A) + \log P(B) + \log P(C)$$

---

## 5. Kahan Summation Algorithm

For more accurate summation:

```
Kahan Summation:
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Algorithm:                                         │
│  ──────────                                         │
│  sum = 0.0                                         │
│  c = 0.0  (compensation for lost low-order bits)   │
│                                                     │
│  for each x in input:                              │
│      y = x - c              # Compensate           │
│      t = sum + y            # Alas, sum is big     │
│      c = (t - sum) - y      # Recover lost bits    │
│      sum = t                                        │
│                                                     │
│  return sum                                         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 6. Condition Number

### Definition

The condition number measures sensitivity to input perturbations:

$$\kappa(A) = ||A|| \cdot ||A^{-1}||$$

```
Condition Number Interpretation:
┌─────────────────────────────────────────────────────┐
│                                                     │
│  κ(A) ≈ 1:     Well-conditioned                    │
│  κ(A) ~ 10⁶:   May lose ~6 digits of precision     │
│  κ(A) ~ 10¹⁶:  Essentially singular (for float64)  │
│                                                     │
│  Rule of thumb:                                     │
│  Lost digits ≈ log₁₀(κ(A))                         │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### For Linear Systems

Solving $Ax = b$ with perturbation $\delta b$:

$$\frac{||\delta x||}{||x||} \leq \kappa(A) \frac{||\delta b||}{||b||}$$

---

## 7. Mixed Precision Training

### Half Precision (float16) in Deep Learning

```
Mixed Precision Strategy:
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Master Weights:    float32                        │
│  Forward Pass:      float16                        │
│  Loss Computation:  float32                        │
│  Backward Pass:     float16                        │
│  Weight Update:     float32                        │
│                                                     │
│  Benefits:                                          │
│  • 2x memory reduction for activations             │
│  • Faster tensor core operations                   │
│  • Similar accuracy with loss scaling              │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Loss Scaling

To prevent gradient underflow in float16:

$$\text{scaled\_loss} = \text{loss} \times S$$
$$\text{gradients} = \nabla(\text{scaled\_loss}) / S$$

where $S$ is typically 512-65536.

---

## 8. Best Practices for ML

### Numerical Stability Checklist

```
✓ Use log-domain for probabilities
✓ Subtract max before softmax/logsumexp
✓ Check for NaN/Inf during training
✓ Use appropriate dtypes (float32 for most, float64 for sensitive)
✓ Clip gradients to prevent explosion
✓ Add epsilon to denominators (1e-8 typical)
✓ Use numerically stable loss functions
✓ Monitor condition numbers for matrix operations
```

### Common Pitfalls

```
Common Numerical Issues in ML:
┌─────────────────────────────────────────────────────┐
│                                                     │
│  1. Division by zero:                              │
│     Bad:  x / y                                    │
│     Good: x / (y + eps)                            │
│                                                     │
│  2. Log of zero:                                   │
│     Bad:  log(p)                                   │
│     Good: log(p + eps) or log(max(p, eps))        │
│                                                     │
│  3. Sqrt of negative:                              │
│     Bad:  sqrt(x)                                  │
│     Good: sqrt(max(x, 0)) or sqrt(abs(x) + eps)   │
│                                                     │
│  4. Large exponentials:                            │
│     Bad:  exp(x) where x > 700                     │
│     Good: exp(clip(x, -700, 700))                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 9. Summary

### Key Formulas

| Operation     | Stable Form                                        |
| ------------- | -------------------------------------------------- |
| Softmax       | $\exp(x_i - \max(x)) / \sum \exp(x_j - \max(x))$   |
| Log-sum-exp   | $\max(x) + \log\sum\exp(x - \max(x))$              |
| Cross-entropy | Use built-in with logits, not softmax              |
| Variance      | $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ → use two-pass |
| Division      | $x / (y + \epsilon)$                               |

### When to Use Each Precision

| Precision | Use Case                                               |
| --------- | ------------------------------------------------------ |
| float16   | Forward pass, large models, inference                  |
| float32   | Default training, most operations                      |
| float64   | Sensitive numerical computations, small cumulative ops |

---

## Exercises

### Exercise 1: Machine Epsilon
Compute and compare machine epsilon for float16, float32, and float64. Verify your results experimentally.

### Exercise 2: Catastrophic Cancellation
Demonstrate catastrophic cancellation in the quadratic formula for $x^2 + 10^8x + 1 = 0$. Implement a numerically stable alternative.

### Exercise 3: Summation Algorithms
Compare naive, pairwise, and Kahan summation for adding $10^7$ random numbers. Measure relative errors.

### Exercise 4: Loss Scaling
Implement automatic mixed precision training with dynamic loss scaling for a simple neural network.

### Exercise 5: Numerical Stability Analysis
Analyze the condition number of a Hilbert matrix and explain why solving linear systems with it becomes unstable.

---

## References

1. Goldberg (1991). "What Every Computer Scientist Should Know About Floating-Point"
2. Higham (2002). "Accuracy and Stability of Numerical Algorithms"
3. Micikevicius et al. (2018). "Mixed Precision Training"
4. IEEE 754-2019 Standard for Floating-Point Arithmetic
