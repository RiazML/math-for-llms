# Number Systems

## Overview

Understanding number systems is the bedrock of all mathematical concepts in machine learning. From representing data to performing computations, every ML algorithm relies on different types of numbers. This comprehensive guide explores number systems through an **ML-focused lens**, emphasizing practical applications in deep learning, neural networks, and model optimization.

## Prerequisites

- Basic arithmetic operations (addition, subtraction, multiplication, division)
- Understanding of fractions and decimals — see Khan Academy: https://www.khanacademy.org/math/arithmetic/fraction-arithmetic
- Familiarity with Python and NumPy basics (recommended: NumPy dtypes overview: https://numpy.org/doc/stable/user/basics.types.html)

## Learning Objectives

After completing this section you will be able to:

- Explain the hierarchy of number systems with examples 
- Describe IEEE-754 floating-point basics and reproduce common precision anomalies 
- Demonstrate two's-complement integer representation and show overflow behavior 
- Implement a simple int8 quantization and measure quantization error in NumPy 
- Use complex numbers for a basic discrete Fourier transform example 
- Identify numerical stability issues (overflow, underflow, catastrophic cancellation) and apply mitigation strategies 

These objectives are measurable and suitable for a self-study session; times are approximate.

### Quick Examples

```python
# Floating point precision
print(0.1 + 0.2 == 0.3)  # False in IEEE-754 binary floats

# Two's-complement wrap-around using int8 (NumPy)
import numpy as np
print(np.int8(127) + np.int8(1))  # results in -128 (wrap)

# Simple int8 quantization
def quantize_int8(x, scale=127.0):
    q = np.round(np.clip(x * scale, -128, 127)).astype(np.int8)
    return q

print(quantize_int8(np.array([0.0, 0.5, -0.5, 1.0])))
```

### Exercises (suggested)

- Implement a NumPy function that converts float32 weights to int8 and computes mean absolute error.
- Visualize binary representations of 0.1 and 0.2, and explain why their sum isn't exactly 0.3.
- Simulate integer overflow for different bit widths and explain consequences for model training or indexing.

### References

- IEEE-754 overview: https://en.wikipedia.org/wiki/IEEE_754
- "What Every Computer Scientist Should Know About Floating-Point" — David Goldberg: https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html
- NumPy dtype reference: https://numpy.org/doc/stable/user/basics.types.html

---

## Table of Contents

1. [The Number System Hierarchy](#the-number-system-hierarchy)
2. [Natural Numbers (ℕ)](#1-natural-numbers-ℕ)
3. [Integers (ℤ)](#2-integers-ℤ)
4. [Rational Numbers (ℚ)](#3-rational-numbers-ℚ)
5. [Real Numbers (ℝ)](#4-real-numbers-ℝ)
6. [Complex Numbers (ℂ)](#5-complex-numbers-ℂ)
7. [Number Bases](#number-bases)
8. [Computer Number Representation](#computer-number-representation)
9. [Quantization for ML](#quantization-for-ml)
10. [Numerical Stability in Deep Learning](#numerical-stability-in-deep-learning)
11. [Special Floating Point Values](#special-floating-point-values)
12. [Mixed Precision Training](#mixed-precision-training)
13. [Common Pitfalls](#common-pitfalls)
14. [Interview Questions](#interview-questions)
15. [Further Reading](#further-reading)

---

## Theory

### The Number System Hierarchy

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                           NUMBER SYSTEMS HIERARCHY                            ║
║                     (Each level contains all levels below)                    ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║                         ┌─────────────────────────┐                           ║
║                         │      COMPLEX (ℂ)        │                           ║
║                         │   a + bi where i²=-1    │                           ║
║                         │  Examples: 3+2i, -i, 5  │                           ║
║                         │                         │                           ║
║                         │  ML: Fourier Transform  │                           ║
║                         │      Signal Processing  │                           ║
║                         │      Eigenvalues        │                           ║
║                         └───────────┬─────────────┘                           ║
║                                     │                                         ║
║                         ┌───────────▼─────────────┐                           ║
║                         │       REAL (ℝ)          │                           ║
║                         │   Continuous number     │                           ║
║                         │   line: -∞ to +∞        │                           ║
║                         │                         │                           ║
║                         │  ML: Weights, Biases    │                           ║
║                         │      Loss Values        │                           ║
║                         │      Probabilities      │                           ║
║                         └───────────┬─────────────┘                           ║
║                                     │                                         ║
║               ┌─────────────────────┼─────────────────────┐                   ║
║               │                     │                     │                   ║
║    ┌──────────▼──────────┐          │          ┌──────────▼──────────┐        ║
║    │    IRRATIONAL       │          │          │    RATIONAL (ℚ)     │        ║
║    │   Cannot be p/q     │          │          │   Can be p/q        │        ║
║    │                     │          │          │   where q ≠ 0       │        ║
║    │   π = 3.14159...    │          │          │                     │        ║
║    │   e = 2.71828...    │          │          │   1/2, 0.75, -3/4   │        ║
║    │   √2 = 1.41421...   │          │          │   0.333... = 1/3    │        ║
║    │   φ = 1.61803...    │          │          │                     │        ║
║    │                     │          │          │   ML: Learning Rate │        ║
║    │   ML: Activation    │          │          │       Batch Ratios  │        ║
║    │       functions     │          │          │       Percentages   │        ║
║    └─────────────────────┘          │          └──────────┬──────────┘        ║
║                                     │                     │                   ║
║                         ┌───────────▼─────────────────────▼───┐               ║
║                         │         INTEGERS (ℤ)                │               ║
║                         │   ..., -3, -2, -1, 0, 1, 2, 3, ...  │               ║
║                         │                                     │               ║
║                         │   ML: Quantized Weights             │               ║
║                         │       Token IDs, Indices            │               ║
║                         │       Embedding Lookups             │               ║
║                         └───────────────┬─────────────────────┘               ║
║                                         │                                     ║
║                         ┌───────────────▼─────────────────────┐               ║
║                         │       NATURAL NUMBERS (ℕ)           │               ║
║                         │       0, 1, 2, 3, 4, 5, ...         │               ║
║                         │       (Counting numbers)            │               ║
║                         │                                     │               ║
║                         │   ML: Batch Size, Epochs            │               ║
║                         │       Layer Count, Class Labels     │               ║
║                         │       Vocabulary Size               │               ║
║                         └─────────────────────────────────────┘               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### Subset Relationships

```
ℕ ⊂ ℤ ⊂ ℚ ⊂ ℝ ⊂ ℂ

Natural ⊂ Integer ⊂ Rational ⊂ Real ⊂ Complex

Every natural number is an integer
Every integer is a rational (n = n/1)
Every rational is a real
Every real is a complex (a = a + 0i)
```

---

### 1. Natural Numbers (ℕ)

**Definition:** The counting numbers, starting from zero (or one in some definitions).

$$\mathbb{N} = \{0, 1, 2, 3, 4, 5, ...\}$$

#### Visual Representation

```
THE NATURAL NUMBERS - COUNTING FROM ZERO TO INFINITY
═══════════════════════════════════════════════════════════════════════

    0     1     2     3     4     5     6     7     8     9    ...
    ●─────●─────●─────●─────●─────●─────●─────●─────●─────●─────────►
    │     │     │     │     │     │     │     │     │     │
    │     │     │     │     │     │     │     │     │     │
  Zero  First Second Third Fourth Fifth Sixth Seventh Eighth Ninth

PROPERTIES:
┌─────────────────────────────────────────────────────────────────────┐
│  ✓ Closed under ADDITION:        3 + 5 = 8    (still natural)      │
│  ✓ Closed under MULTIPLICATION:  3 × 5 = 15   (still natural)      │
│  ✗ NOT closed under SUBTRACTION: 3 - 5 = -2   (not natural!)       │
│  ✗ NOT closed under DIVISION:    3 ÷ 5 = 0.6  (not natural!)       │
└─────────────────────────────────────────────────────────────────────┘
```

#### ML Applications

| Application            | Example                | Why Natural Numbers?         |
| ---------------------- | ---------------------- | ---------------------------- |
| **Sample Count**       | `n_samples = 10000`    | Cannot have negative samples |
| **Feature Dimensions** | `n_features = 784`     | MNIST image pixels           |
| **Class Labels**       | `labels ∈ {0,1,...,9}` | Discrete categories          |
| **Epochs**             | `epochs = 100`         | Training iterations          |
| **Batch Size**         | `batch_size = 32`      | Samples per update           |
| **Layer Count**        | `n_layers = 12`        | Transformer depth            |
| **Vocabulary Size**    | `vocab_size = 50257`   | GPT-2 tokens                 |
| **Sequence Length**    | `max_seq_len = 512`    | Token positions              |

#### Key Properties for ML

```
PEANO AXIOMS (Foundation of Natural Numbers)
════════════════════════════════════════════

1. 0 is a natural number
2. Every natural number n has a successor S(n)
3. 0 is not the successor of any natural number
4. If S(n) = S(m), then n = m
5. Mathematical Induction holds

WHY THIS MATTERS FOR ML:
─────────────────────────
• Induction → Proves correctness of recursive algorithms
• Successor function → Basis for iteration (epochs, steps)
• Well-ordering → Guarantees termination of training loops
```

#### Simple Code Example

```python
# Natural numbers in ML context
n_samples = 1000        # Training examples
n_features = 784        # Input dimensions (28×28 pixels)
n_classes = 10          # Output classes (digits 0-9)
n_epochs = 50           # Training iterations
batch_size = 32         # Samples per gradient update
n_batches = n_samples // batch_size  # 31 complete batches
```

---

### 2. Integers (ℤ)

**Definition:** All whole numbers, positive, negative, and zero.

$$\mathbb{Z} = \{..., -3, -2, -1, 0, 1, 2, 3, ...\}$$

#### Visual Representation

```
THE INTEGER NUMBER LINE
═══════════════════════════════════════════════════════════════════════

    ◄────●─────●─────●─────●─────●─────●─────●─────●─────●─────●────►
        -4    -3    -2    -1     0     1     2     3     4     5
                              ORIGIN
         ◄─────────────────────┼─────────────────────►
          Negative Integers    │    Positive Integers
                            Zero (neither positive
                                  nor negative)

PROPERTIES:
┌─────────────────────────────────────────────────────────────────────┐
│  ✓ Closed under ADDITION:        3 + (-5) = -2   (still integer)   │
│  ✓ Closed under SUBTRACTION:     3 - 5 = -2      (still integer)   │
│  ✓ Closed under MULTIPLICATION:  3 × (-5) = -15  (still integer)   │
│  ✗ NOT closed under DIVISION:    3 ÷ 5 = 0.6    (not integer!)     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Two's Complement Representation (How Computers Store Integers)

```
8-BIT TWO'S COMPLEMENT
═══════════════════════════════════════════════════════════════════════

POSITIVE NUMBERS (0 to 127):
    0 = 00000000
    1 = 00000001
    2 = 00000010
   42 = 00101010
  127 = 01111111  ← Maximum positive

NEGATIVE NUMBERS (-128 to -1):
   -1 = 11111111  ← All ones = -1
   -2 = 11111110
  -42 = 11010110
 -128 = 10000000  ← Minimum negative

┌─────────────────────────────────────────────────────────────────────┐
│  TO NEGATE A NUMBER:                                                │
│  1. Flip all bits (ones' complement)                                │
│  2. Add 1                                                           │
│                                                                     │
│  Example: -42                                                       │
│    42  = 00101010                                                   │
│   flip = 11010101                                                   │
│    +1  = 11010110 = -42                                             │
└─────────────────────────────────────────────────────────────────────┘

INTEGER RANGES BY BIT WIDTH:
┌──────────┬─────────────────────┬──────────────────────────────────────┐
│  Type    │  Range              │  ML Use Case                         │
├──────────┼─────────────────────┼──────────────────────────────────────┤
│  int8    │  -128 to 127        │  Quantized weights, activations      │
│  int16   │  -32,768 to 32,767  │  Audio samples, sensor data          │
│  int32   │  -2.1B to 2.1B      │  Token IDs, batch indices            │
│  int64   │  ±9.2×10¹⁸          │  Dataset sizes, file offsets         │
└──────────┴─────────────────────┴──────────────────────────────────────┘
```

#### ML Applications

| Application                | Example                | Why Integers?                   |
| -------------------------- | ---------------------- | ------------------------------- |
| **Negative Indexing**      | `arr[-1]`              | Python array access from end    |
| **Padding Values**         | `pad = -1`             | Sentinel for variable sequences |
| **Label Smoothing Offset** | `smooth = labels - 1`  | Shifting class indices          |
| **Quantized Weights**      | `int8_weight = 42`     | Model compression               |
| **Token IDs**              | `token_id = 15496`     | Vocabulary lookup               |
| **Position Encodings**     | `pos ∈ {-512,...,512}` | Relative positions              |

#### Integer Overflow - A Critical ML Bug

```
OVERFLOW VISUALIZATION
═══════════════════════════════════════════════════════════════════════

For 8-bit signed integers:

         127 + 1 = ???

    01111111  (127)
  +        1
  ──────────
    10000000  (-128!)  ← OVERFLOW! Wraps to minimum

    ┌────────────────────────────────────────────────────────────┐
    │                    CIRCULAR OVERFLOW                        │
    │                                                             │
    │                         127                                 │
    │                    ╭─────●─────╮                            │
    │               126 ●           ● -128                        │
    │                  /             \                            │
    │                 /               \                           │
    │           ...  ●                 ● -127                     │
    │                 \               /                           │
    │                  \             /                            │
    │                1  ●           ● -2                          │
    │                    ╰─────●─────╯                            │
    │                          0                                  │
    │                                                             │
    │     Add 1 → Move clockwise                                  │
    │     Subtract 1 → Move counter-clockwise                     │
    └────────────────────────────────────────────────────────────┘
```

#### Simple Code Example

```python
import numpy as np

# Integer overflow example (dangerous!)
int8_max = np.int8(127)
overflow = np.int8(int8_max + 1)  # Result: -128 (wrapped!)

# Safe integer operations in ML
indices = np.array([-1, 0, 1, 2])  # Negative indexing
token_ids = np.array([101, 2003, 102], dtype=np.int32)

# Quantization example
float_weight = 0.75
scale = 127.0
int8_weight = np.int8(round(float_weight * scale))  # 95
```

---

### 3. Rational Numbers (ℚ)

**Definition:** Numbers that can be expressed as a fraction p/q where p and q are integers and q ≠ 0.

$$\mathbb{Q} = \left\{\frac{p}{q} : p, q \in \mathbb{Z}, q \neq 0\right\}$$

#### Visual Representation

```
RATIONAL NUMBERS - FRACTIONS ON THE NUMBER LINE
═══════════════════════════════════════════════════════════════════════

    ◄────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────►
        -1  -3/4 -1/2 -1/4  0  1/4  1/2  3/4   1   5/4  3/2

        ─┬─   ─┬─   ─┬─   ─┬─   ─┬─   ─┬─   ─┬─
        │     │     │     │     │     │     │
       0.75  0.5   0.25   0   0.25  0.5   0.75

DENSITY PROPERTY:
┌─────────────────────────────────────────────────────────────────────┐
│  Between ANY two rational numbers, there are INFINITELY many       │
│  other rational numbers!                                           │
│                                                                     │
│    Between 0 and 1:                                                 │
│    0 ──── 1/2 ──── 1                                                │
│    0 ── 1/4 ── 1/2 ── 3/4 ── 1                                      │
│    0 ─ 1/8 ─ 1/4 ─ 3/8 ─ 1/2 ─ 5/8 ─ 3/4 ─ 7/8 ─ 1                  │
│    ... infinitely divisible ...                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Decimal Representations

```
TERMINATING vs REPEATING DECIMALS
═══════════════════════════════════════════════════════════════════════

TERMINATING (denominator has only factors of 2 and 5):
┌─────────────────────────────────────────────────────────────────────┐
│  1/2 = 0.5           ← denominator: 2                               │
│  1/4 = 0.25          ← denominator: 4 = 2²                          │
│  1/5 = 0.2           ← denominator: 5                               │
│  3/8 = 0.375         ← denominator: 8 = 2³                          │
│  7/20 = 0.35         ← denominator: 20 = 2² × 5                     │
└─────────────────────────────────────────────────────────────────────┘

REPEATING (denominator has other prime factors):
┌─────────────────────────────────────────────────────────────────────┐
│  1/3 = 0.333...       = 0.3̄                                         │
│  1/6 = 0.1666...      = 0.16̄                                        │
│  1/7 = 0.142857142857... = 0.1̄4̄2̄8̄5̄7̄                                 │
│  1/9 = 0.111...       = 0.1̄                                         │
│  1/11 = 0.090909...   = 0.0̄9̄                                        │
└─────────────────────────────────────────────────────────────────────┘

WHY 0.1 + 0.2 ≠ 0.3 IN COMPUTERS:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  1/10 = 0.1 (decimal) = 0.00011001100110011... (binary)             │
│                                  ↑                                  │
│                          REPEATING in binary!                       │
│                                                                     │
│  Computers store a FINITE approximation, causing tiny errors:       │
│                                                                     │
│  0.1 ≈ 0.10000000000000000555...                                    │
│  0.2 ≈ 0.20000000000000001110...                                    │
│  0.1 + 0.2 ≈ 0.30000000000000004441...                              │
│                                                                     │
│  0.3 ≈ 0.29999999999999998889...                                    │
│                                                                     │
│  Result: 0.1 + 0.2 ≠ 0.3 in floating point!                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### ML Applications

| Application         | Example           | Why Rationals?           |
| ------------------- | ----------------- | ------------------------ |
| **Learning Rate**   | `lr = 1e-3`       | Fine-tuned gradient step |
| **Dropout Rate**    | `dropout = 0.1`   | Fraction of neurons      |
| **Train/Val Split** | `0.8/0.2`         | Dataset partitioning     |
| **Weight Decay**    | `wd = 1e-4`       | Regularization strength  |
| **Momentum**        | `beta = 0.9`      | Gradient smoothing       |
| **Threshold**       | `threshold = 0.5` | Classification cutoff    |
| **Temperature**     | `temp = 0.7`      | Softmax scaling          |

#### Simple Code Example

```python
import numpy as np

# Learning rate schedules (rational number applications)
initial_lr = 0.001      # 1/1000
warmup_ratio = 0.1      # First 10% of training
decay_factor = 0.1      # Reduce LR by 10x

# The infamous floating point issue
result = 0.1 + 0.2
print(result == 0.3)           # False!
print(np.isclose(result, 0.3)) # True - use this instead!

# Percentage calculations
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)
```

---

### 4. Real Numbers (ℝ)

**Definition:** All numbers on the continuous number line, including both rational and irrational numbers.

$$\mathbb{R} = \mathbb{Q} \cup \text{Irrationals}$$

#### Visual Representation

```
THE REAL NUMBER LINE - CONTINUOUS AND COMPLETE
═══════════════════════════════════════════════════════════════════════

    ◄───────────────────────────────●───────────────────────────────►
   -∞                               0                               +∞

ZOOMING IN BETWEEN 0 AND 4:
    ────●────────●────────●────────●────────────●────────●──────────►
        0        1        2        3            π       4
                 │        │                     │
                 │        │                 3.14159...
                 │       √2 ≈ 1.414...          │
                 │        │                     │
    Rationals:   1     1.414    2    3      3.14159    4
    Irrationals:       √2               π

THE GAP THAT RATIONALS CAN'T FILL:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Consider: Find x where x² = 2                                      │
│                                                                     │
│  1² = 1     < 2      too small                                      │
│  2² = 4     > 2      too big                                        │
│  1.4² = 1.96   < 2   getting closer                                 │
│  1.5² = 2.25   > 2   overshot                                       │
│  1.41² = 1.9881 < 2  closer                                         │
│  1.42² = 2.0164 > 2  overshot                                       │
│  ...                                                                │
│                                                                     │
│  √2 = 1.41421356237... (never terminates, never repeats)            │
│                                                                     │
│  THIS NUMBER EXISTS on the number line but is NOT RATIONAL!         │
│  Irrationals "fill the gaps" between rationals.                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Famous Irrational Numbers in ML

```
IMPORTANT IRRATIONAL CONSTANTS
═══════════════════════════════════════════════════════════════════════

┌──────────┬───────────────────────┬──────────────────────────────────┐
│ Symbol   │ Value                 │ ML Application                   │
├──────────┼───────────────────────┼──────────────────────────────────┤
│ π (pi)   │ 3.14159265358979...   │ Circular distributions           │
│          │                       │ Angular position encoding        │
│          │                       │ Periodic activations             │
├──────────┼───────────────────────┼──────────────────────────────────┤
│ e        │ 2.71828182845904...   │ Exponential decay/growth         │
│          │                       │ Softmax function: e^x            │
│          │                       │ Natural logarithm base           │
│          │                       │ Learning rate schedules          │
├──────────┼───────────────────────┼──────────────────────────────────┤
│ √2       │ 1.41421356237309...   │ Normalization factors            │
│          │                       │ He initialization: √(2/n)        │
│          │                       │ Attention scaling: 1/√d_k        │
├──────────┼───────────────────────┼──────────────────────────────────┤
│ φ        │ 1.61803398874989...   │ Fibonacci learning rates         │
│ (phi)    │ Golden ratio          │ Network architecture ratios      │
└──────────┴───────────────────────┴──────────────────────────────────┘
```

#### The Completeness Property (Why Real Numbers Matter for Optimization)

```
COMPLETENESS - THE KEY TO GRADIENT DESCENT CONVERGENCE
═══════════════════════════════════════════════════════════════════════

The real numbers are "COMPLETE" - there are no gaps.

FORMALLY: Every Cauchy sequence of real numbers converges to a real.

INTUITIVELY:
┌─────────────────────────────────────────────────────────────────────┐
│  GRADIENT DESCENT converges because:                                │
│                                                                     │
│  Step 1: Start at some point θ₀                                     │
│  Step 2: Move: θ₁ = θ₀ - α∇L(θ₀)                                    │
│  Step 3: Move: θ₂ = θ₁ - α∇L(θ₁)                                    │
│  ...                                                                │
│                                                                     │
│  Loss:  ████████                                                    │
│         ███████                                                     │
│         ██████                                                      │
│         █████         Converging sequence                           │
│         ████            ↓                                           │
│         ███         θ* (optimal point EXISTS in ℝ)                  │
│         ██                                                          │
│         █                                                           │
│         ▀  ← Minimum exists because ℝ is complete                   │
│                                                                     │
│  Without completeness, the optimal θ* might not exist!              │
│  (Like if √2 didn't exist when we needed x² = 2)                    │
└─────────────────────────────────────────────────────────────────────┘

WHY THIS MATTERS:
─────────────────
• Guarantees optimization algorithms can find minima
• Ensures limits of training sequences exist
• Foundation for convergence proofs
```

#### Intervals and Their ML Uses

```
REAL NUMBER INTERVALS IN ML
═══════════════════════════════════════════════════════════════════════

CLOSED INTERVAL [a, b]:    a ≤ x ≤ b    (includes endpoints)
┌─────────────────────────────────────────────────────────────────────┐
│  [0, 1]   Used for:  Probabilities, sigmoid output, normalized data│
│  [-1, 1]  Used for:  tanh output, normalized features              │
│  [0, 255] Used for:  Pixel values (integer, but continuous approx)  │
└─────────────────────────────────────────────────────────────────────┘

OPEN INTERVAL (a, b):      a < x < b    (excludes endpoints)
┌─────────────────────────────────────────────────────────────────────┐
│  (0, 1)   Used for:  Log inputs (need > 0), dropout keep prob      │
│  (0, ∞)   Used for:  ReLU output, variance, learning rate          │
└─────────────────────────────────────────────────────────────────────┘

HALF-OPEN [a, b) or (a, b]:
┌─────────────────────────────────────────────────────────────────────┐
│  [0, 2π)  Used for:  Angular representations (no wrap-around dup)  │
└─────────────────────────────────────────────────────────────────────┘
```

#### ML Applications

| Application          | Example           | Why Real Numbers?       |
| -------------------- | ----------------- | ----------------------- |
| **Weights & Biases** | `W ∈ ℝ^(n×m)`     | Continuous optimization |
| **Loss Values**      | `loss = 0.0342`   | Measures model fit      |
| **Probabilities**    | `p ∈ [0,1]`       | Softmax outputs         |
| **Activations**      | `σ(x), tanh(x)`   | Non-linear transforms   |
| **Embeddings**       | `embed ∈ ℝ^768`   | Dense representations   |
| **Attention Scores** | `score ∈ ℝ`       | Similarity measure      |
| **Gradients**        | `∂L/∂W ∈ ℝ^(n×m)` | Direction of descent    |

#### Simple Code Example

```python
import numpy as np

# Famous irrational numbers in ML
pi = np.pi              # 3.14159..., used in positional encoding
e = np.e                # 2.71828..., base of natural log
sqrt2 = np.sqrt(2)      # 1.41421..., used in initializations

# He initialization uses √(2/n)
n_inputs = 512
he_std = np.sqrt(2.0 / n_inputs)  # ≈ 0.0625

# Attention scaling uses 1/√d_k
d_k = 64
scale = 1.0 / np.sqrt(d_k)  # ≈ 0.125

# Continuous outputs
sigmoid = lambda x: 1 / (1 + np.exp(-x))
prob = sigmoid(2.5)  # 0.924... (a real number in [0,1])
```

---

### 5. Complex Numbers (ℂ)

**Definition:** Numbers of the form a + bi where a, b are real and i² = -1.

$$\mathbb{C} = \{a + bi : a, b \in \mathbb{R}, i^2 = -1\}$$

#### Visual Representation - The Complex Plane

```
THE COMPLEX PLANE (ARGAND DIAGRAM)
═══════════════════════════════════════════════════════════════════════

                        Imaginary Axis (Im)
                              │
                         4i   │
                              │        ● (3+4i)
                         3i   │       /│
                              │      / │
                         2i   │ r=5 /  │ b=4
                              │    /   │
                          i   │   /    │
                              │  /θ    │
    ──────────────────────────●────────●────────────────► Real Axis (Re)
         -4   -3   -2   -1    0    1   2   3   4   5
                              │        a=3
                         -i   │
                              │
                        -2i   │    ● (2-2i)
                              │
                        -3i   │
                              │

COMPONENTS OF z = a + bi:
┌─────────────────────────────────────────────────────────────────────┐
│  • a = Real part = Re(z)                                            │
│  • b = Imaginary part = Im(z)                                       │
│  • |z| = √(a² + b²) = Magnitude (distance from origin)              │
│  • θ = atan2(b, a) = Argument/Phase (angle from positive real axis) │
└─────────────────────────────────────────────────────────────────────┘

EXAMPLE: z = 3 + 4i
    Re(z) = 3
    Im(z) = 4
    |z| = √(3² + 4²) = √25 = 5
    θ = atan2(4, 3) ≈ 53.13°
```

#### Polar Form and Euler's Formula

```
RECTANGULAR vs POLAR REPRESENTATION
═══════════════════════════════════════════════════════════════════════

RECTANGULAR:  z = a + bi           (Cartesian coordinates)
POLAR:        z = r(cos θ + i sin θ)  = r·e^(iθ)

EULER'S FORMULA (The Most Beautiful Equation):
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                    e^(iθ) = cos(θ) + i·sin(θ)                       │
│                                                                     │
│  Special case when θ = π:                                           │
│                                                                     │
│                    e^(iπ) + 1 = 0                                   │
│                    ─────────────                                    │
│                    EULER'S IDENTITY                                 │
│                                                                     │
│  Connects: e (calculus), i (complex), π (geometry), 1, 0           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

THE UNIT CIRCLE - WHERE |z| = 1:

                      e^(iπ/2) = i
                          │
                          ●
                         /│\
                        / │ \
            e^(iπ)=-1 ●───┼───● e^(i·0) = 1
                        \ │ /
                         \│/
                          ●
                          │
                    e^(i3π/2) = -i

    Any point on unit circle: e^(iθ) = cos(θ) + i·sin(θ)
```

#### Complex Arithmetic Visualized

```
COMPLEX OPERATIONS
═══════════════════════════════════════════════════════════════════════

ADDITION: Vector addition (tip-to-tail)
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│    (2+3i) + (1+i) = (3+4i)                                          │
│                                                                     │
│         Im │                                                        │
│        4   │           ● (3+4i)                                     │
│        3   ●──────────/                                             │
│            │\(2+3i)  /                                              │
│        2   │ \      /                                               │
│        1   │  \    /● (1+i)                                         │
│            │   \  /                                                 │
│         ───●────●────────── Re                                      │
│            0    1   2   3                                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

MULTIPLICATION: Multiply magnitudes, add angles
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│    z₁ × z₂ = (r₁·r₂) · e^(i(θ₁+θ₂))                                 │
│                                                                     │
│    Example: (1+i) × (1+i)                                           │
│    |1+i| = √2,  θ = 45°                                             │
│                                                                     │
│    Result: |z|² = 2,  θ = 90°  →  2i                                │
│                                                                     │
│    Check: (1+i)² = 1 + 2i + i² = 1 + 2i - 1 = 2i  ✓                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

CONJUGATE: Reflect across real axis
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│    z = a + bi  →  z̄ = a - bi                                        │
│                                                                     │
│         Im │                                                        │
│        2   │   ● z = 3+2i                                           │
│        1   │                                                        │
│         ───●───────●───── Re                                        │
│       -1   │       3                                                │
│       -2   │   ● z̄ = 3-2i                                           │
│                                                                     │
│    Property: z × z̄ = |z|² = a² + b²  (always real!)                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Fourier Transform - Why Complex Numbers Matter in ML

```
FOURIER TRANSFORM - THE ML POWER TOOL
═══════════════════════════════════════════════════════════════════════

The Discrete Fourier Transform (DFT) decomposes signals into frequencies:

                    N-1
    X[k] = Σ x[n] · e^(-2πi·kn/N)
                   n=0

WHAT IT DOES:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  TIME DOMAIN ─────────── DFT ──────────→ FREQUENCY DOMAIN           │
│                                                                     │
│  Signal (what we measure)              Spectrum (frequencies)       │
│                                                                     │
│    ╭─╮   ╭─╮   ╭─╮                         ┃                        │
│   ╱   ╲ ╱   ╲ ╱   ╲          →          ▓▓▓┃▓▓                      │
│  ╱     V     V     ╲                    ▓▓▓┃▓▓▓▓▓                   │
│  ───────────────────                    ▓▓▓┃▓▓▓▓▓▓▓▓                │
│                                         0  f1 f2 f3...              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

ML APPLICATIONS:
┌─────────────────────────────────────────────────────────────────────┐
│  • Audio Processing: Speech recognition (spectrograms)              │
│  • Image Processing: Frequency filtering, compression               │
│  • NLP: Periodic patterns in sequences                              │
│  • Time Series: Seasonal decomposition                              │
│  • Convolutions: FFT speeds up convolutions to O(n log n)           │
│  • Attention: Some efficient transformers use FFT                   │
└─────────────────────────────────────────────────────────────────────┘

WHY COMPLEX NUMBERS:
    e^(-2πi·kn/N) = cos(2πkn/N) - i·sin(2πkn/N)

    Phase information (imaginary part) tells us WHEN peaks occur
    Magnitude (|X[k]|) tells us HOW STRONG each frequency is
```

#### ML Applications

| Application             | Example          | Why Complex Numbers?        |
| ----------------------- | ---------------- | --------------------------- |
| **Spectrograms**        | Audio ML         | Represent phase & magnitude |
| **FFT Convolution**     | Fast Conv2D      | O(n log n) vs O(n²)         |
| **Eigenvalues**         | PCA, stability   | Non-symmetric matrices      |
| **Positional Encoding** | Transformers     | Rotational representation   |
| **Quantum ML**          | Quantum circuits | State amplitudes            |
| **Signal Filters**      | Audio/Image      | Frequency domain ops        |

#### Simple Code Example

```python
import numpy as np

# Complex number basics
z = 3 + 4j          # Python uses j, not i
print(z.real)       # 3.0
print(z.imag)       # 4.0
print(abs(z))       # 5.0 (magnitude)
print(np.angle(z))  # 0.927... radians (phase)

# Euler's formula
theta = np.pi / 4   # 45 degrees
euler = np.exp(1j * theta)
print(euler)        # (0.707... + 0.707...j)
print(np.cos(theta) + 1j * np.sin(theta))  # Same!

# FFT for audio/signal processing
signal = np.sin(np.linspace(0, 4*np.pi, 100))  # Sine wave
spectrum = np.fft.fft(signal)  # Complex spectrum
magnitudes = np.abs(spectrum)  # Frequency strengths
phases = np.angle(spectrum)    # Timing information
```

---

### Extension: Quaternions (ℍ)

**Definition:** An extension of complex numbers used for 3D rotations.
$$q = a + bi + cj + dk$$
where $i^2 = j^2 = k^2 = ijk = -1$.

**ML Application:**

- **3D Rotations:** Essential for robotics, computer vision (camera pose), and molecular modeling.
- **Quaternion Neural Networks:** Process 3D data more efficiently than standard numeric inputs.
- Avoids "Gimbal Lock" issues present in Euler angles.

---

## Number Bases

Understanding different number bases is essential for working with computer systems, memory, and ML model optimization.

### Base Comparison

```
NUMBER BASES - SAME VALUE, DIFFERENT REPRESENTATION
═══════════════════════════════════════════════════════════════════════

THE NUMBER "42" IN DIFFERENT BASES:
┌─────────┬───────────┬─────────────┬─────────────────────────────────┐
│ Base    │ Name      │ Value       │ Calculation                     │
├─────────┼───────────┼─────────────┼─────────────────────────────────┤
│ 2       │ Binary    │ 101010      │ 32+8+2 = 42                     │
│ 8       │ Octal     │ 52          │ 5×8 + 2 = 42                    │
│ 10      │ Decimal   │ 42          │ 4×10 + 2 = 42                   │
│ 16      │ Hex       │ 2A          │ 2×16 + 10 = 42                  │
└─────────┴───────────┴─────────────┴─────────────────────────────────┘

WHY DIFFERENT BASES:
┌─────────────────────────────────────────────────────────────────────┐
│  Binary (2)  → Hardware: Transistors are ON or OFF                  │
│  Octal (8)   → Legacy: Early PDP computers (rarely used now)        │
│  Decimal (10)→ Human: We have 10 fingers                            │
│  Hex (16)    → Compact: 1 hex digit = 4 binary bits                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Binary - The Foundation of Computing

```
BINARY (BASE 2) - THE LANGUAGE OF COMPUTERS
═══════════════════════════════════════════════════════════════════════

PLACE VALUES (powers of 2):
┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
│  128   │   64   │   32   │   16   │    8   │    4   │    2   │    1   │
│  2⁷    │   2⁶   │   2⁵   │   2⁴   │   2³   │   2²   │   2¹   │   2⁰   │
├────────┼────────┼────────┼────────┼────────┼────────┼────────┼────────┤
│   0    │    1   │    0   │    1   │    0   │    1   │    0   │    0   │
└────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘
                    ↓
           64 + 16 + 4 = 84 (decimal)

POWERS OF 2 YOU SHOULD MEMORIZE:
┌──────┬───────┬───────────────────────────────────────────────────────┐
│  2ⁿ  │ Value │ ML Significance                                       │
├──────┼───────┼───────────────────────────────────────────────────────┤
│  2⁰  │     1 │ Binary digit                                          │
│  2⁴  │    16 │ Half-precision float (float16) exponent bits          │
│  2⁶  │    64 │ Common batch size, attention head dimension           │
│  2⁷  │   128 │ int8 range (-128 to 127), common embedding dim        │
│  2⁸  │   256 │ uint8 range, pixel values                             │
│  2⁹  │   512 │ BERT-base hidden size, common sequence length         │
│  2¹⁰ │  1024 │ 1 KB, GPT-2 medium hidden size                        │
│  2¹² │  4096 │ Common FFN dimension, batch size                      │
│  2¹⁴ │ 16384 │ 16 KB L1 cache                                        │
│  2²⁰ │  ~1M  │ 1 MB                                                  │
│  2³⁰ │  ~1B  │ 1 GB (gigabyte)                                       │
└──────┴───────┴───────────────────────────────────────────────────────┘
```

### Hexadecimal - Compact Binary Representation

```
HEXADECIMAL (BASE 16) - HUMAN-READABLE BINARY
═══════════════════════════════════════════════════════════════════════

HEX DIGITS: 0 1 2 3 4 5 6 7 8 9 A  B  C  D  E  F
DECIMAL:    0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

HEX TO BINARY CONVERSION (each hex digit = 4 bits):
┌──────┬────────┬──────┬────────┬──────┬────────┬──────┬────────┐
│  0   │ 0000   │  4   │ 0100   │  8   │ 1000   │  C   │ 1100   │
│  1   │ 0001   │  5   │ 0101   │  9   │ 1001   │  D   │ 1101   │
│  2   │ 0010   │  6   │ 0110   │  A   │ 1010   │  E   │ 1110   │
│  3   │ 0011   │  7   │ 0111   │  B   │ 1011   │  F   │ 1111   │
└──────┴────────┴──────┴────────┴──────┴────────┴──────┴────────┘

EXAMPLE: 0xDEADBEEF (a famous memory debug value)
    D    E    A    D    B    E    E    F
   1101 1110 1010 1101 1011 1110 1110 1111
   = 3,735,928,559 (decimal)

ML APPLICATIONS:
┌─────────────────────────────────────────────────────────────────────┐
│  Memory addresses:  0x7fff5fbff8a0 (stack pointer)                  │
│  Color values:      #FF5733 (RGB: 255, 87, 51)                      │
│  Bit patterns:      0xFFFF = all ones for masking                   │
│  Model checksums:   SHA256 hashes of weights                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Base Conversion Methods

```
CONVERSION BETWEEN BASES
═══════════════════════════════════════════════════════════════════════

DECIMAL TO BINARY (repeated division):
┌─────────────────────────────────────────────────────────────────────┐
│  Convert 42 to binary:                                              │
│                                                                     │
│  42 ÷ 2 = 21 remainder 0  ↑                                         │
│  21 ÷ 2 = 10 remainder 1  │                                         │
│  10 ÷ 2 =  5 remainder 0  │  Read remainders upward                 │
│   5 ÷ 2 =  2 remainder 1  │                                         │
│   2 ÷ 2 =  1 remainder 0  │                                         │
│   1 ÷ 2 =  0 remainder 1  │                                         │
│                                                                     │
│  Result: 42 = 101010 (binary)                                       │
└─────────────────────────────────────────────────────────────────────┘

BINARY TO DECIMAL (sum of powers):
┌─────────────────────────────────────────────────────────────────────┐
│  Convert 101010 to decimal:                                         │
│                                                                     │
│  1×2⁵ + 0×2⁴ + 1×2³ + 0×2² + 1×2¹ + 0×2⁰                            │
│  = 32  +  0   +  8   +  0   +  2   +  0                             │
│  = 42                                                               │
└─────────────────────────────────────────────────────────────────────┘

HEX TO BINARY (substitute each digit):
┌─────────────────────────────────────────────────────────────────────┐
│  Convert 0x2A to binary:                                            │
│                                                                     │
│  2    →  0010                                                       │
│  A    →  1010                                                       │
│                                                                     │
│  0x2A = 00101010 (binary) = 42 (decimal)                            │
└─────────────────────────────────────────────────────────────────────┘
```

#### Simple Code Example

```python
# Number base conversions in Python
decimal = 42

# Convert to different bases
binary = bin(42)    # '0b101010'
octal = oct(42)     # '0o52'
hexval = hex(42)    # '0x2a'

# Parse from different bases
from_binary = int('101010', 2)    # 42
from_hex = int('2A', 16)          # 42
from_octal = int('52', 8)         # 42

# Bit manipulation (common in ML optimizations)
x = 0b11110000    # 240
y = x >> 4        # Right shift: 0b1111 = 15
z = x & 0b00001111  # Mask lower 4 bits: 0
```

---

## Computer Number Representation

### Floating Point Numbers (IEEE 754)

```
IEEE 754 FLOATING POINT FORMAT
═══════════════════════════════════════════════════════════════════════

COMPONENTS:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Value = (-1)^Sign × 2^(Exponent - Bias) × (1 + Mantissa)           │
│                                                                     │
│  ┌──────┐ ┌──────────────┐ ┌─────────────────────────────────────┐  │
│  │ Sign │ │   Exponent   │ │           Mantissa (Fraction)       │  │
│  │ (±)  │ │  (scale)     │ │        (precision digits)           │  │
│  └──────┘ └──────────────┘ └─────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

32-BIT FLOAT (single precision):
┌───┬──────────────────┬────────────────────────────────────────────┐
│ S │     Exponent     │              Mantissa                      │
│ 1 │      8 bits      │              23 bits                       │
└───┴──────────────────┴────────────────────────────────────────────┘
    Bias = 127

64-BIT FLOAT (double precision):
┌───┬──────────────────┬──────────────────────────────────────────────────────────────┐
│ S │     Exponent     │                        Mantissa                              │
│ 1 │     11 bits      │                        52 bits                               │
└───┴──────────────────┴──────────────────────────────────────────────────────────────┘
    Bias = 1023

16-BIT FORMATS (used in ML):
┌───┬────────────┬───────────────────┐  ┌───┬──────────┬─────────────────────────┐
│ S │  Exponent  │     Mantissa      │  │ S │ Exponent │        Mantissa         │
│ 1 │   5 bits   │     10 bits       │  │ 1 │  8 bits  │         7 bits          │
└───┴────────────┴───────────────────┘  └───┴──────────┴─────────────────────────┘
        float16 (IEEE)                           bfloat16 (Google Brain)
```

### Float Type Comparison for ML

```
FLOATING POINT TYPES IN MACHINE LEARNING
═══════════════════════════════════════════════════════════════════════

┌──────────┬───────┬─────┬────────┬────────────────┬─────────────────────┐
│ Type     │ Bits  │Exp  │Mantissa│ Range          │ ML Use Case         │
├──────────┼───────┼─────┼────────┼────────────────┼─────────────────────┤
│ float16  │  16   │  5  │   10   │ ±6.5×10⁴       │ Inference, Training │
│ bfloat16 │  16   │  8  │    7   │ ±3.4×10³⁸      │ TPU training        │
│ float32  │  32   │  8  │   23   │ ±3.4×10³⁸      │ Standard training   │
│ float64  │  64   │ 11  │   52   │ ±1.8×10³⁰⁸     │ Scientific/research │
└──────────┴───────┴─────┴────────┴────────────────┴─────────────────────┘

PRECISION VS RANGE TRADEOFF:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  float16:    ████░░░░░░░░ Range    ████████████ Precision          │
│  bfloat16:   ████████████ Range    ████░░░░░░░░ Precision          │
│  float32:    ████████████ Range    ████████████ Precision          │
│                                                                     │
│  bfloat16 = Same range as float32, but fewer precision bits         │
│           → Better for neural nets (gradients don't need precision) │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Machine Epsilon and Precision

```
MACHINE EPSILON - THE PRECISION LIMIT
═══════════════════════════════════════════════════════════════════════

Definition: Smallest ε where 1.0 + ε ≠ 1.0

┌──────────┬─────────────────┬───────────────────────────────────────┐
│ Type     │ Machine Epsilon │ Decimal Precision                     │
├──────────┼─────────────────┼───────────────────────────────────────┤
│ float16  │    ~9.77×10⁻⁴   │ ~3-4 significant digits               │
│ bfloat16 │    ~7.81×10⁻³   │ ~2-3 significant digits               │
│ float32  │    ~1.19×10⁻⁷   │ ~7 significant digits                 │
│ float64  │    ~2.22×10⁻¹⁶  │ ~16 significant digits                │
└──────────┴─────────────────┴───────────────────────────────────────┘

WHY THIS MATTERS FOR ML:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  1. Gradient precision:                                             │
│     Small gradients (< epsilon) effectively become ZERO             │
│     → Training stops for those weights                              │
│                                                                     │
│  2. Loss accumulation:                                              │
│     Adding tiny values to large sums may have no effect             │
│     → Batch losses may not update correctly                         │
│                                                                     │
│  3. Numerical stability:                                            │
│     Division by small numbers amplifies errors                      │
│     → Always add epsilon to denominators                            │
│                                                                     │
│  Example: BatchNorm                                                 │
│     x_norm = (x - mean) / sqrt(variance + epsilon)                  │
│                                      ↑                              │
│                               Prevents div by 0                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quantization for ML

Quantization reduces model size and speeds up inference by using lower-precision numbers.

### Quantization Basics

```
QUANTIZATION - REDUCING PRECISION FOR EFFICIENCY
═══════════════════════════════════════════════════════════════════════

CONCEPT:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  float32 weight: 0.374628901...  (32 bits, ~7 decimal places)       │
│                         ↓ Quantize                                  │
│  int8 weight:          48        (8 bits, scaled integer)           │
│                         ↓ Dequantize                                │
│  Reconstructed:  0.376...        (approximate)                      │
│                                                                     │
│  Memory: 32 bits → 8 bits = 4× compression!                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

QUANTIZATION FORMULA:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Quantize:     q = round((x - zero_point) / scale)                  │
│  Dequantize:   x' = q × scale + zero_point                          │
│                                                                     │
│  Where:                                                             │
│    scale = (max_val - min_val) / (2^bits - 1)                       │
│    zero_point = offset to handle negative values                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Quantization Types

```
QUANTIZATION APPROACHES
═══════════════════════════════════════════════════════════════════════

BY PRECISION:
┌──────────┬────────┬──────────────┬──────────────────────────────────┐
│ Format   │ Bits   │ Compression  │ Use Case                         │
├──────────┼────────┼──────────────┼──────────────────────────────────┤
│ FP32     │   32   │     1×       │ Training (baseline)              │
│ FP16     │   16   │     2×       │ Mixed precision training         │
│ BF16     │   16   │     2×       │ TPU training                     │
│ INT8     │    8   │     4×       │ Inference deployment             │
│ INT4     │    4   │     8×       │ Edge devices, LLM inference      │
│ INT2     │    2   │    16×       │ Extreme compression (research)   │
│ Binary   │    1   │    32×       │ BinaryConnect, XNOR-Net          │
└──────────┴────────┴──────────────┴──────────────────────────────────┘

BY TIMING:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Post-Training Quantization (PTQ):                                  │
│    • Apply after training is complete                               │
│    • Fast, no retraining needed                                     │
│    • May have accuracy loss                                         │
│                                                                     │
│  Quantization-Aware Training (QAT):                                 │
│    • Simulate quantization during training                          │
│    • Model learns to be robust to quantization                      │
│    • Better accuracy, requires full training                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

BY SCHEME:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Symmetric:   -max ←───── 0 ─────→ +max                             │
│               Zero point = 0                                        │
│               Simpler computation                                   │
│                                                                     │
│  Asymmetric:  min ←───── ? ─────→ max                               │
│               Zero point ≠ 0                                        │
│               Better for ReLU (all positive)                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### LLM Quantization (GGUF, GPTQ, AWQ)

```
LARGE LANGUAGE MODEL QUANTIZATION
═══════════════════════════════════════════════════════════════════════

MODEL SIZE COMPARISON (LLaMA 7B example):
┌──────────┬────────────┬────────────────────────────────────────────┐
│ Format   │ Size       │ Notes                                      │
├──────────┼────────────┼────────────────────────────────────────────┤
│ FP32     │ ~28 GB     │ Original (rarely used)                     │
│ FP16     │ ~14 GB     │ Standard distribution                      │
│ INT8     │ ~7 GB      │ Good quality, fits more GPUs               │
│ INT4     │ ~3.5 GB    │ Popular for consumer GPUs                  │
│ INT3     │ ~2.6 GB    │ Aggressive, some quality loss              │
│ INT2     │ ~1.75 GB   │ Research, significant quality loss         │
└──────────┴────────────┴────────────────────────────────────────────┘

POPULAR METHODS:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  GGUF (llama.cpp):                                                  │
│    • CPU-friendly quantization                                      │
│    • Various bit-widths (Q4_0, Q4_K_M, Q5_K_M, Q8_0)                │
│    • Works without GPU                                              │
│                                                                     │
│  GPTQ:                                                              │
│    • GPU-optimized 4-bit quantization                               │
│    • Uses second-order information                                  │
│    • Good accuracy preservation                                     │
│                                                                     │
│  AWQ (Activation-aware Weight Quantization):                        │
│    • Protects important weights based on activations                │
│    • State-of-the-art quality at 4-bit                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Simple Code Example

```python
import numpy as np

# Simple quantization example
def quantize_int8(weights, min_val=None, max_val=None):
    if min_val is None:
        min_val = weights.min()
    if max_val is None:
        max_val = weights.max()

    scale = (max_val - min_val) / 255
    zero_point = -min_val / scale

    quantized = np.round(weights / scale + zero_point).astype(np.uint8)
    return quantized, scale, zero_point

def dequantize(quantized, scale, zero_point):
    return (quantized.astype(np.float32) - zero_point) * scale

# Example usage
original = np.array([0.1, -0.5, 0.8, -0.2, 0.0])
quantized, scale, zp = quantize_int8(original)
reconstructed = dequantize(quantized, scale, zp)
error = np.abs(original - reconstructed).mean()
print(f"Mean quantization error: {error:.6f}")
```

---

## Numerical Stability in Deep Learning

Numerical issues are among the most common bugs in deep learning. Understanding these helps debug training failures.

### Overflow and Underflow

```
OVERFLOW AND UNDERFLOW - WHEN NUMBERS BREAK
═══════════════════════════════════════════════════════════════════════

OVERFLOW (Too Large):
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  exp(1000) = ???                                                    │
│                                                                     │
│  float32 max ≈ 3.4×10³⁸                                             │
│  e^1000 ≈ 10^434  →  FAR exceeds max  →  inf                        │
│                                                                     │
│  COMMON CAUSES IN ML:                                               │
│  • Large logits in softmax                                          │
│  • Exploding gradients                                              │
│  • Large learning rates                                             │
│  • Missing normalization                                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

UNDERFLOW (Too Small):
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  0.0001 × 0.0001 × ... × 0.0001 (100 times) = ???                   │
│                                                                     │
│  = 10^(-400)  →  Below float32 min  →  0.0                          │
│                                                                     │
│  COMMON CAUSES IN ML:                                               │
│  • Products of probabilities                                        │
│  • Deep networks with small activations                             │
│  • Vanishing gradients                                              │
│  • Very small learning rates                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### The Log-Sum-Exp Trick

```
LOG-SUM-EXP TRICK - STABLE SOFTMAX
═══════════════════════════════════════════════════════════════════════

PROBLEM: Softmax on large numbers
    softmax(x)_i = exp(x_i) / Σ exp(x_j)

    If x = [1000, 1001, 1002]:
        exp(1000) = inf
        exp(1001) = inf
        exp(1002) = inf
        Result: inf/inf = NaN  ✗

SOLUTION: Subtract the maximum
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  softmax(x)_i = exp(x_i - max(x)) / Σ exp(x_j - max(x))             │
│                                                                     │
│  If x = [1000, 1001, 1002]:                                         │
│      max(x) = 1002                                                  │
│      x - max(x) = [-2, -1, 0]                                       │
│                                                                     │
│      exp(-2) = 0.135                                                │
│      exp(-1) = 0.368                                                │
│      exp(0)  = 1.000                                                │
│                                                                     │
│      softmax = [0.090, 0.245, 0.665]  ✓                             │
│                                                                     │
│  MATHEMATICALLY EQUIVALENT but numerically stable!                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

FOR LOG-PROBABILITIES:
    log(Σ exp(x)) = max(x) + log(Σ exp(x - max(x)))
```

### Gradient Clipping

```
GRADIENT CLIPPING - PREVENTING EXPLOSIONS
═══════════════════════════════════════════════════════════════════════

EXPLODING GRADIENTS:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Loss         Gradient                                              │
│   │             ↑                                                   │
│   │      █████████████████ EXPLOSION!                               │
│   │      ████                                                       │
│   │     ███                                                         │
│   │    ██                                                           │
│   │   █             Normal gradient                                 │
│   │  █              ↓                                               │
│   └──────────────── Step                                            │
│                                                                     │
│  Solution: Clip gradients to max norm                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

CLIPPING METHODS:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  Value clipping:   g = clip(g, -max_val, max_val)                   │
│                    Clips each gradient independently                │
│                                                                     │
│  Norm clipping:    if ||g|| > max_norm:                             │
│                        g = g × (max_norm / ||g||)                   │
│                    Preserves gradient direction                     │
│                                                                     │
│  COMMON VALUES:                                                     │
│    max_norm = 1.0  (conservative)                                   │
│    max_norm = 5.0  (moderate)                                       │
│    max_norm = 10.0 (aggressive)                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Special Floating Point Values

### Infinity, NaN, and Denormals

```
SPECIAL FLOATING POINT VALUES
═══════════════════════════════════════════════════════════════════════

INFINITY (inf):
┌─────────────────────────────────────────────────────────────────────┐
│  Created by:                                                        │
│    • Overflow: exp(1000)                                            │
│    • Division: 1.0 / 0.0                                            │
│                                                                     │
│  Properties:                                                        │
│    inf + 1 = inf                                                    │
│    inf × 2 = inf                                                    │
│    inf - inf = nan                                                  │
│    1 / inf = 0                                                      │
│    inf > any_finite_number                                          │
│                                                                     │
│  ML impact: Training divergence, loss explosion                     │
└─────────────────────────────────────────────────────────────────────┘

NaN (Not a Number):
┌─────────────────────────────────────────────────────────────────────┐
│  Created by:                                                        │
│    • 0 / 0                                                          │
│    • inf - inf                                                      │
│    • inf / inf                                                      │
│    • sqrt(-1) [without complex]                                     │
│                                                                     │
│  Properties (NaN is TOXIC):                                         │
│    nan + 1 = nan                                                    │
│    nan × 0 = nan                                                    │
│    nan == nan → False!  (NaN ≠ anything, including itself)          │
│    nan > 0 → False                                                  │
│    nan < 0 → False                                                  │
│                                                                     │
│  Detection: Use isnan(), not == comparison!                         │
│  ML impact: Complete training failure, all weights become NaN       │
└─────────────────────────────────────────────────────────────────────┘

DENORMALIZED NUMBERS (Subnormals):
┌─────────────────────────────────────────────────────────────────────┐
│  Numbers smaller than the minimum normal float                      │
│                                                                     │
│  float32 min normal: ~1.2×10⁻³⁸                                     │
│  float32 min denormal: ~1.4×10⁻⁴⁵                                   │
│                                                                     │
│  Problem: Denormals are SLOW (10-100× slower on some hardware)      │
│  Solution: Flush to zero (FTZ) mode in GPU training                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Mixed Precision Training

Mixed precision uses different precisions for different operations to maximize speed while maintaining accuracy.

### How Mixed Precision Works

```
MIXED PRECISION TRAINING FLOW
═══════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   MASTER WEIGHTS (float32) ──────┐                                  │
│         ↓                        │                                  │
│   Copy to float16 ───────────────│───────────────────┐              │
│         ↓                        │                   │              │
│   ┌───────────────┐              │                   │              │
│   │ FORWARD PASS  │ float16     │                   │              │
│   │ (fast!)       │              │                   │              │
│   └───────┬───────┘              │                   │              │
│           ↓                      │                   │              │
│   ┌───────────────┐              │                   │              │
│   │ COMPUTE LOSS  │ float32     │                   │              │
│   └───────┬───────┘              │                   │              │
│           ↓                      │                   │              │
│   ┌───────────────┐              │                   │              │
│   │ SCALE LOSS    │ ×1024       │  LOSS SCALING     │              │
│   └───────┬───────┘              │  (prevents       │              │
│           ↓                      │   underflow)     │              │
│   ┌───────────────┐              │                   │              │
│   │ BACKWARD PASS │ float16     │                   │              │
│   │ (fast!)       │              │                   │              │
│   └───────┬───────┘              │                   │              │
│           ↓                      │                   │              │
│   ┌───────────────┐              │                   │              │
│   │ UNSCALE GRADS │ ÷1024       │                   │              │
│   └───────┬───────┘              │                   │              │
│           ↓                      │                   │              │
│   ┌───────────────┐              │                   │              │
│   │ UPDATE        │ float32 ←───┴───────────────────┘              │
│   │ MASTER WEIGHTS│                                                 │
│   └───────────────┘                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Loss Scaling

```
LOSS SCALING - PREVENTING GRADIENT UNDERFLOW
═══════════════════════════════════════════════════════════════════════

PROBLEM:
┌─────────────────────────────────────────────────────────────────────┐
│  Small gradients in float16 become zero (underflow)                 │
│                                                                     │
│  Example gradient: 0.00001  →  Smaller than float16 min             │
│                              →  Becomes 0.0                         │
│                              →  Weight never updates!               │
└─────────────────────────────────────────────────────────────────────┘

SOLUTION: Scale loss before backward, unscale after
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  1. Scale: loss_scaled = loss × scale_factor (e.g., 1024)           │
│  2. Backward: gradients are also scaled                             │
│  3. Unscale: gradient = gradient / scale_factor                     │
│  4. Update: optimizer.step() with unscaled gradients                │
│                                                                     │
│  Gradient: 0.00001 × 1024 = 0.01024  →  Survives in float16!        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

DYNAMIC LOSS SCALING:
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│  • Start with large scale (e.g., 2^16)                              │
│  • If gradients overflow: reduce scale by 2                         │
│  • If gradients OK for N steps: increase scale by 2                 │
│  • Automatically finds optimal scale                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Simple Code Example

```python
import torch

# PyTorch automatic mixed precision
scaler = torch.cuda.amp.GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass in float16
    with torch.cuda.amp.autocast():
        outputs = model(batch['input'])
        loss = criterion(outputs, batch['target'])

    # Backward pass with scaled loss
    scaler.scale(loss).backward()

    # Unscale and update
    scaler.step(optimizer)
    scaler.update()
```

---

## Common Pitfalls

### Floating Point Comparison

```python
# WRONG - Direct comparison fails!
0.1 + 0.2 == 0.3  # False!

# CORRECT - Use tolerance
import numpy as np
np.isclose(0.1 + 0.2, 0.3)  # True
np.allclose(array1, array2)  # For arrays

# Or use math.isclose
import math
math.isclose(0.1 + 0.2, 0.3)  # True
```

### Integer Division

```python
# Python 3 behavior
7 / 2   # 3.5 (float division)
7 // 2  # 3   (integer/floor division)

# Common bug
batch_size = 32
n_samples = 100
n_batches = n_samples / batch_size   # 3.125 (float!)
n_batches = n_samples // batch_size  # 3 (int, correct)
```

### Accumulation Errors

```python
# Adding many small numbers to a large number
total = 1e10
for i in range(1000000):
    total += 1e-5  # This addition might have no effect!

# Solution: Kahan summation or use higher precision
total = np.float64(1e10)  # Use float64 for accumulation
```

### NaN Propagation

```python
import numpy as np

# NaN is toxic - spreads through computation
arr = np.array([1.0, np.nan, 3.0])
print(arr.mean())  # nan
print(arr.sum())   # nan

# Solution: Use nan-safe functions
print(np.nanmean(arr))  # 2.0
print(np.nansum(arr))   # 4.0

# Check for NaN in training
if torch.isnan(loss):
    print("Training diverged!")
```

---

## Interview Questions

### Basic Questions

1. **Q: Why do we add epsilon in division operations?**

   A: To prevent division by zero and improve numerical stability. Example: `x / (y + epsilon)` ensures we never divide by exactly zero, even if y becomes very small during training.

2. **Q: What's the difference between float16 and bfloat16?**

   A: float16 has more precision (10 mantissa bits) but smaller range. bfloat16 has less precision (7 mantissa bits) but same range as float32. bfloat16 is preferred for training because gradients need range more than precision.

3. **Q: Why might training loss become NaN?**

   A: Common causes: (1) Learning rate too high causing exploding gradients, (2) Division by zero, (3) Log of negative/zero values, (4) Overflow in softmax without the log-sum-exp trick.

### Advanced Questions

4. **Q: Explain the log-sum-exp trick.**

   A: Instead of computing `log(sum(exp(x)))` directly (which can overflow), we compute `max(x) + log(sum(exp(x - max(x))))`. This is mathematically equivalent but shifts values to prevent overflow.

5. **Q: How does mixed precision training work?**

   A: Master weights are stored in float32. Forward/backward passes use float16 for speed. Loss scaling prevents gradient underflow. Weight updates happen in float32 for precision.

6. **Q: What is quantization-aware training (QAT)?**

   A: QAT simulates quantization during training by inserting fake quantization operators. The model learns to be robust to quantization noise, resulting in better accuracy when actually quantized for deployment.

---

## Key Formulas Summary

| Concept           | Formula                                          | Description                |
| ----------------- | ------------------------------------------------ | -------------------------- |
| Complex magnitude | $\|z\| = \sqrt{a^2 + b^2}$                       | Distance from origin       |
| Euler's formula   | $e^{i\theta} = \cos\theta + i\sin\theta$         | Links exponential and trig |
| IEEE 754 value    | $(-1)^S \times 2^{E-bias} \times (1+M)$          | Float encoding             |
| Quantization      | $q = round((x - zp) / scale)$                    | Float to int               |
| Dequantization    | $x' = q \times scale + zp$                       | Int back to float          |
| Machine epsilon   | $\epsilon: 1 + \epsilon \neq 1$                  | Precision limit            |
| Softmax (stable)  | $\frac{e^{x_i - max(x)}}{\sum e^{x_j - max(x)}}$ | Numerically stable         |

---

## Further Reading

### Videos

- 📺 [3Blue1Brown - Euler's Formula](https://www.youtube.com/watch?v=mvmuCPvRoWQ)
- 📺 [3Blue1Brown - What is e?](https://www.youtube.com/watch?v=m2MIpDrF7Es)
- 📺 [Computerphile - Floating Point Numbers](https://www.youtube.com/watch?v=PZRI1IfStY0)
- 📺 [Computerphile - Why 0.1 + 0.2 ≠ 0.3](https://www.youtube.com/watch?v=2gIxbTn7GSc)

### Documentation

- 📖 [IEEE 754 Standard](https://en.wikipedia.org/wiki/IEEE_754)
- 📖 [NVIDIA Mixed Precision Training](https://developer.nvidia.com/automatic-mixed-precision)
- 📖 [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
- 📖 [llama.cpp Quantization Types](https://github.com/ggerganov/llama.cpp)

### Papers

- 📄 [Mixed Precision Training (Micikevicius et al.)](https://arxiv.org/abs/1710.03740)
- 📄 [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- 📄 [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)

---

## What's Next?

After mastering number systems, proceed to:
→ [Sets and Logic](../02-Sets-and-Logic/README.md) - Foundation for understanding data structures and boolean operations in ML

---

_Last updated: January 2025_
