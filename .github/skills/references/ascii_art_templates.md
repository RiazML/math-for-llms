# ASCII Art Visualization Guidelines for AI/ML Mathematics Modules

## Purpose and Philosophy

ASCII art visualizations serve as **immediate, text-based representations** of mathematical concepts that:
- ✅ Render instantly in any text editor, terminal, or markdown viewer
- ✅ Require no external libraries or image rendering
- ✅ Provide quick conceptual understanding before detailed visualizations
- ✅ Work in code comments, documentation, and plain text environments
- ✅ Are accessible to screen readers and text-based interfaces

**Golden Rule:** Every major mathematical concept should have BOTH ASCII art (for immediate intuition) AND proper visualizations (for detailed exploration).

---

## When to Use ASCII Art

### Required ASCII Art (MUST INCLUDE)

Use ASCII art for:

1. **Matrix Structures and Operations**
   - Matrix dimensions and shapes
   - Matrix multiplication visualization
   - Matrix transformations
   - Special matrix types (diagonal, symmetric, etc.)

2. **Vector Operations**
   - Vector addition/subtraction
   - Dot products
   - Cross products
   - Vector projections

3. **Neural Network Architectures**
   - Layer connections
   - Forward pass flow
   - Backward pass flow
   - Network topology

4. **Computational Graphs**
   - Operation nodes
   - Data flow
   - Gradient flow
   - Graph structure

5. **Coordinate Systems and Transformations**
   - 2D/3D coordinate axes
   - Transformations (rotation, scaling, shearing)
   - Basis vectors
   - Projections

6. **Probability Distributions**
   - Distribution shapes (normal, uniform, etc.)
   - PDF/CDF curves
   - Joint distributions
   - Conditional probabilities

7. **Optimization Landscapes**
   - Gradient descent paths
   - Local/global minima
   - Contour lines
   - Loss surfaces

8. **Algorithm Flowcharts**
   - Step-by-step procedures
   - Decision trees
   - Iteration loops
   - Conditional branches

### Optional ASCII Art (RECOMMENDED)

Consider ASCII art for:
- Timeline diagrams
- Hierarchical structures
- Comparison tables with visual elements
- State transitions
- Data flow pipelines

---

## ASCII Art Style Guide

### General Principles

**Clarity:**
- Use clean, simple lines
- Adequate spacing between elements
- Clear labels and annotations
- Consistent character choices

**Consistency:**
- Use same style throughout repository
- Maintain character conventions (see below)
- Consistent spacing and alignment
- Uniform label positioning

**Accessibility:**
- Works in monospace fonts
- Readable at standard terminal width (80-120 chars)
- Clear without color
- Meaningful when read by screen readers

---

## Character Conventions

### Standard Character Set

**Boxes and Borders:**
```
┌─┬─┐   ╔═╦═╗   ┏━┳━┓   +---+
├─┼─┤   ╠═╬═╣   ┣━╋━┫   |   |
└─┴─┘   ╚═╩═╝   ┗━┻━┛   +---+

Use ┌─┐ style for matrices, boxes, and tables
Use ╔═╗ style for emphasis or nested structures
Use +--+ style for maximum compatibility
```

**Arrows:**
```
→  ←  ↑  ↓  ↔  ⇒  ⇐  ⇔    (Unicode arrows - preferred)
->  <-  ^  v  <->  =>     (ASCII arrows - compatibility)

Curved: ↱  ↰  ↲  ↳  ⤴  ⤵
Thick:  ▶  ◀  ▲  ▼
```

**Connection Lines:**
```
Horizontal:  ─  ═  —  -
Vertical:    │  ║  |
Corners:     └ ┘ ┌ ┐ ├ ┤ ┬ ┴ ┼
Curves:      ╭ ╮ ╰ ╯
```

**Mathematical Symbols:**
```
Multiplication:  ×  ·  *
Division:        ÷  /
Plus/Minus:      ±
Approximately:   ≈  ~
Not equal:       ≠  !=
Less/Greater:    ≤  ≥  <  >
Infinity:        ∞
Sum:             Σ
Product:         Π
Square root:     √
Integral:        ∫
```

**Shapes and Markers:**
```
Dots:       ·  •  ○  ●  ◦  ⊙
Squares:    □  ■  ▢  ▣
Triangles:  △  ▲  ▽  ▼  ◁  ◀  ▷  ▶
Stars:      ☆  ★
Circles:    ○  ●  ◎  ⊙  ◉
```

**Grid and Graph:**
```
Origin:     +  ┼  ⊕
Axis:       ─  │  ┼
Points:     •  ●  ×  +
Lines:      /  \  |  -  ·
```

---

## Template Library

### 1. Matrix Visualization

#### Standard Matrix Display
```
Basic Matrix:

    ┌           ┐
    │ a₁₁  a₁₂ │
A = │ a₂₁  a₂₂ │
    │ a₃₁  a₃₂ │
    └           ┘

With Values:

    ┌         ┐
    │  4   1  │
A = │  2   3  │
    └         ┘

Large Matrix with Dimensions:

         n columns
    ┌─────────────┐
    │ a₁₁ ··· a₁ₙ │
m   │  ⋮   ⋱   ⋮  │  = A (m×n)
    │ aₘ₁ ··· aₘₙ │
    └─────────────┘
```

#### Matrix Operations
```
Matrix Multiplication:

    ┌     ┐       ┌     ┐       ┌           ┐
    │ 1 2 │       │ 5 6 │       │ 1·5+2·7   │
A = │ 3 4 │   B = │ 7 8 │   AB= │ 3·5+4·7   │
    └     ┘       └     ┘       └           ┘

                                 ┌       ┐
                             =   │ 19 22 │
                                 │ 43 50 │
                                 └       ┘

Matrix-Vector Multiplication:

    ┌         ┐   ┌   ┐       ┌         ┐
    │ a₁₁ a₁₂ │   │ x │       │ a₁₁x+a₁₂y │
A = │ a₂₁ a₂₂ │ · │ y │  =  b=│ a₂₁x+a₂₂y │
    └         ┘   └   ┘       └         ┘

Transpose:

         ┌     ┐              ┌     ┐
         │ 1 2 │              │ 1 3 │
    A =  │ 3 4 │      Aᵀ =    │ 2 4 │
         └     ┘              └     ┘
```

#### Special Matrices
```
Identity Matrix:

    ┌         ┐
    │ 1  0  0 │
I = │ 0  1  0 │
    │ 0  0  1 │
    └         ┘

Diagonal Matrix:

    ┌         ┐
    │ λ₁ 0  0 │
D = │ 0  λ₂ 0 │
    │ 0  0  λ₃│
    └         ┘

Symmetric Matrix:

    ┌         ┐
    │ a  b  c │
S = │ b  d  e │  (Sᵀ = S)
    │ c  e  f │
    └         ┘

Upper Triangular:

    ┌         ┐
    │ × × × × │
    │ 0 × × × │
U = │ 0 0 × × │
    │ 0 0 0 × │
    └         ┘
```

---

### 2. Vector Visualization

#### Vector Representation
```
Column Vector:        Row Vector:

    ┌   ┐
    │ x │             v = [ x  y  z ]
v = │ y │
    │ z │
    └   ┘

Vector in 2D Space:

         y
         ↑
         │    v
         │   ↗
         │  /
         │ /
    ─────┼─────→ x
         │
         │

Vector Addition:

         ↑
         │      u+v
         │     ↗
         │    /
    u   │   / v
    ↗   │  ↗
   /    │ /
  /     │/
─────────────→

      u + v = [u₁+v₁, u₂+v₂]
```

#### Vector Operations
```
Dot Product:

    u · v = |u| |v| cos(θ)

         u
        ↗
       /  ) θ
      /   ↗ v
     /   /
    /   /
   ────────

    Result: scalar value

Cross Product (3D):

    u × v = vector perpendicular to both

         ↑ u×v
         │
         │
    u   │   v
    ↗   │  ↗
   /    │ /
  /     │/
─────────────

    Result: vector normal to plane

Vector Projection:

    proj_u(v) = (v·u/|u|²)u

         v
        ↗
       /│
      / │
     /  │ (v - proj_u v)
    /   │
   ↗────┴────→ u
   proj_u(v)
```

---

### 3. Neural Network Architectures

#### Simple Feedforward Network
```
Basic 3-Layer Network:

Input     Hidden    Output
Layer     Layer     Layer

  ○         ●         ○
   ╲       ╱ ╲       ╱
    ╲     ╱   ╲     ╱
  ○─────●─────●─────○
    ╱     ╲   ╱     ╲
   ╱       ╲ ╱       ╲
  ○         ●         ○
            │
         (3→2→1)

With Dimensions:

  x₁ ○───┐
         ├──● h₁ ──┐
  x₂ ○───┤         ├── ○ ŷ
         ├──● h₂ ──┘
  x₃ ○───┘

  (3)    (2)      (1)
```

#### Deep Network with Skip Connections
```
ResNet-style Architecture:

Input ──→ [Conv] ──→ [Conv] ──→ (+) ──→ Output
         │                      ↑
         └──────────────────────┘
              (skip connection)

Detailed:

x ──→ ┌────────┐    ┌────────┐    ┌───┐
      │ Conv1  │───→│ Conv2  │───→│ + │──→ y
      └────────┘    └────────┘    └─↑─┘
         │                          │
         └──────────────────────────┘
              F(x) + x = y
```

#### Multi-Head Attention
```
Transformer Attention Mechanism:

    Input Sequence
         │
    ┌────┴────┬────┬────┐
    │         │    │    │
    ▼         ▼    ▼    ▼
  Query      Key  Value ...
    │         │    │
    └────┬────┴────┘
         │
    [Attention]
         │
    Weighted Sum
         │
       Output

Multi-Head:

Input ─┬─→ [Head 1] ─┬─→ Concat ─→ Linear ─→ Output
       ├─→ [Head 2] ─┤
       ├─→ [Head 3] ─┤
       └─→ [Head 4] ─┘
```

---

### 4. Computational Graphs

#### Forward Pass
```
Simple Expression: z = (x + y) × w

     x ──┐
         ├──[+]── a ──┐
     y ──┘            ├──[×]── z
                  w ──┘

Values Flow:
    x=2 ──┐
          ├──[+]── a=5 ──┐
    y=3 ──┘               ├──[×]── z=15
                   w=3 ──┘
```

#### Backward Pass (Backpropagation)
```
Gradient Flow (∂L/∂z = 1):

         ∂L/∂x=3
           ↓
     x ──[+]── a ──[×]── z
           ↑        ↑      ↑
         ∂L/∂y=3  ∂L/∂w=5  ∂L/∂z=1

Chain Rule Application:

    Forward:  x → [+] → a → [×] → z
              y ↗       w ↗

    Backward: ∂L/∂x ← [+] ← ∂L/∂a ← [×] ← ∂L/∂z
              ∂L/∂y ↖       ∂L/∂w ↖
```

#### Complex Graph
```
Expression: f = (a×b) + (a×c)

         a ──┬──[×]── d ──┐
             │            ├──[+]── f
         b ──┘        e ──┘
                      ↑
         a ──┬──[×]───┘
             │
         c ──┘

With Gradients:

    ∂f/∂a = b+c
    ∂f/∂b = a
    ∂f/∂c = a
    ∂f/∂d = 1
    ∂f/∂e = 1
```

---

### 5. Coordinate Systems and Transformations

#### 2D Coordinate System
```
Standard Cartesian:

         y
         ↑
