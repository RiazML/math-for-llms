# Convex Optimization

> **Navigation**: [← 04-Expectation-and-Moments](../../06-Probability-Theory/04-Expectation-and-Moments/) | [Optimization](../) | [02-Gradient-Descent →](../02-Gradient-Descent/)

**Files in this section:**
- [examples.ipynb](examples.ipynb) - 12 worked examples
- [exercises.ipynb](exercises.ipynb) - 10 practice problems with solutions

---

## Introduction

Convex optimization represents one of the most profound and practical achievements in mathematical optimization. At its core, convex optimization studies problems where both the objective function and the feasible region possess a special geometric property—convexity—that fundamentally transforms the nature of optimization. This property guarantees that any locally optimal solution is also globally optimal, eliminating the need to search through a landscape of potentially suboptimal local minima.

The significance of convex optimization for machine learning cannot be overstated. When we train a linear regression model, fit a logistic classifier, or optimize the soft-margin SVM, we are solving convex optimization problems. The theoretical guarantees provided by convexity—convergence to global optima, polynomial-time algorithms, and well-characterized solution properties—form the foundation upon which these algorithms reliably operate. Even when we venture into non-convex territory with deep neural networks, the insights from convex optimization guide our intuitions about algorithm behavior and convergence.

Historically, the study of convex optimization traces back to the early work on linear programming by Kantorovich and Dantzig in the 1940s. The field matured significantly with the development of interior-point methods in the 1980s, which demonstrated that a broad class of convex problems could be solved in polynomial time. Today, convex optimization pervades virtually every quantitative discipline: portfolio optimization in finance, signal reconstruction in communications, experiment design in statistics, and model training in machine learning all rely on convex optimization's elegant theory and efficient algorithms.

This chapter provides a comprehensive treatment of convex optimization, beginning with the foundational concepts of convex sets and functions, progressing through the elegant theory of Lagrangian duality, and culminating in the powerful KKT conditions that characterize optimal solutions. Throughout, we emphasize both rigorous mathematical development and practical intuition, with extensive worked examples drawn from machine learning applications.

## Prerequisites

- Linear algebra: matrices, eigenvalues, positive (semi-)definiteness
- Multivariable calculus: gradients, Hessians, Taylor expansions
- Basic optimization concepts: minima, critical points, constraints
- Probability basics (for some ML applications)

## Learning Objectives

By the end of this chapter, you will be able to:

1. **Define and recognize** convex sets and convex functions using multiple equivalent characterizations
2. **Verify convexity** using the Hessian, first-order conditions, or composition rules
3. **Formulate** machine learning problems as convex optimization problems
4. **Derive and interpret** the Lagrangian dual problem and understand duality gaps
5. **Apply KKT conditions** to characterize optimal solutions and derive closed-form solutions
6. **Understand** why local minima equal global minima for convex problems (with proof)
7. **Recognize** non-convex problems and apply convex relaxation techniques

---

## 1. Convex Sets

### 1.1 Definition and Geometric Intuition

A set $C \subseteq \mathbb{R}^n$ is **convex** if for any two points $x, y \in C$ and any $\theta \in [0, 1]$, the point $\theta x + (1 - \theta) y$ also belongs to $C$:

$$\theta x + (1 - \theta) y \in C \quad \forall x, y \in C, \; \theta \in [0, 1]$$

The geometric interpretation is immediate and powerful: a set is convex if the line segment connecting any two points in the set lies entirely within the set. This seemingly simple definition has profound implications. It means that if you stand at any point in the set and walk in a straight line toward any other point in the set, you never leave the set.

Consider why this matters. When we optimize over a convex set, we need not worry about the feasible region having "holes" or "indentations" that might trap our optimization algorithm. The optimization landscape, at least in terms of the constraint region, has no hidden corners to explore.

The definition extends naturally to convex combinations of more than two points. A **convex combination** of points $x_1, \ldots, x_k$ is any point of the form:

$$\sum_{i=1}^{k} \theta_i x_i \quad \text{where} \quad \theta_i \geq 0 \;\text{and}\; \sum_{i=1}^{k} \theta_i = 1$$

A set is convex if and only if it contains all convex combinations of its points.

```
         Convex Sets                           Non-Convex Sets

    ╭───────────────────╮               ╭──────╮    ╭──────╮
    │                   │               │      ╰────╯      │
    │    x •────────• y │               │  x •        • y  │
    │     (line in set) │               │     ╲      ╱     │
    │                   │               │      ╲____╱      │
    ╰───────────────────╯               ╰──────────────────╯
          Ellipse                         Line leaves the set!

    ╭─────────────────╮                     ●───────────●
    │●               ●│                    ╱             ╲
    │ ╲             ╱ │                   ╱               ╲
    │  ╲           ╱  │                  ●                 ●
    │   ●─────────●   │                   ╲               ╱
    ╰─────────────────╯                    ● ● ● ● ● ● ● ●
     Convex Polygon                     Star (non-convex)
```

### 1.2 Important Examples of Convex Sets

Understanding the palette of convex sets is essential for recognizing convexity in optimization problems. Here we catalog the most important examples:

**Hyperplanes and Affine Sets:**
A hyperplane is a set of the form $\{x : a^T x = b\}$ where $a \neq 0$. This is the set of all points with a constant inner product with $a$. Geometrically, a hyperplane is an $(n-1)$-dimensional flat surface in $\mathbb{R}^n$. More generally, an affine set is a translation of a subspace: $\{x : Ax = b\}$. All affine sets are convex.

**Halfspaces:**
A halfspace is defined by $\{x : a^T x \leq b\}$ or $\{x : a^T x \geq b\}$. It represents all points on one side of a hyperplane. Halfspaces are fundamental building blocks of convex sets.

**Balls and Ellipsoids:**
The Euclidean ball $B(x_c, r) = \{x : \|x - x_c\|_2 \leq r\}$ centered at $x_c$ with radius $r$ is convex. More generally, an ellipsoid $\mathcal{E} = \{x : (x - x_c)^T P^{-1}(x - x_c) \leq 1\}$ where $P \succ 0$ is symmetric positive definite, is convex. Ellipsoids generalize balls by allowing different "radii" in different directions.

**Polyhedra:**
A polyhedron is the intersection of a finite number of halfspaces and hyperplanes:
$$\mathcal{P} = \{x : Ax \leq b, Cx = d\}$$
This includes polygons (2D), polytopes (bounded polyhedra), simplices, and cubes. Polyhedra appear constantly in linear and quadratic programming.

**Positive Semidefinite Cone:**
The set $S^n_+ = \{X \in \mathbb{R}^{n \times n} : X = X^T, X \succeq 0\}$ of symmetric positive semidefinite matrices is a convex cone. This set is crucial in semidefinite programming and appears in covariance estimation, kernel methods, and beyond.

**Norm Balls:**
For any norm $\|\cdot\|$, the unit ball $\{x : \|x\| \leq 1\}$ is convex. This includes the $\ell_1$ ball (diamond/cross-polytope), $\ell_2$ ball (Euclidean ball), $\ell_\infty$ ball (hypercube), and more exotic norms like the nuclear norm for matrices.

| Set | Definition | Dimension | Common Use |
|-----|------------|-----------|------------|
| Hyperplane | $\{x: a^Tx = b\}$ | $n-1$ | Equality constraints |
| Halfspace | $\{x: a^Tx \leq b\}$ | $n$ | Inequality constraints |
| Euclidean Ball | $\{x: \|x - c\|_2 \leq r\}$ | $n$ | Trust regions |
| Polyhedron | $\{x: Ax \leq b\}$ | varies | LP feasible region |
| PSD Cone | $\{X: X \succeq 0\}$ | $n(n+1)/2$ | SDP constraints |
| Probability Simplex | $\{x: x \geq 0, \mathbf{1}^T x = 1\}$ | $n-1$ | Probability distributions |

### 1.3 Operations That Preserve Convexity

One of the most powerful aspects of convex sets is that convexity is preserved under many natural operations. This allows us to build complex convex sets from simple ones.

**Intersection:**
If $C_1$ and $C_2$ are convex, then $C_1 \cap C_2$ is convex. This extends to arbitrary (even infinite) intersections: if $C_\alpha$ is convex for all $\alpha \in \mathcal{A}$, then $\bigcap_{\alpha \in \mathcal{A}} C_\alpha$ is convex.

*Proof:* Let $x, y \in C_1 \cap C_2$ and $\theta \in [0, 1]$. Since $x, y \in C_1$ and $C_1$ is convex, $\theta x + (1-\theta)y \in C_1$. Similarly, $\theta x + (1-\theta)y \in C_2$. Therefore $\theta x + (1-\theta)y \in C_1 \cap C_2$. $\square$

This is why polyhedra (intersections of halfspaces) are convex!

**Affine Transformations:**
If $C$ is convex and $f(x) = Ax + b$ is an affine function, then:
- The image $f(C) = \{Ax + b : x \in C\}$ is convex
- The preimage $f^{-1}(C) = \{x : Ax + b \in C\}$ is convex

**Scaling and Translation:**
If $C$ is convex, then $\alpha C + \beta = \{\alpha x + \beta : x \in C\}$ is convex for any scalar $\alpha$ and vector $\beta$.

**Minkowski Sum:**
If $C_1$ and $C_2$ are convex, then $C_1 + C_2 = \{x + y : x \in C_1, y \in C_2\}$ is convex.

**Projection:**
If $C \subseteq \mathbb{R}^n \times \mathbb{R}^m$ is convex, its projection onto $\mathbb{R}^n$ is convex:
$$\text{proj}(C) = \{x \in \mathbb{R}^n : \exists y \in \mathbb{R}^m, (x, y) \in C\}$$

> **📊 Why This Matters for ML**
> 
> The constraint sets in machine learning optimization problems are typically constructed from simple convex building blocks using these operations. For example:
> - **Ridge regression**: unconstrained (all of $\mathbb{R}^n$, trivially convex)
> - **Lasso with bounds**: polyhedron $\{w : \|w\|_1 \leq t, w \geq 0\}$
> - **SVM margin constraints**: intersection of halfspaces $\{w : y_i(w^T x_i + b) \geq 1\}$
> - **Probability simplex**: $\{p : p \geq 0, \sum_i p_i = 1\}$ for mixture models
> 
> Knowing these operations lets you quickly verify that your feasible region is convex.

### 1.4 Worked Example: Verifying Convexity of a Set

**Example 1.1:** Prove that the probability simplex $\Delta_n = \{x \in \mathbb{R}^n : x_i \geq 0, \sum_{i=1}^n x_i = 1\}$ is convex.

**Solution:** We can show this in two ways:

*Method 1 (Direct):* Let $x, y \in \Delta_n$ and $\theta \in [0, 1]$. Let $z = \theta x + (1-\theta) y$.
- Non-negativity: For each $i$, $z_i = \theta x_i + (1-\theta) y_i \geq \theta \cdot 0 + (1-\theta) \cdot 0 = 0$ ✓
- Sum to one: $\sum_i z_i = \sum_i (\theta x_i + (1-\theta) y_i) = \theta \sum_i x_i + (1-\theta) \sum_i y_i = \theta \cdot 1 + (1-\theta) \cdot 1 = 1$ ✓

Therefore $z \in \Delta_n$, so $\Delta_n$ is convex.

*Method 2 (Intersection):* Write $\Delta_n$ as:
$$\Delta_n = \{x : -x_1 \leq 0\} \cap \cdots \cap \{x : -x_n \leq 0\} \cap \{x : \mathbf{1}^T x = 1\}$$

This is an intersection of $n$ halfspaces (requiring non-negativity) and one hyperplane (requiring sum equals 1). Since each component set is convex and intersection preserves convexity, $\Delta_n$ is convex. $\square$

**Example 1.2:** Is the set $S = \{(x, y) \in \mathbb{R}^2 : xy \geq 1, x > 0, y > 0\}$ convex?

**Solution:** Yes, this set is convex. To see this, rewrite the defining inequality as $y \geq 1/x$ for $x > 0$. The function $f(x) = 1/x$ is convex on $x > 0$ (since $f''(x) = 2/x^3 > 0$), and the set above the graph of a convex function is convex.

Alternatively, this is the hyperbolic cone, which can be shown to be convex via direct verification or as the image of the second-order cone under an affine transformation. $\square$

---

## 2. Convex Functions

### 2.1 Definition and Multiple Characterizations

A function $f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if its domain $\text{dom}(f)$ is a convex set and for all $x, y \in \text{dom}(f)$ and $\theta \in [0, 1]$:

$$f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$$

This definition has a beautiful geometric interpretation: the chord connecting any two points on the graph of $f$ lies on or above the graph itself. Equivalently, the set of points above the graph of $f$—the **epigraph** $\text{epi}(f) = \{(x, t) : f(x) \leq t\}$—is a convex set.

A function is **strictly convex** if the inequality is strict whenever $x \neq y$ and $\theta \in (0, 1)$:

$$f(\theta x + (1-\theta)y) < \theta f(x) + (1-\theta)f(y)$$

Strict convexity guarantees that any minimum is unique.

A function is **concave** if $-f$ is convex. Maximizing a concave function is equivalent to minimizing a convex function.

```
         Convex Function                      Concave Function

    f(x) │                               f(x) │    ╭──────╮
         │         ╱                          │   ╱        ╲
         │        ╱                           │  ╱          ╲
         │   ●───●───●  chord above          │ ●────●────●  chord below
         │  ╱    curve                        │╱     curve   ╲
         │ ╱                                  │                ╲
         │╱                                   │
         └────────────── x                    └────────────────── x

           f(θx + (1-θ)y) ≤                    f(θx + (1-θ)y) ≥
           θf(x) + (1-θ)f(y)                   θf(x) + (1-θ)f(y)
```

**Jensen's Inequality:** For a convex function $f$ and any convex combination:
$$f\left(\sum_{i=1}^k \theta_i x_i\right) \leq \sum_{i=1}^k \theta_i f(x_i)$$

This extends to expectations: if $f$ is convex and $X$ is a random variable, then:
$$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$$

Jensen's inequality is ubiquitous in machine learning, appearing in variational inference, EM algorithms, and information-theoretic bounds.

### 2.2 First-Order Characterization (Gradient Condition)

If $f$ is differentiable, $f$ is convex if and only if its domain is convex and:

$$f(y) \geq f(x) + \nabla f(x)^T (y - x) \quad \forall x, y \in \text{dom}(f)$$

**Geometric Interpretation:** The graph of $f$ lies on or above every tangent hyperplane. At each point, the linear (first-order Taylor) approximation globally underestimates the function.

```
    f(x) │
         │              ╱
         │             ╱
         │            ●  f(y)
         │           ╱│
         │          ╱ │
         │   ●─────╱──│── tangent line at x
         │  ╱f(x) ╱   │
         │ ╱     ╱    │ gap = f(y) - [f(x) + ∇f(x)ᵀ(y-x)] ≥ 0
         │╱     ╱     │
         └──────●─────●─── x
                x     y
```

**Proof (Necessity):** Assume $f$ is convex. For any $\theta \in (0, 1]$:
$$f(x + \theta(y - x)) \leq (1-\theta)f(x) + \theta f(y)$$

Rearranging:
$$f(y) \geq f(x) + \frac{f(x + \theta(y-x)) - f(x)}{\theta}$$

Taking $\theta \to 0^+$ and using the definition of the directional derivative:
$$f(y) \geq f(x) + \nabla f(x)^T(y - x) \quad \square$$

This first-order condition is extremely useful for proving optimality: if $\nabla f(x^*) = 0$ for a convex function, then for all $y$:
$$f(y) \geq f(x^*) + 0^T(y - x^*) = f(x^*)$$

So $x^*$ is a global minimum!

### 2.3 Second-Order Characterization (Hessian Condition)

If $f$ is twice continuously differentiable, $f$ is convex if and only if its domain is convex and its Hessian is positive semidefinite everywhere:

$$\nabla^2 f(x) \succeq 0 \quad \forall x \in \text{dom}(f)$$

Similarly:
- **Strictly convex** if $\nabla^2 f(x) \succ 0$ (positive definite) for all $x$
- **Concave** if $\nabla^2 f(x) \preceq 0$ (negative semidefinite) for all $x$

**Proof Sketch:** The Hessian is positive semidefinite if and only if all eigenvalues are non-negative. This means that the second directional derivative $v^T \nabla^2 f(x) v \geq 0$ in any direction $v$, so the function curves upward (or stays flat) in every direction.

**Checking Positive Semidefiniteness:**
For an $n \times n$ matrix $H$, $H \succeq 0$ if and only if:
1. All eigenvalues are $\geq 0$, or equivalently,
2. $v^T H v \geq 0$ for all vectors $v$, or equivalently,
3. All leading principal minors are $\geq 0$ (Sylvester's criterion), or equivalently,
4. $H$ can be written as $H = A^T A$ for some matrix $A$

For $2 \times 2$ matrices $H = \begin{pmatrix} a & b \\ b & c \end{pmatrix}$:
$$H \succeq 0 \iff a \geq 0, \; c \geq 0, \; \text{and} \; ac - b^2 \geq 0$$

### 2.4 Worked Example: Verifying Convexity via Hessian

**Example 2.1:** Prove that $f(x) = x^T A x + b^T x + c$ is convex if and only if $A \succeq 0$.

**Solution:** Compute the gradient and Hessian:
$$\nabla f(x) = 2Ax + b$$
$$\nabla^2 f(x) = 2A$$

The Hessian $2A$ is constant (doesn't depend on $x$). Thus:
- $f$ is convex $\iff \nabla^2 f(x) \succeq 0 \iff 2A \succeq 0 \iff A \succeq 0$
- $f$ is strictly convex $\iff A \succ 0$

**Numerical Example:** Consider $f(x_1, x_2) = 3x_1^2 + 2x_1x_2 + 3x_2^2 - 4x_1 + 5$.

This is quadratic with:
$$A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}, \quad b = \begin{pmatrix} -4 \\ 0 \end{pmatrix}, \quad c = 5$$

Check: $a = 3 > 0$ ✓, $c = 3 > 0$ ✓, $ac - b^2 = 9 - 1 = 8 > 0$ ✓

Since all conditions hold strictly, $A \succ 0$, so $f$ is strictly convex. $\square$

**Example 2.2:** Show that the logistic loss $f(w) = \log(1 + e^{-w^T x})$ is convex in $w$.

**Solution:** Let $z = -w^T x$. Then $f(w) = \log(1 + e^z)$ where $z$ is linear in $w$.

First, show $g(z) = \log(1 + e^z)$ is convex:
$$g'(z) = \frac{e^z}{1 + e^z} = \sigma(z) \quad \text{(sigmoid function)}$$
$$g''(z) = \sigma(z)(1 - \sigma(z)) = \sigma(z)\sigma(-z)$$

Since $\sigma(z) \in (0, 1)$, we have $g''(z) > 0$ for all $z$, so $g$ is strictly convex.

Now, for $f(w) = g(-w^T x)$:
$$\nabla_w f = g'(z) \cdot (-x) = -\sigma(z) x$$
$$\nabla^2_w f = g''(z) \cdot x x^T = \sigma(z)(1-\sigma(z)) \cdot x x^T$$

Since $\sigma(z)(1-\sigma(z)) > 0$ and $xx^T \succeq 0$ (rank-1 positive semidefinite), we have $\nabla^2_w f \succeq 0$.

Therefore, logistic loss is convex in $w$. $\square$

> **⚠️ Common Pitfalls**
> 
> 1. **Forgetting the domain must be convex**: A function can have a positive semidefinite Hessian on a non-convex domain and still not be convex overall.
> 
> 2. **Confusing strict/non-strict conditions**: $\nabla^2 f \succ 0$ implies strict convexity, but strict convexity does NOT imply $\nabla^2 f \succ 0$. Example: $f(x) = x^4$ is strictly convex but $f''(0) = 0$.
> 
> 3. **Not checking all points**: The Hessian must be positive semidefinite for ALL $x$ in the domain, not just at the minimum.
> 
> 4. **Sign errors with concavity**: Remember that for maximization problems, you want concave objectives. The negative log-likelihood is convex (good for minimization).

---

## 3. Examples of Convex and Non-Convex Functions

### 3.1 Catalog of Convex Functions

Understanding a library of convex functions helps you recognize convexity in complex expressions.

**Affine Functions:** $f(x) = a^T x + b$ is both convex and concave.

**Powers:** $f(x) = x^p$ is convex on $x > 0$ for $p \geq 1$ or $p \leq 0$, and convex on all of $\mathbb{R}$ for even positive integers.

**Exponential:** $f(x) = e^{ax}$ is convex for any $a \in \mathbb{R}$.

**Negative Logarithm:** $f(x) = -\log x$ is convex on $x > 0$.

**Negative Entropy:** $f(x) = x \log x$ is convex on $x > 0$.

**Norms:** Any norm $f(x) = \|x\|$ is convex. This includes $\ell_1$, $\ell_2$, $\ell_\infty$, and matrix norms.

**Max Function:** $f(x) = \max\{x_1, \ldots, x_n\}$ is convex.

**Log-Sum-Exp (Softmax):** $f(x) = \log(\sum_{i=1}^n e^{x_i})$ is convex. This smooth approximation to $\max$ is crucial in attention mechanisms.

**Quadratic-over-Linear:** $f(x, y) = x^2/y$ is convex on $\{(x, y) : y > 0\}$.

| Function | Domain | Convex? | Strictly Convex? |
|----------|--------|---------|------------------|
| $ax + b$ | $\mathbb{R}$ | ✓ (also concave) | ✗ |
| $x^2$ | $\mathbb{R}$ | ✓ | ✓ |
| $|x|$ | $\mathbb{R}$ | ✓ | ✗ |
| $e^x$ | $\mathbb{R}$ | ✓ | ✓ |
| $-\log x$ | $x > 0$ | ✓ | ✓ |
| $x \log x$ | $x > 0$ | ✓ | ✓ |
| $\|x\|_2$ | $\mathbb{R}^n$ | ✓ | ✗ |
| $\|x\|_2^2$ | $\mathbb{R}^n$ | ✓ | ✓ |
| $\|x\|_1$ | $\mathbb{R}^n$ | ✓ | ✗ |
| $\max_i x_i$ | $\mathbb{R}^n$ | ✓ | ✗ |
| $\log\sum_i e^{x_i}$ | $\mathbb{R}^n$ | ✓ | ✓ |

### 3.2 Machine Learning Loss Functions

**Mean Squared Error (MSE):**
$$\mathcal{L}(w) = \|Xw - y\|_2^2 = w^T X^T X w - 2y^T X w + y^T y$$

This is a quadratic in $w$ with Hessian $2X^T X \succeq 0$, so MSE is convex.

**Cross-Entropy Loss:**
$$\mathcal{L}(p) = -\sum_{i=1}^{n} y_i \log p_i$$

For fixed targets $y_i \geq 0$, this is convex in $p$ since $-\log$ is convex and positive combinations preserve convexity.

**Hinge Loss (SVM):**
$$\mathcal{L}(w) = \max(0, 1 - y \cdot w^T x)$$

This is the maximum of two affine functions (0 and $1 - yw^Tx$), hence convex.

**Logistic Loss:**
$$\mathcal{L}(w) = \log(1 + e^{-y \cdot w^T x})$$

As shown in Example 2.2, this is convex in $w$.

**Huber Loss:**
$$\mathcal{L}_\delta(r) = \begin{cases} \frac{1}{2}r^2 & |r| \leq \delta \\ \delta(|r| - \frac{\delta}{2}) & |r| > \delta \end{cases}$$

This is convex: quadratic (convex) near zero, linear (affine, hence convex) for large residuals, with continuous first derivative.

### 3.3 Non-Convex Functions and Why They Matter

**Powers with $p \in (0, 1)$:** $f(x) = x^p$ for $0 < p < 1$ is concave on $x > 0$.

**Sine and Cosine:** Oscillating functions are neither convex nor concave.

**Gaussian/Radial Basis Functions:** $f(x) = e^{-\|x\|^2}$ is not convex (it has a maximum, not minimum).

**Neural Network Losses:** The loss landscape of a neural network with multiple layers is highly non-convex due to:
- Composition of non-linear activations
- Weight-space symmetries (permuting hidden units)
- Saddle points and local minima

```
    Non-Convex Loss Landscape (Neural Network)

    Loss │     ╱╲        ╱╲
         │    ╱  ╲      ╱  ╲
         │   ╱    ╲    ╱    ╲
         │  ╱      ╲  ╱      ╲
         │ ╱        ╲╱        ╲    global
         │╱    local minimum   ╲   minimum
         └──────────────────────╲─────── weights
              ↑                   ↑
         SGD might get           or here
         stuck here
```

> **📊 Why This Matters for ML**
> 
> Understanding which loss functions are convex tells you what to expect from optimization:
> 
> | Model | Loss Convexity | Implication |
> |-------|---------------|-------------|
> | Linear Regression | ✓ Convex | Closed-form solution exists |
> | Logistic Regression | ✓ Convex | Gradient descent finds global opt |
> | SVM | ✓ Convex | Strong theoretical guarantees |
> | Neural Networks | ✗ Non-convex | No global optimum guarantee |
> | Matrix Factorization | ✗ Non-convex | Multiple equivalent solutions |
> 
> For convex problems, algorithm design focuses on speed (convergence rate).
> For non-convex problems, we hope for "good enough" local minima.

---

## 4. Operations That Preserve Convexity

Knowing how to build convex functions from simpler convex functions is invaluable for recognizing and constructing convex optimization problems.

### 4.1 Non-Negative Weighted Sums

If $f_1, \ldots, f_m$ are convex and $\alpha_1, \ldots, \alpha_m \geq 0$, then:
$$f(x) = \sum_{i=1}^{m} \alpha_i f_i(x)$$
is convex.

**Proof:** For any $x, y$ and $\theta \in [0, 1]$:
$$f(\theta x + (1-\theta)y) = \sum_i \alpha_i f_i(\theta x + (1-\theta)y) \leq \sum_i \alpha_i [\theta f_i(x) + (1-\theta)f_i(y)]$$
$$= \theta \sum_i \alpha_i f_i(x) + (1-\theta) \sum_i \alpha_i f_i(y) = \theta f(x) + (1-\theta)f(y) \quad \square$$

**ML Application:** Regularized objectives are convex:
$$\mathcal{L}(w) = \underbrace{\|Xw - y\|^2}_{\text{convex loss}} + \underbrace{\lambda \|w\|^2}_{\text{convex regularizer}}$$

### 4.2 Pointwise Maximum

If $f_1, \ldots, f_m$ are convex, then:
$$f(x) = \max_{i=1,\ldots,m} f_i(x)$$
is convex.

**Proof:** The epigraph of $\max_i f_i$ is the intersection of the epigraphs of each $f_i$:
$$\text{epi}(f) = \{(x, t) : f_i(x) \leq t \; \forall i\} = \bigcap_i \text{epi}(f_i)$$
Since each $\text{epi}(f_i)$ is convex (as $f_i$ is convex) and intersection preserves convexity, $\text{epi}(f)$ is convex, so $f$ is convex. $\square$

**ML Application:** Hinge loss is the max of two affine functions:
$$\max(0, 1 - y_i w^T x_i) = \max\{f_1(w), f_2(w)\}$$
where $f_1(w) = 0$ (affine) and $f_2(w) = 1 - y_i w^T x_i$ (affine). Hence convex.

### 4.3 Composition Rules

For $f = h \circ g$ (i.e., $f(x) = h(g(x))$):

| Outer $h$ | Inner $g$ | Condition on $h$ | $f = h \circ g$ |
|-----------|-----------|------------------|-----------------|
| Convex | Convex | Non-decreasing | **Convex** |
| Convex | Concave | Non-increasing | **Convex** |
| Convex | Affine | — | **Convex** |
| Concave | Convex | Non-increasing | **Concave** |
| Concave | Concave | Non-decreasing | **Concave** |
| Concave | Affine | — | **Concave** |

**Example Applications:**
- $e^{g(x)}$ is convex if $g$ is convex ($e^t$ is convex and increasing)
- $\log(g(x))$ is concave if $g$ is concave and positive ($\log t$ is concave and increasing)
- $1/g(x)$ is convex if $g$ is concave and positive ($1/t$ is convex and decreasing on $t > 0$)
- $\|Ax + b\|^2$ is convex ($g(x) = Ax + b$ is affine, $h(z) = \|z\|^2$ is convex)

### 4.4 Affine Transformation of Argument

If $f: \mathbb{R}^n \to \mathbb{R}$ is convex, then $g(x) = f(Ax + b)$ is convex.

This is crucial: linear regression loss $\|Xw - y\|^2 = \|Aw + b\|^2$ with $A = X$, $b = -y$ is the composition of $\|\cdot\|^2$ (convex) with an affine map, hence convex.

### 4.5 Partial Minimization

If $f(x, y)$ is convex in $(x, y)$ jointly, then:
$$g(x) = \inf_y f(x, y)$$
is convex in $x$ (when the infimum is achieved).

**ML Application:** This appears in dual formulations. The Lagrangian dual function $g(\lambda) = \inf_x L(x, \lambda)$ is concave in $\lambda$, even if the primal problem is non-convex.

### 4.6 Worked Example: Building Complex Convex Functions

**Example 4.1:** Show that the elastic net penalty $\Omega(w) = \alpha \|w\|_1 + (1-\alpha) \|w\|_2^2$ is convex for $\alpha \in [0, 1]$.

**Solution:**
- $\|w\|_1$ is convex (it's a norm)
- $\|w\|_2^2$ is convex (quadratic with Hessian $2I \succ 0$)
- $\alpha \geq 0$ and $(1-\alpha) \geq 0$ for $\alpha \in [0, 1]$
- Non-negative weighted sum of convex functions is convex ✓ $\square$

**Example 4.2:** Show that softmax cross-entropy loss $f(z) = \log\sum_{j} e^{z_j} - z_y$ is convex.

**Solution:**
- $\log\sum_j e^{z_j}$ is the log-sum-exp, known to be convex
- $-z_y$ is affine (hence convex)
- Sum of convex functions is convex ✓ $\square$

---

## 5. The Convex Optimization Problem

### 5.1 Standard Form

A convex optimization problem in **standard form** is:

$$
\begin{aligned}
\minimize_{x \in \mathbb{R}^n} \quad & f_0(x) \\
\text{subject to} \quad & f_i(x) \leq 0, \quad i = 1, \ldots, m \\
& a_j^T x = b_j, \quad j = 1, \ldots, p
\end{aligned}
$$

where:
- $f_0$ is the **objective function** (convex)
- $f_1, \ldots, f_m$ are **inequality constraint functions** (convex)
- $a_j^T x = b_j$ are **equality constraints** (affine)

The **feasible set** is $\mathcal{X} = \{x : f_i(x) \leq 0, \; a_j^T x = b_j\}$, which is convex (intersection of sublevel sets of convex functions and hyperplanes).

**Optimal value:** $p^* = \inf\{f_0(x) : x \in \mathcal{X}\}$

**Optimal set:** $\mathcal{X}^* = \{x \in \mathcal{X} : f_0(x) = p^*\}$

### 5.2 The Fundamental Theorem: Local Implies Global

**Theorem (Local = Global for Convex Problems):** For a convex optimization problem, any locally optimal point is globally optimal.

**Proof:** Suppose $x^*$ is locally optimal but not globally optimal. Then there exists $y \in \mathcal{X}$ with $f_0(y) < f_0(x^*)$.

Consider points on the line segment from $x^*$ to $y$:
$$z_\theta = \theta y + (1-\theta)x^* \quad \text{for } \theta \in (0, 1)$$

Since $\mathcal{X}$ is convex, $z_\theta \in \mathcal{X}$ for all $\theta \in [0, 1]$.

By convexity of $f_0$:
$$f_0(z_\theta) \leq \theta f_0(y) + (1-\theta)f_0(x^*) < \theta f_0(x^*) + (1-\theta)f_0(x^*) = f_0(x^*)$$

So for any $\theta > 0$ (no matter how small), we have $f_0(z_\theta) < f_0(x^*)$.

But $\|z_\theta - x^*\| = \theta\|y - x^*\| \to 0$ as $\theta \to 0$.

This means there are feasible points arbitrarily close to $x^*$ with strictly smaller objective values, contradicting that $x^*$ is a local minimum. $\square$

```
    Convex (local = global)               Non-Convex (local ≠ global)

    f(x) │                                f(x) │     ╱╲
         │╲                                    │    ╱  ╲
         │ ╲                                   │   ╱    ╲
         │  ╲                                  │  ╱      ╲  local
         │   ╲     any descent               │ ╱        ╲  minimum
         │    ╲    direction leads           │╱          ╲
         │     ╲   to global min             │            ╲╱ global
         │      ●                             │               ●
         └────────── x                        └──────────────── x
```

### 5.3 Optimality Conditions for Unconstrained Problems

For an unconstrained convex problem $\min_x f(x)$:

**Theorem:** If $f$ is convex and differentiable, then $x^*$ is a global minimum if and only if:
$$\nabla f(x^*) = 0$$

**Proof:**
- ($\Rightarrow$) Standard calculus: first-order necessary condition.
- ($\Leftarrow$) By first-order convexity condition, for all $y$:
  $$f(y) \geq f(x^*) + \nabla f(x^*)^T(y - x^*) = f(x^*) + 0 = f(x^*)$$
  So $x^*$ is a global minimum. $\square$

For constrained problems, the KKT conditions generalize this.

> **📊 Why This Matters for ML**
> 
> The local=global property has profound implications:
> 
> 1. **Algorithm simplicity**: Gradient descent, with appropriate step size, is guaranteed to find the global optimum for convex problems.
> 
> 2. **Reproducibility**: Different runs of optimization will converge to the same solution (for strictly convex problems), not different local minima.
> 
> 3. **Theoretical analysis**: Convergence rates and error bounds can be precisely characterized.
> 
> 4. **Model interpretation**: The solution is unique and meaningful, not one of many equivalent local optima.
> 
> This is why logistic regression, SVMs, and linear regression are so well-behaved compared to neural networks!

### 5.4 Worked Example: Formulating ML Problems

**Example 5.1:** Write Ridge Regression as a convex optimization problem and verify convexity.

**Problem:** Given data $(X, y)$ with $X \in \mathbb{R}^{n \times d}$, $y \in \mathbb{R}^n$, and regularization parameter $\lambda > 0$:

$$\minimize_{w \in \mathbb{R}^d} \quad \frac{1}{2}\|Xw - y\|_2^2 + \frac{\lambda}{2}\|w\|_2^2$$

**Verification:**
- Objective: $f(w) = \frac{1}{2}w^T(X^TX + \lambda I)w - y^TXw + \frac{1}{2}y^Ty$
- Hessian: $\nabla^2 f(w) = X^TX + \lambda I$
- Since $X^TX \succeq 0$ and $\lambda I \succ 0$, we have $X^TX + \lambda I \succ 0$
- Therefore $f$ is strictly convex ✓

The unique global minimum is:
$$w^* = (X^TX + \lambda I)^{-1}X^Ty$$

**Example 5.2:** Write binary SVM as a convex optimization problem.

**Hard-margin SVM (linearly separable):**
$$
\begin{aligned}
\minimize_{w, b} \quad & \frac{1}{2}\|w\|_2^2 \\
\text{subject to} \quad & y_i(w^T x_i + b) \geq 1, \quad i = 1, \ldots, n
\end{aligned}
$$

- Objective: $\frac{1}{2}\|w\|^2$ is strictly convex (Hessian = $I \succ 0$)
- Constraints: $1 - y_i(w^Tx_i + b) \leq 0$ are affine in $(w, b)$, hence convex
- This is a convex QP ✓

**Soft-margin SVM:**
$$
\begin{aligned}
\minimize_{w, b, \xi} \quad & \frac{1}{2}\|w\|_2^2 + C\sum_{i=1}^n \xi_i \\
\text{subject to} \quad & y_i(w^T x_i + b) \geq 1 - \xi_i, \quad i = 1, \ldots, n \\
& \xi_i \geq 0, \quad i = 1, \ldots, n
\end{aligned}
$$

- Objective is convex (quadratic + linear)
- All constraints are affine
- Convex QP ✓

---

## 6. Lagrangian Duality

Lagrangian duality provides a powerful framework for analyzing constrained optimization problems. It reformulates the problem in terms of dual variables, often revealing structure that makes the problem easier to solve or analyze.

### 6.1 The Lagrangian

For the optimization problem:
$$
\begin{aligned}
\minimize_x \quad & f_0(x) \\
\text{subject to} \quad & f_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, p
\end{aligned}
$$

The **Lagrangian** $L: \mathbb{R}^n \times \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$ is:

$$L(x, \lambda, \nu) = f_0(x) + \sum_{i=1}^{m} \lambda_i f_i(x) + \sum_{j=1}^{p} \nu_j h_j(x)$$

where:
- $\lambda_i \geq 0$ are **Lagrange multipliers** for inequality constraints
- $\nu_j \in \mathbb{R}$ are **Lagrange multipliers** for equality constraints
- $\lambda = (\lambda_1, \ldots, \lambda_m)^T$ and $\nu = (\nu_1, \ldots, \nu_p)^T$ are the **dual variables**

**Intuition:** The Lagrangian "softens" the constraints by adding penalty terms. If $\lambda_i > 0$ and $f_i(x) > 0$ (constraint violated), the Lagrangian increases. The multipliers enforce the constraints indirectly.

### 6.2 The Lagrange Dual Function

The **Lagrange dual function** $g: \mathbb{R}^m \times \mathbb{R}^p \to \mathbb{R}$ is the minimum of the Lagrangian over $x$:

$$g(\lambda, \nu) = \inf_{x \in \mathcal{D}} L(x, \lambda, \nu) = \inf_x \left[ f_0(x) + \sum_i \lambda_i f_i(x) + \sum_j \nu_j h_j(x) \right]$$

where $\mathcal{D} = \bigcap_{i=0}^m \text{dom}(f_i) \cap \bigcap_j \text{dom}(h_j)$.

**Key Property:** $g(\lambda, \nu)$ is **always concave**, regardless of whether the primal problem is convex. This is because $g$ is the pointwise infimum of a family of affine functions in $(\lambda, \nu)$.

**Proof of Concavity:** For fixed $x$, $L(x, \lambda, \nu)$ is affine in $(\lambda, \nu)$. The infimum over $x$ is the pointwise infimum of affine functions, which is concave. $\square$

### 6.3 Weak Duality

**Theorem (Weak Duality):** For any $\lambda \geq 0$ and any $\nu$:
$$g(\lambda, \nu) \leq p^*$$

where $p^* = \inf_x\{f_0(x) : f_i(x) \leq 0, h_j(x) = 0\}$ is the optimal primal value.

**Proof:** Let $\tilde{x}$ be any feasible point. Then:
$$L(\tilde{x}, \lambda, \nu) = f_0(\tilde{x}) + \sum_i \underbrace{\lambda_i}_{\geq 0} \underbrace{f_i(\tilde{x})}_{\leq 0} + \sum_j \nu_j \underbrace{h_j(\tilde{x})}_{= 0} \leq f_0(\tilde{x})$$

Therefore:
$$g(\lambda, \nu) = \inf_x L(x, \lambda, \nu) \leq L(\tilde{x}, \lambda, \nu) \leq f_0(\tilde{x})$$

Since this holds for all feasible $\tilde{x}$:
$$g(\lambda, \nu) \leq \inf_{\tilde{x} \text{ feasible}} f_0(\tilde{x}) = p^* \quad \square$$

### 6.4 The Dual Problem

The **Lagrange dual problem** is:
$$\maximize_{\lambda, \nu} \quad g(\lambda, \nu) \quad \text{subject to} \quad \lambda \geq 0$$

Let $d^* = \sup_{\lambda \geq 0, \nu} g(\lambda, \nu)$ be the **optimal dual value**.

By weak duality: $d^* \leq p^*$.

The difference $p^* - d^*$ is called the **duality gap**.

### 6.5 Strong Duality and Slater's Condition

**Strong duality** means $d^* = p^*$ (zero duality gap).

Strong duality holds under various **constraint qualifications**. The most important is:

**Slater's Condition:** If the primal problem is convex and there exists a **strictly feasible** point $x$ (i.e., $f_i(x) < 0$ for all $i$ and $h_j(x) = 0$ for all $j$), then strong duality holds.

When strong duality holds:
- The dual problem gives the same optimal value as the primal
- We can solve whichever is easier
- KKT conditions become necessary and sufficient

```
    Weak vs Strong Duality

    Value
      │
    p*├──────────────── primal optimal
      │
      │   duality gap = p* - d*
      │   (always ≥ 0 by weak duality)
      │
    d*├──────────────── dual optimal
      │
      │     Strong duality: gap = 0
      │     Weak duality: gap ≥ 0
      │
      └─────────────────────────────
```

### 6.6 Worked Example: Deriving a Dual Problem

**Example 6.1:** Find the dual of: $\min_x \frac{1}{2}x^2$ subject to $x \geq 1$.

**Step 1:** Write in standard form. The constraint $x \geq 1$ becomes $1 - x \leq 0$.

**Step 2:** Form the Lagrangian:
$$L(x, \lambda) = \frac{1}{2}x^2 + \lambda(1 - x) = \frac{1}{2}x^2 - \lambda x + \lambda$$

**Step 3:** Minimize over $x$:
$$\frac{\partial L}{\partial x} = x - \lambda = 0 \implies x^* = \lambda$$

**Step 4:** Substitute back:
$$g(\lambda) = L(\lambda, \lambda) = \frac{1}{2}\lambda^2 - \lambda \cdot \lambda + \lambda = -\frac{1}{2}\lambda^2 + \lambda$$

**Step 5:** The dual problem is:
$$\maximize_{\lambda \geq 0} \quad -\frac{1}{2}\lambda^2 + \lambda$$

**Step 6:** Solve the dual:
$$\frac{d g}{d \lambda} = -\lambda + 1 = 0 \implies \lambda^* = 1$$

Check: $\lambda^* = 1 \geq 0$ ✓

**Step 7:** Verify:
- Dual optimal: $d^* = g(1) = -\frac{1}{2} + 1 = \frac{1}{2}$
- Primal optimal: $p^* = \frac{1}{2}(1)^2 = \frac{1}{2}$ (at $x^* = 1$)
- Strong duality holds: $d^* = p^* = \frac{1}{2}$ ✓

> **📊 Why This Matters for ML**
> 
> Lagrangian duality is the foundation for:
> 
> 1. **SVM dual formulation**: The kernel trick works because the dual only depends on inner products $x_i^T x_j$
> 
> 2. **Regularization interpretation**: The regularization parameter $\lambda$ in $\min_w \text{loss}(w) + \lambda\|w\|^2$ is related to a Lagrange multiplier for a constraint $\|w\|^2 \leq t$
> 
> 3. **Convergence analysis**: Duality gaps provide stopping criteria for optimization algorithms
> 
> 4. **Constrained optimization**: Convert hard constraints to soft penalties

---

## 7. KKT Conditions

The Karush-Kuhn-Tucker (KKT) conditions are the centerpiece of constrained optimization. For convex problems with strong duality, they provide necessary and sufficient conditions for optimality.

### 7.1 Derivation of KKT Conditions

Consider a convex optimization problem where strong duality holds. Let $x^*$ be primal optimal and $(\lambda^*, \nu^*)$ be dual optimal. Since there's zero duality gap:

$$f_0(x^*) = g(\lambda^*, \nu^*) = \inf_x L(x, \lambda^*, \nu^*)$$

This means $x^*$ minimizes $L(x, \lambda^*, \nu^*)$ over $x$. For an unconstrained minimum:

$$\nabla_x L(x^*, \lambda^*, \nu^*) = 0$$

Expanding:
$$\nabla f_0(x^*) + \sum_{i=1}^m \lambda_i^* \nabla f_i(x^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(x^*) = 0$$

This is the **stationarity condition**.

Additionally, from weak duality derivation, we had:
$$f_0(x^*) = g(\lambda^*, \nu^*) \leq L(x^*, \lambda^*, \nu^*) = f_0(x^*) + \sum_i \lambda_i^* f_i(x^*) + \sum_j \nu_j^* h_j(x^*)$$

Since $h_j(x^*) = 0$ (primal feasibility):
$$f_0(x^*) \leq f_0(x^*) + \sum_i \lambda_i^* f_i(x^*)$$

This means $\sum_i \lambda_i^* f_i(x^*) \geq 0$.

But $\lambda_i^* \geq 0$ and $f_i(x^*) \leq 0$, so each term $\lambda_i^* f_i(x^*) \leq 0$.

For the sum to be $\geq 0$, each term must be exactly 0:
$$\lambda_i^* f_i(x^*) = 0 \quad \text{for all } i$$

This is **complementary slackness**.

### 7.2 The Complete KKT Conditions

For a convex problem with strong duality, $(x^*, \lambda^*, \nu^*)$ is primal-dual optimal if and only if:

**1. Primal Feasibility:**
$$f_i(x^*) \leq 0 \quad \text{for } i = 1, \ldots, m$$
$$h_j(x^*) = 0 \quad \text{for } j = 1, \ldots, p$$

**2. Dual Feasibility:**
$$\lambda_i^* \geq 0 \quad \text{for } i = 1, \ldots, m$$

**3. Complementary Slackness:**
$$\lambda_i^* f_i(x^*) = 0 \quad \text{for } i = 1, \ldots, m$$

**4. Stationarity:**
$$\nabla f_0(x^*) + \sum_{i=1}^m \lambda_i^* \nabla f_i(x^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(x^*) = 0$$

### 7.3 Interpretation of Each Condition

**Primal Feasibility:** The solution $x^*$ must satisfy all constraints. This is the minimal requirement for any candidate solution.

**Dual Feasibility:** Lagrange multipliers for inequality constraints must be non-negative. Intuitively, "pushing" on an inequality constraint should increase the objective, not decrease it.

**Complementary Slackness:** For each inequality constraint, either:
- The constraint is **active** (binding): $f_i(x^*) = 0$ — the constraint is tight at the optimum
- The multiplier is zero: $\lambda_i^* = 0$ — the constraint doesn't affect the optimum

This means only active constraints "matter" at the optimum.

```
    Complementary Slackness Visualization

    Feasible           x*              Feasible
    Region         (optimum)           Region
    ┌──────────────────●──────────────────┐
    │                  │                  │
    │   λ₁* = 0        │      λ₂* > 0     │
    │   (inactive)     │      (active)    │
    │   f₁(x*) < 0     │      f₂(x*) = 0  │
    │                  │                  │
    └──────────────────┴──────────────────┘
                       ↑
                  constraint 2 is
                  binding at x*
```

**Stationarity:** At the optimum, the gradient of the objective is a non-negative combination of the constraint gradients. Geometrically, you can't improve the objective without violating a constraint.

### 7.4 Worked Example: Applying KKT Conditions

**Example 7.1:** Solve $\min_x \frac{1}{2}(x_1^2 + x_2^2)$ subject to $x_1 + x_2 \geq 2$ using KKT conditions.

**Step 1:** Convert to standard form. Constraint becomes $2 - x_1 - x_2 \leq 0$.

**Step 2:** Write the Lagrangian:
$$L(x, \lambda) = \frac{1}{2}(x_1^2 + x_2^2) + \lambda(2 - x_1 - x_2)$$

**Step 3:** Write KKT conditions:

*Stationarity:*
$$\frac{\partial L}{\partial x_1} = x_1 - \lambda = 0 \implies x_1 = \lambda$$
$$\frac{\partial L}{\partial x_2} = x_2 - \lambda = 0 \implies x_2 = \lambda$$

*Primal feasibility:* $x_1 + x_2 \geq 2$

*Dual feasibility:* $\lambda \geq 0$

*Complementary slackness:* $\lambda(2 - x_1 - x_2) = 0$

**Step 4:** Solve the system.

From stationarity: $x_1 = x_2 = \lambda$.

*Case 1:* $\lambda = 0$. Then $x_1 = x_2 = 0$. Check primal feasibility: $0 + 0 = 0 \not\geq 2$. ✗

*Case 2:* $\lambda > 0$. By complementary slackness, $2 - x_1 - x_2 = 0$, so $x_1 + x_2 = 2$.
Substituting $x_1 = x_2 = \lambda$: $2\lambda = 2 \implies \lambda = 1$.
So $x_1^* = x_2^* = 1$, $\lambda^* = 1$.

**Step 5:** Verify all KKT conditions:
- Stationarity: $x_1 = \lambda$ ✓, $x_2 = \lambda$ ✓
- Primal feasibility: $1 + 1 = 2 \geq 2$ ✓
- Dual feasibility: $\lambda = 1 \geq 0$ ✓
- Complementary slackness: $1 \cdot (2 - 1 - 1) = 0$ ✓

**Solution:** $x^* = (1, 1)$, optimal value $= \frac{1}{2}(1 + 1) = 1$.

**Example 7.2:** Derive the closed-form solution for Ridge Regression using KKT.

**Problem:** $\min_w \frac{1}{2}\|Xw - y\|^2 + \frac{\lambda}{2}\|w\|^2$

This is unconstrained, so KKT reduces to stationarity:
$$\nabla_w \left[\frac{1}{2}\|Xw - y\|^2 + \frac{\lambda}{2}\|w\|^2\right] = 0$$

Compute:
$$X^T(Xw - y) + \lambda w = 0$$
$$X^TXw + \lambda w = X^Ty$$
$$(X^TX + \lambda I)w = X^Ty$$

Therefore:
$$w^* = (X^TX + \lambda I)^{-1}X^Ty$$

This is the famous Ridge Regression closed-form solution! $\square$

> **⚠️ Common Pitfalls**
>
> 1. **Forgetting to check all cases**: When solving KKT, you must consider all possible combinations of active/inactive constraints.
>
> 2. **Sign errors**: Be careful about the sign convention for constraints. Standard form uses $f_i(x) \leq 0$.
>
> 3. **Assuming strong duality**: KKT conditions are necessary and sufficient only when strong duality holds. Check Slater's condition.
>
> 4. **Non-differentiable functions**: For functions like the $\ell_1$ norm, use subgradients instead of gradients.

---

## 8. Machine Learning Applications

### 8.1 Ridge Regression (L2 Regularization)

**Problem:**
$$\minimize_w \quad \frac{1}{2}\|Xw - y\|_2^2 + \frac{\lambda}{2}\|w\|_2^2$$

**Convexity:** Strictly convex (sum of strictly convex quadratics)

**Solution:** $w^* = (X^TX + \lambda I)^{-1}X^Ty$

**Numerical Example:**
Let $X = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{pmatrix}$, $y = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$, $\lambda = 0.1$.

$$X^TX = \begin{pmatrix} 35 & 44 \\ 44 & 56 \end{pmatrix}, \quad X^Ty = \begin{pmatrix} 22 \\ 28 \end{pmatrix}$$

$$X^TX + \lambda I = \begin{pmatrix} 35.1 & 44 \\ 44 & 56.1 \end{pmatrix}$$

$$(X^TX + \lambda I)^{-1} \approx \begin{pmatrix} 1.508 & -1.183 \\ -1.183 & 0.944 \end{pmatrix}$$

$$w^* \approx \begin{pmatrix} 1.508 & -1.183 \\ -1.183 & 0.944 \end{pmatrix} \begin{pmatrix} 22 \\ 28 \end{pmatrix} \approx \begin{pmatrix} 0.03 \\ 0.40 \end{pmatrix}$$

### 8.2 Lasso (L1 Regularization)

**Problem:**
$$\minimize_w \quad \frac{1}{2}\|Xw - y\|_2^2 + \lambda\|w\|_1$$

**Convexity:** Convex (but not strictly convex, not differentiable everywhere)

**Key Property:** Produces sparse solutions (many $w_i = 0$)

**Why L1 gives sparsity:** The L1 ball $\{w : \|w\|_1 \leq t\}$ has corners on the axes. The optimal solution often occurs at a corner, giving zeros.

```
    L1 Ball (Diamond)              L2 Ball (Circle)

         │                              │
        ╱│╲                           ╭─┼─╮
       ╱ │ ╲                         │  │  │
    ──●──┼──●──                    ──●──┼──●──
       ╲ │ ╱                         │  │  │
        ╲│╱                           ╰─┼─╯
         │                              │

    Corners → sparse              Smooth → not sparse
    solutions                     solutions
```

**Subgradient Optimality:** At optimum $w^*$:
$$0 \in X^T(Xw^* - y) + \lambda \cdot \partial\|w^*\|_1$$

where $\partial\|w\|_1$ is the subdifferential of the L1 norm.

### 8.3 Support Vector Machines

**Hard-margin SVM (Primal):**
$$\minimize_{w, b} \quad \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^Tx_i + b) \geq 1$$

**Dual derivation:**

*Step 1:* Lagrangian:
$$L(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^n \alpha_i[y_i(w^Tx_i + b) - 1]$$

*Step 2:* Minimize over $w$ and $b$:
$$\nabla_w L = w - \sum_i \alpha_i y_i x_i = 0 \implies w = \sum_i \alpha_i y_i x_i$$
$$\nabla_b L = -\sum_i \alpha_i y_i = 0 \implies \sum_i \alpha_i y_i = 0$$

*Step 3:* Substitute back:
$$g(\alpha) = \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j (x_i^Tx_j)$$

**Dual Problem:**
$$\maximize_\alpha \quad \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j (x_i^Tx_j)$$
$$\text{s.t.} \quad \alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0$$

**Kernel Trick:** The dual only involves inner products $x_i^Tx_j$, which can be replaced by $k(x_i, x_j)$ for any kernel function.

### 8.4 Logistic Regression

**Problem:**
$$\minimize_w \quad \sum_{i=1}^n \log(1 + e^{-y_i w^T x_i}) + \frac{\lambda}{2}\|w\|^2$$

**Convexity:** Strictly convex (shown in Example 2.2 + L2 term)

**No closed-form solution** — solved iteratively via gradient descent or Newton's method.

**Gradient:**
$$\nabla_w \mathcal{L} = -\sum_{i=1}^n \frac{y_i x_i}{1 + e^{y_i w^T x_i}} + \lambda w = \sum_{i=1}^n (\sigma(w^Tx_i) - \mathbf{1}[y_i=1])x_i + \lambda w$$

### 8.5 Summary: Convexity in ML

| Model | Objective | Convex? | Closed Form? | Algorithm |
|-------|-----------|---------|--------------|-----------|
| Linear Regression | MSE | Yes | Yes | Normal equations |
| Ridge Regression | MSE + L2 | Yes (strict) | Yes | Normal equations |
| Lasso | MSE + L1 | Yes | No | Coordinate descent, ISTA |
| Logistic Regression | Log loss | Yes | No | Newton, SGD |
| SVM | Hinge + L2 | Yes | No* | SMO, SGD |
| Neural Networks | Various | **No** | No | SGD, Adam |

*SVM has closed form via dual but requires solving QP.

---

## 9. Convex Relaxations for Non-Convex Problems

### 9.1 When Problems Aren't Convex

Many important problems are inherently non-convex:
- **Matrix factorization:** $\min_{U,V} \|X - UV^T\|^2$
- **Sparse recovery with L0:** $\min_w \|y - Xw\|^2$ s.t. $\|w\|_0 \leq k$
- **Binary optimization:** $\min_x f(x)$ s.t. $x \in \{0, 1\}^n$
- **Neural networks:** Composition of nonlinear layers

The strategy of **convex relaxation** replaces a hard non-convex problem with a tractable convex approximation.

### 9.2 L1 Relaxation of L0 (Basis Pursuit)

**Original (NP-hard):**
$$\minimize_w \quad \|w\|_0 \quad \text{s.t.} \quad Xw = y$$

**Convex relaxation:**
$$\minimize_w \quad \|w\|_1 \quad \text{s.t.} \quad Xw = y$$

```
    L0 "Ball" (Non-convex)        L1 Ball (Convex relaxation)

         │                              │
         ●                             ╱│╲
         │                            ╱ │ ╲
    ─────●─────                   ──●──┼──●──
         │                            ╲ │ ╱
         ●                             ╲│╱
         │                              │

    Only axes                     Diamond (tightest convex
    (sparse points)               approximation)
```

**When does it work?** Under conditions like the Restricted Isometry Property (RIP), the L1 solution equals the L0 solution. This is the foundation of compressed sensing.

### 9.3 Semidefinite Relaxation

**Original (non-convex):**
$$\maximize_x \quad x^T A x \quad \text{s.t.} \quad x \in \{-1, +1\}^n$$

**SDP relaxation:** Replace $xx^T$ with matrix variable $X$:
$$\maximize_X \quad \text{tr}(AX) \quad \text{s.t.} \quad X \succeq 0, \; X_{ii} = 1$$

The constraint $X = xx^T$ for $x \in \{-1,1\}^n$ is relaxed to $X \succeq 0$.

**Application:** MAX-CUT problem, approximation algorithms.

### 9.4 Nuclear Norm Relaxation (Matrix Completion)

**Original:**
$$\minimize_M \quad \text{rank}(M) \quad \text{s.t.} \quad M_{i,j} = X_{i,j} \text{ for observed entries}$$

**Convex relaxation:**
$$\minimize_M \quad \|M\|_* \quad \text{s.t.} \quad M_{i,j} = X_{i,j} \text{ for observed entries}$$

where $\|M\|_* = \sum_i \sigma_i(M)$ is the nuclear norm (sum of singular values).

**Application:** Netflix prize, recommender systems.

---

## 10. Hierarchy of Convex Problems

### 10.1 Linear Programming (LP)

$$\minimize_x \quad c^Tx \quad \text{s.t.} \quad Ax \leq b, \quad Cx = d$$

- Linear objective, linear constraints
- Complexity: polynomial time (simplex practical, interior-point polynomial)
- Applications: resource allocation, network flow, diet problems

### 10.2 Quadratic Programming (QP)

$$\minimize_x \quad \frac{1}{2}x^TPx + q^Tx \quad \text{s.t.} \quad Ax \leq b$$

- Quadratic objective ($P \succeq 0$), linear constraints
- Reduces to LP when $P = 0$
- Applications: portfolio optimization, SVM, MPC

### 10.3 Second-Order Cone Programming (SOCP)

$$\minimize_x \quad c^Tx \quad \text{s.t.} \quad \|A_ix + b_i\|_2 \leq c_i^Tx + d_i$$

- Second-order cone constraints
- More general than QP (QP can be reformulated as SOCP)
- Applications: robust optimization, signal processing

### 10.4 Semidefinite Programming (SDP)

$$\minimize_X \quad \text{tr}(CX) \quad \text{s.t.} \quad \text{tr}(A_iX) = b_i, \quad X \succeq 0$$

- Optimization over positive semidefinite matrices
- Most general tractable convex problem
- Applications: MAX-CUT relaxation, control, sensor localization

```
    Hierarchy of Convex Problems

                    SDP
                   ╱   ╲
                  ╱     ╲
               SOCP    (other cone programs)
                │
                │
               QP
                │
                │
               LP

    Each level includes all problems below it
```

---

## 11. Verifying Convexity: A Systematic Approach

### 11.1 Decision Tree for Convexity Verification

```
    Is f convex?
    │
    ├── Can you compute the Hessian ∇²f(x)?
    │   │
    │   ├── YES: Is ∇²f(x) ≽ 0 for all x in domain?
    │   │   │
    │   │   ├── YES → f is convex ✓
    │   │   │   └── Is ∇²f(x) ≻ 0? → f is strictly convex ✓
    │   │   │
    │   │   └── NO → f is NOT convex ✗
    │   │
    │   └── NO (non-differentiable, complex)
    │       │
    │       ├── Is f built from known convex functions?
    │       │   │
    │       │   ├── Sum of convex? → Convex ✓
    │       │   ├── Max of convex? → Convex ✓
    │       │   ├── f(Ax+b) with f convex? → Convex ✓
    │       │   └── Composition rules apply? → Check rules
    │       │
    │       └── Try restriction to a line:
    │           g(t) = f(x + tv) convex for all x, v?
    │
    └── Domain check: Is dom(f) convex?
        └── If NO → f is NOT convex (by definition)
```

### 11.2 Practical Algorithm

1. **Check domain convexity** first
2. **Compute Hessian** if differentiable
3. **Test eigenvalues** or use matrix tests ($2 \times 2$: check $a \geq 0$, $c \geq 0$, $ac - b^2 \geq 0$)
4. **Use composition rules** for complex expressions
5. **Verify numerically** by sampling random pairs and checking Jensen

### 11.3 Worked Examples

**Example 11.1:** Is $f(x, y) = e^x + e^y + e^{-(x+y)}$ convex?

**Solution:**
$$\nabla f = \begin{pmatrix} e^x - e^{-(x+y)} \\ e^y - e^{-(x+y)} \end{pmatrix}$$

$$\nabla^2 f = \begin{pmatrix} e^x + e^{-(x+y)} & e^{-(x+y)} \\ e^{-(x+y)} & e^y + e^{-(x+y)} \end{pmatrix}$$

Let $a = e^x > 0$, $b = e^y > 0$, $c = e^{-(x+y)} > 0$.

$$H = \begin{pmatrix} a + c & c \\ c & b + c \end{pmatrix}$$

Check positive semidefiniteness:
- $H_{11} = a + c > 0$ ✓
- $\det(H) = (a+c)(b+c) - c^2 = ab + ac + bc + c^2 - c^2 = ab + ac + bc > 0$ ✓

Therefore $f$ is strictly convex. $\square$

**Example 11.2:** Is $f(x) = \|Ax - b\|_1$ convex?

**Solution:** Use composition:
- $g(z) = \|z\|_1$ is convex (it's a norm)
- $z = Ax - b$ is affine in $x$
- Affine composition of convex function → Convex ✓

Alternative: $\|Ax-b\|_1 = \sum_i |a_i^Tx - b_i|$. Each term $|a_i^Tx - b_i| = \max\{a_i^Tx - b_i, -(a_i^Tx - b_i)\}$ is the max of two affine functions, hence convex. Sum of convex is convex. ✓ $\square$

---

## 12. Summary and Key Takeaways

### 12.1 Core Concepts Summary

| Concept | Definition | Key Property |
|---------|------------|--------------|
| Convex Set | Contains all line segments | Intersection of halfspaces |
| Convex Function | Chord above graph | $f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$ |
| First-Order Condition | Above tangent planes | $f(y) \geq f(x) + \nabla f(x)^T(y-x)$ |
| Second-Order Condition | Non-negative curvature | $\nabla^2 f \succeq 0$ |
| Convex Problem | Convex $f$, convex constraints | Local = Global |
| Lagrangian | Soft constraint formulation | $L = f + \sum \lambda_i g_i + \sum \nu_j h_j$ |
| Dual Function | $\inf_x L(x, \lambda, \nu)$ | Always concave |
| Weak Duality | $d^* \leq p^*$ | Always holds |
| Strong Duality | $d^* = p^*$ | Under Slater's condition |
| KKT Conditions | Primal/Dual feas., Comp. slack., Stat. | Necessary & sufficient |

### 12.2 What Makes Convex Optimization Special

1. **Tractability**: Convex problems can be solved efficiently (polynomial time)
2. **Global Optimality**: Any local minimum is global
3. **Duality**: Provides bounds and alternative formulations
4. **Theory**: Rich mathematical structure for analysis
5. **Algorithms**: Well-understood convergence guarantees

### 12.3 Connections to Machine Learning

| ML Concept | Convex Optimization Connection |
|------------|-------------------------------|
| Loss minimization | Convex objective |
| Regularization | Adding convex penalty preserves convexity |
| Constraints | Convex feasible set |
| Support vectors | Complementary slackness (only active constraints) |
| Kernel methods | Dual formulation |
| Sparsity (Lasso) | L1 as relaxation of L0 |
| Matrix completion | Nuclear norm relaxation |

### 12.4 When Convexity Fails

For non-convex problems (neural networks, matrix factorization):
- Multiple local minima exist
- No polynomial-time algorithm guaranteed
- Empirically, SGD often finds good solutions
- Research: understanding loss landscapes, saddle points, mode connectivity

---

## Exercises

### Foundational Exercises

1. **Convex Set Proof**: Prove that the set $S = \{(x, y) : x^2 \leq y\}$ (region above a parabola) is convex. Use the definition directly.

2. **Hessian Verification**: Show that $f(x, y) = x^2 + 4xy + 5y^2 - 2x + 6y$ is convex by computing and analyzing its Hessian. Is it strictly convex?

3. **First-Order Condition**: Using the first-order characterization of convexity, prove that $f(x) = e^x$ is convex without computing the second derivative.

4. **Non-Convexity**: Show that $f(x, y) = xy$ is NOT convex by finding points that violate Jensen's inequality.

### Intermediate Exercises

5. **Composition**: Determine whether $f(x) = \log(\sum_{i=1}^n e^{a_i^T x + b_i})$ is convex, where $a_i \in \mathbb{R}^d$ and $b_i \in \mathbb{R}$.

6. **Duality**: For the problem $\min_x x^2$ subject to $|x| \leq 1$:
   a) Formulate the Lagrangian
   b) Derive the dual function
   c) Solve both primal and dual
   d) Verify strong duality

7. **KKT Application**: Use KKT conditions to solve:
   $$\minimize_{x, y} \quad x^2 + y^2 \quad \text{s.t.} \quad x + y \geq 4, \; x \geq 0$$

8. **ML Connection**: Prove that the softmax cross-entropy loss $\ell(z, y) = -z_y + \log\sum_j e^{z_j}$ is convex in $z$ for any fixed label $y$.

### Advanced Exercises

9. **Ridge Regression from First Principles**: Starting from the KKT conditions, derive the closed-form solution for Ridge regression with non-negativity constraints: $\min_w \|Xw - y\|^2 + \lambda\|w\|^2$ s.t. $w \geq 0$.

10. **SVM Dual Derivation**: Given the soft-margin SVM primal:
    $$\minimize_{w, b, \xi} \quad \frac{1}{2}\|w\|^2 + C\sum_i \xi_i$$
    $$\text{s.t.} \quad y_i(w^Tx_i + b) \geq 1 - \xi_i, \; \xi_i \geq 0$$
    
    Derive the dual problem step by step. Show that the dual involves only inner products $x_i^T x_j$.

11. **Convex Relaxation**: The boolean least squares problem $\min_{x \in \{-1, 1\}^n} \|Ax - b\|^2$ is NP-hard. Propose a convex relaxation and explain why it might give a good approximate solution.

12. **Loss Landscape Analysis**: For $f(w_1, w_2) = (w_1 w_2 - 1)^2$, show that this function is NOT convex despite having a unique global minimum. Classify all critical points.

---

## References

### Primary References

1. **Boyd, S. & Vandenberghe, L.** (2004). *Convex Optimization*. Cambridge University Press.
   - The definitive textbook. Available free online at [stanford.edu/~boyd/cvxbook](https://stanford.edu/~boyd/cvxbook)
   
2. **Boyd, S.** *Convex Optimization I & II*. Stanford EE364a/b.
   - Lecture videos and materials freely available

### Additional Resources

3. **Nesterov, Y.** (2004). *Introductory Lectures on Convex Optimization*. Springer.
   - Theoretical perspective, convergence analysis

4. **Nocedal, J. & Wright, S.J.** (2006). *Numerical Optimization*. Springer.
   - Algorithms and implementation

5. **Ben-Tal, A. & Nemirovski, A.** (2001). *Lectures on Modern Convex Optimization*. SIAM.
   - Conic programming, robust optimization

### Machine Learning Connections

6. **Shalev-Shwartz, S. & Ben-David, S.** (2014). *Understanding Machine Learning*. Cambridge.
   - Convexity in learning theory

7. **Hastie, T., Tibshirani, R. & Friedman, J.** (2009). *Elements of Statistical Learning*. Springer.
   - Regularization, sparsity

8. **Bottou, L., Curtis, F.E. & Nocedal, J.** (2018). "Optimization Methods for Large-Scale Machine Learning". *SIAM Review*.
   - Modern perspective on optimization in ML

---

*This chapter provides the foundation for understanding optimization in machine learning. The concepts introduced here—convexity, duality, and KKT conditions—will appear repeatedly as we study gradient descent, constrained optimization, and advanced optimization techniques.*
