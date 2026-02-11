<div align="center">

# 🧮 Mathematics for AI/ML Mastery

### The Complete Mathematical Foundation for Artificial Intelligence & Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.20+-green.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Sections](https://img.shields.io/badge/sections-14-blueviolet)
![Topics](https://img.shields.io/badge/topics-72-orange)
![Notebooks](https://img.shields.io/badge/notebooks-144-red)

*From number systems to neural network theory — everything you need, in one place.*

> *"The only way to learn mathematics is to do mathematics."* — Paul Halmos

</div>

---

A comprehensive, structured guide to mastering the mathematics required for AI/ML, combining the best content from DeepLearning.AI, Khan Academy, MIT OCW, Stanford CS229, 3Blue1Brown, and StatQuest. Every topic comes with **textbook-quality theory**, **interactive Jupyter notebooks** with working code, and **hands-on exercises** to solidify understanding.

## 📋 Table of Contents

- [Why This Repository](#-why-this-repository)
- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Learning Roadmap](#-learning-roadmap)
- [Detailed Curriculum](#-detailed-curriculum)
- [How to Use This Repository](#-how-to-use-this-repository)
- [Prerequisites](#-prerequisites)
- [Quick Reference Docs](#-quick-reference-docs)
- [Progress Tracker](#-progress-tracker)
- [Resources](#-resources)
- [Contributing](#-contributing)
- [License](#-license)

---

## 💡 Why This Repository

Most ML math resources fall into two camps: abstract textbooks that never connect to code, or shallow tutorials that skip the "why." This repository bridges that gap.

| Problem | Our Solution |
|---------|-------------|
| Math textbooks feel disconnected from ML | Every formula is tied to a real ML application |
| Code-only tutorials skip the theory | Every notebook begins with rigorous derivations |
| Hard to know what math to learn first | A clear 14-section roadmap from foundations → mastery |
| No way to test understanding | 72 exercise notebooks with progressive difficulty |
| Scattered across many sites | One cohesive, self-contained repository |

### What Makes This Different

- **🔗 Theory ↔ Code Loop** — Each topic has a README explaining the math, then notebooks where you implement it in NumPy/SciPy
- **📖 Textbook-Quality Writing** — READMEs read like book chapters with intuition, derivations, "Why This Matters for ML" sections, insight callouts, and key takeaways
- **🎯 ML-First Perspective** — We don't teach math for its own sake; every topic explains *exactly* how and where it appears in machine learning
- **📓 144 Interactive Notebooks** — Run code, tweak parameters, see visualizations, and build intuition hands-on
- **🧩 Modular Design** — Jump to any section; each topic is self-contained with cross-references where needed

---

## 🎯 Overview

**14 sections · 72 topics · 144 notebooks · 72 README chapters**

Each of the 72 topics follows a consistent **3-File System**:

| File | Purpose |
|------|---------|
| `README.md` | Book-style chapter: theory, derivations, intuition, diagrams, ML connections |
| `theory.ipynb` | Jupyter notebook with theory explanations, working code, and visualizations |
| `exercises.ipynb` | Practice problems from basic to advanced, with full solutions |

### Who Is This For?

| Audience | What You'll Get |
|----------|----------------|
| 🎓 **Students** preparing for ML/AI careers | A structured curriculum that builds from foundations to mastery |
| 💼 **ML Engineers** wanting deeper understanding | The "why" behind every algorithm, not just the API calls |
| 📊 **Data Scientists** strengthening foundations | Fill gaps in linear algebra, probability, optimization theory |
| 🎤 **Interview Candidates** | 26+ solved interview questions, derivations you'll be asked |
| 🔄 **Career Switchers** transitioning to AI/ML | A clear learning path with estimated time commitments |
| 📝 **Researchers** reading ML papers | Notation guide + every formula you'll encounter, decoded |

### After Completing This Repository, You Will Be Able To:

✅ Read ML papers and understand every equation  
✅ Derive backpropagation, attention, and VAE ELBO from scratch  
✅ Implement optimization algorithms (SGD, Adam, L-BFGS) from the math  
✅ Explain probabilistic models — GMMs, HMMs, GPs — mathematically  
✅ Understand kernel methods, spectral graph theory, and functional analysis  
✅ Reason about numerical stability, convergence, and generalization  
✅ Answer ML math interview questions with confidence and depth

---

## 🏗️ Repository Structure

```
math_for_ai/
│
├── 📐 01-Mathematical-Foundations/      ← Start here
│   ├── 01-Number-Systems/
│   ├── 02-Sets-and-Logic/
│   ├── 03-Functions-and-Mappings/
│   ├── 04-Summation-and-Product-Notation/
│   └── 05-Proof-Techniques/
│
├── 📊 02-Linear-Algebra-Basics/         ← Essential for all ML
│   ├── 01-Vectors-and-Spaces/
│   ├── 02-Matrix-Operations/
│   ├── 03-Systems-of-Equations/
│   ├── 04-Determinants/
│   ├── 05-Matrix-Rank/
│   └── 06-Vector-Spaces-Subspaces/
│
├── 🔬 03-Advanced-Linear-Algebra/       ← PCA, SVD, spectral methods
│   ├── 01-Eigenvalues-and-Eigenvectors/
│   ├── 02-Singular-Value-Decomposition/
│   ├── 03-Principal-Component-Analysis/
│   ├── 04-Linear-Transformations/
│   ├── 05-Orthogonality-and-Orthonormality/
│   ├── 06-Matrix-Norms/
│   ├── 07-Positive-Definite-Matrices/
│   └── 08-Matrix-Decompositions/
│
├── 📈 04-Calculus-Fundamentals/          ← Derivatives & integration
│   ├── 01-Limits-and-Continuity/
│   ├── 02-Derivatives-and-Differentiation/
│   ├── 03-Integration/
│   └── 04-Series-and-Sequences/
│
├── 🌊 05-Multivariate-Calculus/          ← Gradients & backprop
│   ├── 01-Partial-Derivatives-and-Gradients/
│   ├── 02-Jacobians-and-Hessians/
│   ├── 03-Chain-Rule-and-Backpropagation/
│   └── 04-Optimization-Theory/
│
├── 🎲 06-Probability-Theory/             ← Uncertainty & inference
│   ├── 01-Introduction-and-Random-Variables/
│   ├── 02-Common-Distributions/
│   ├── 03-Joint-Distributions/
│   └── 04-Expectation-and-Moments/
│
├── 📉 07-Statistics/                     ← Estimation & testing
│   ├── 01-Descriptive-Statistics/
│   ├── 02-Estimation-Theory/
│   ├── 03-Hypothesis-Testing/
│   └── 04-Bayesian-Inference/
│
├── ⚡ 08-Optimization/                   ← Training algorithms
│   ├── 01-Convex-Optimization/
│   ├── 02-Gradient-Descent/
│   ├── 03-Second-Order-Methods/
│   ├── 04-Constrained-Optimization/
│   ├── 05-Stochastic-Optimization/
│   ├── 06-Optimization-Landscape/
│   ├── 07-Adaptive-Learning-Rate/
│   ├── 08-Regularization-Methods/
│   └── 09-Hyperparameter-Optimization/
│
├── 📡 09-Information-Theory/             ← Loss functions & metrics
│   ├── 01-Entropy/
│   ├── 02-KL-Divergence/
│   ├── 03-Mutual-Information/
│   └── 04-Cross-Entropy/
│
├── 🔢 10-Numerical-Methods/              ← Stability & precision
│   ├── 01-Floating-Point-Arithmetic/
│   ├── 02-Numerical-Linear-Algebra/
│   ├── 03-Numerical-Optimization/
│   ├── 04-Interpolation-and-Approximation/
│   └── 05-Numerical-Integration/
│
├── 🕸️ 11-Graph-Theory/                   ← GNNs & networks
│   ├── 01-Graph-Basics/
│   ├── 02-Graph-Representations/
│   ├── 03-Graph-Algorithms/
│   ├── 04-Spectral-Graph-Theory/
│   └── 05-Graph-Neural-Networks/
│
├── 🏛️ 12-Functional-Analysis/            ← Kernels & RKHS
│   ├── 01-Vector-Spaces/
│   ├── 02-Normed-Spaces/
│   ├── 03-Hilbert-Spaces/
│   └── 04-Kernel-Methods/
│
├── 🤖 13-ML-Specific-Math/               ← Applied ML concepts
│   ├── 01-Loss-Functions/
│   ├── 02-Regularization-Theory/
│   ├── 03-Kernel-Methods/
│   ├── 04-Information-Geometry/
│   └── 05-Sampling-Methods/
│
├── 🧠 14-Math-for-Specific-Models/       ← Model derivations
│   ├── 01-Linear-Models/
│   ├── 02-Neural-Networks/
│   ├── 03-Probabilistic-Models/
│   ├── 04-Sequence-Models/
│   └── 05-Generative-Models/
│
├── 📚 docs/                               ← Reference materials
│   ├── CHEATSHEET.md
│   ├── NOTATION_GUIDE.md
│   ├── ML_MATH_MAP.md
│   ├── INTERVIEW_PREP.md
│   └── VISUALIZATION_GUIDE.md
│
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md                              ← You are here
```

---

## 🗺️ Learning Roadmap

```
                                ┌───────────────────────────────────────────────┐
                                │        MATHEMATICS FOR AI/ML MASTERY          │
                                │          14 Sections · 72 Topics              │
                                └───────────────────────────────────────────────┘
                                                      │
                  ┌───────────────────────────────────┼────────────────────────────────────────┐
                  │                                   │                                        │
                  ▼                                   ▼                                        ▼
       ┌─────────────────────┐             ┌─────────────────────┐              ┌──────────────────────────┐
       │  01 FOUNDATIONS     │             │ 02 LINEAR ALGEBRA   │              │  04 CALCULUS             │
       │  • Number Systems   │             │ • Vectors & Spaces  │              │  • Limits & Continuity   │
       │  • Sets & Logic     │────────────▶│ • Matrix Operations │─────────────▶│  • Derivatives           │
       │  • Functions        │             │ • Systems of Eqns   │              │  • Integration           │
       │  • Summation/Product│             │ • Determinants      │              │  • Series & Sequences    │
       │  • Proof Techniques │             │ • Matrix Rank       │              └──────────────────────────┘
       └─────────────────────┘             │ • Vector Spaces     │                             │
                                           └─────────────────────┘                             │
                                                      │                                        │
                                                      ▼                                        ▼
                                ┌──────────────────────────────┐       ┌──────────────────────────────┐
                                │  03 ADVANCED LINEAR ALGEBRA  │       │  05 MULTIVARIATE CALCULUS    │
                                │  • Eigenvalues & Eigenvectors│       │  • Partial Derivs & Gradients│
                                │  • SVD                       │       │  • Jacobians & Hessians      │
                                │  • PCA                       │       │  • Chain Rule & Backprop     │
                                │  • Linear Transformations    │       │  • Optimization Theory       │
                                │  • Orthogonality             │       └──────────────────────────────┘
                                │  • Matrix Norms              │                      │
                                │  • Positive Definite Matrices│                      │
                                │  • Matrix Decompositions     │                      │
                                └──────────────────────────────┘                      │
                                                      │                               │
                  ┌───────────────────────────────────┼───────────────────────────────┘
                  │                                   │
                  ▼                                   ▼
       ┌─────────────────────────┐         ┌─────────────────────────┐
       │  06 PROBABILITY THEORY  │         │  07 STATISTICS          │
       │  • Random Variables     │         │  • Descriptive Stats    │
       │  • Common Distributions │────────▶│  • Estimation Theory    │
       │  • Joint Distributions  │         │  • Hypothesis Testing   │
       │  • Expectation & Moments│         │  • Bayesian Inference   │
       └─────────────────────────┘         └─────────────────────────┘
                  │                                   │
                  └──────────────────┬────────────────┘
                                     │
                  ┌──────────────────┼──────────────────────────────────────────┐
                  │                  │                                          │
                  ▼                  ▼                                          ▼
  ┌───────────────────────────┐ ┌────────────────────┐        ┌────────────────────────────────┐
  │  08 OPTIMIZATION          │ │ 09 INFO THEORY     │        │  10 NUMERICAL METHODS          │
  │  • Convex Optimization    │ │ • Entropy          │        │  • Floating-Point Arithmetic   │
  │  • Gradient Descent       │ │ • KL Divergence    │        │  • Numerical Linear Algebra    │
  │  • Second-Order Methods   │ │ • Mutual Info      │        │  • Numerical Optimization      │
  │  • Constrained Opt        │ │ • Cross-Entropy    │        │  • Interpolation & Approx      │
  │  • Stochastic Opt         │ └────────────────────┘        │  • Numerical Integration       │
  │  • Optimization Landscape │            │                  └────────────────────────────────┘
  │  • Adaptive Learning Rate │            │                               │
  │  • Regularization Methods │            │                               │
  │  • Hyperparameter Opt     │            │                               │
  └───────────────────────────┘            │                               │
                  │                        │                               │
                  └────────────────────────┼───────────────────────────────┘
                                           │
                  ┌────────────────────────┼───────────────────────────────┐
                  │                        │                               │
                  ▼                        ▼                               ▼
  ┌────────────────────────┐ ┌──────────────────────┐   ┌──────────────────────────┐
  │  11 GRAPH THEORY       │ │ 12 FUNCTIONAL        │   │  13 ML-SPECIFIC MATH     │
  │  • Graph Basics        │ │    ANALYSIS           │   │  • Loss Functions        │
  │  • Representations     │ │ • Vector Spaces       │   │  • Regularization Theory │
  │  • Graph Algorithms    │ │ • Normed Spaces       │   │  • Kernel Methods        │
  │  • Spectral Graph Thy  │ │ • Hilbert Spaces      │   │  • Information Geometry  │
  │  • Graph Neural Nets   │ │ • Kernel Methods      │   │  • Sampling Methods      │
  └────────────────────────┘ └──────────────────────┘   └──────────────────────────┘
                  │                        │                               │
                  └────────────────────────┴───────────────────────────────┘
                                           │
                                           ▼
                              ┌──────────────────────────┐
                              │  14 MODEL MATH           │
                              │  • Linear Models         │
                              │  • Neural Networks       │
                              │  • Probabilistic Models  │
                              │  • Sequence Models       │
                              │  • Generative Models     │
                              │                          │
                              │       🎓 MASTERY!        │
                              └──────────────────────────┘
```


---

## 📖 Detailed Curriculum

### Section 01 — Mathematical Foundations

> *Build the mathematical language for everything that follows.*

| Topic | What You'll Learn | ML Connection |
|-------|------------------|---------------|
| Number Systems | Real, complex, floating-point representation | Why `0.1 + 0.2 ≠ 0.3` matters in training |
| Sets and Logic | Set operations, propositional logic, quantifiers | Boolean operations in decision trees, data filtering |
| Functions and Mappings | Domain, range, composition, invertibility | Activation functions, loss functions, feature maps |
| Summation & Product | Sigma/Pi notation, index manipulation | Loss computation, likelihood functions |
| Proof Techniques | Induction, contradiction, direct proof | Understanding algorithm correctness proofs |

### Section 02 — Linear Algebra Basics

> *The single most important math for ML. Master this thoroughly.*

| Topic | What You'll Learn | ML Connection |
|-------|------------------|---------------|
| Vectors and Spaces | Vector operations, spans, linear independence | Feature vectors, word embeddings |
| Matrix Operations | Multiplication, transpose, inverse | Weight matrices, forward pass |
| Systems of Equations | Gaussian elimination, solution spaces | Least squares, linear regression |
| Determinants | Volume scaling, singularity detection | Matrix invertibility checks |
| Matrix Rank | Row/column rank, rank-nullity theorem | Dimensionality, data redundancy |
| Vector Spaces & Subspaces | Basis, dimension, null/column space | Latent spaces, kernel of transformations |

### Section 03 — Advanced Linear Algebra

> *Eigenvalues to SVD — the power tools of ML mathematics.*

| Topic | What You'll Learn | ML Connection |
|-------|------------------|---------------|
| Eigenvalues & Eigenvectors | Characteristic polynomial, diagonalization | PCA, spectral clustering, PageRank |
| SVD | Singular value decomposition, low-rank approximation | Recommender systems, image compression |
| PCA | Variance maximization, covariance eigen-analysis | Dimensionality reduction, feature extraction |
| Linear Transformations | Matrix representation, change of basis | Neural network layers, data augmentation |
| Orthogonality | Gram-Schmidt, QR decomposition, projections | Least squares, orthogonal regularization |
| Matrix Norms | Frobenius, spectral, nuclear norms | Regularization, stability analysis |
| Positive Definite Matrices | Cholesky, definiteness tests | Covariance matrices, kernel matrices |
| Matrix Decompositions | LU, QR, Cholesky, Schur | Efficient solving, numerical stability |

### Section 04 — Calculus Fundamentals

> *Understand how neural networks learn through calculus.*

| Topic | What You'll Learn | ML Connection |
|-------|------------------|---------------|
| Limits and Continuity | ε-δ definition, L'Hôpital's rule | Convergence of training, smooth activations |
| Derivatives | Differentiation rules, higher-order derivatives | Gradient computation, learning rate intuition |
| Integration | Definite/indefinite, techniques of integration | Expected values, probability densities |
| Series and Sequences | Taylor series, convergence tests | Function approximation, series expansions |

### Section 05 — Multivariate Calculus

> *Gradients, Jacobians, and the chain rule that makes deep learning work.*

| Topic | What You'll Learn | ML Connection |
|-------|------------------|---------------|
| Partial Derivatives & Gradients | Gradient vector, directional derivatives | Gradient descent direction |
| Jacobians and Hessians | Matrix of derivatives, second-order information | Sensitivity analysis, Newton's method |
| Chain Rule & Backpropagation | Multivariate chain rule, computational graphs | The heart of neural network training |
| Optimization Theory | Critical points, convexity, Lagrange multipliers | Finding minima of loss functions |

### Section 06 — Probability Theory

> *Reason about uncertainty — the foundation of statistical ML.*

| Topic | What You'll Learn | ML Connection |
|-------|------------------|---------------|
| Random Variables | Discrete/continuous, PMF, PDF, CDF | Model outputs, data distributions |
| Common Distributions | Gaussian, Bernoulli, Poisson, exponential family | Prior/posterior selection, noise modeling |
| Joint Distributions | Marginal, conditional, independence, Bayes' theorem | Bayesian inference, graphical models |
| Expectation and Moments | Mean, variance, covariance, MGF | Loss expectations, variance reduction |

### Section 07 — Statistics

> *From data to decisions — estimation, testing, and Bayesian reasoning.*

| Topic | What You'll Learn | ML Connection |
|-------|------------------|---------------|
| Descriptive Statistics | Central tendency, dispersion, correlation | EDA, feature engineering |
| Estimation Theory | MLE, MAP, bias-variance, confidence intervals | Model parameter estimation |
| Hypothesis Testing | t-test, p-values, multiple comparisons | A/B testing, model comparison |
| Bayesian Inference | Priors, posteriors, conjugacy, MCMC | Bayesian neural networks, uncertainty |

### Section 08 — Optimization

> *The largest section — because training IS optimization.*

| Topic | What You'll Learn | ML Connection |
|-------|------------------|---------------|
| Convex Optimization | Convex sets/functions, duality, KKT | SVM formulation, theoretical guarantees |
| Gradient Descent | Batch, mini-batch, convergence analysis | Core training algorithm |
| Second-Order Methods | Newton's method, BFGS, L-BFGS | Faster convergence for smaller models |
| Constrained Optimization | Lagrangian, KKT conditions, duality | SVMs, constrained learning |
| Stochastic Optimization | SGD, variance reduction, convergence | Large-scale training |
| Optimization Landscape | Saddle points, local minima, loss surfaces | Understanding training dynamics |
| Adaptive Learning Rate | Adam, RMSProp, AdaGrad, learning rate scheduling | Modern deep learning optimizers |
| Regularization Methods | L1, L2, dropout, early stopping | Preventing overfitting |
| Hyperparameter Optimization | Grid search, Bayesian optimization, Hyperband | Tuning model performance |

### Section 09 — Information Theory

> *The mathematical foundation of loss functions and model comparison.*

| Topic | What You'll Learn | ML Connection |
|-------|------------------|---------------|
| Entropy | Shannon entropy, differential entropy | Measuring uncertainty, decision trees |
| KL Divergence | Asymmetry, forward vs reverse KL | VAE loss, knowledge distillation |
| Mutual Information | Dependence measurement, data processing inequality | Feature selection, representation learning |
| Cross-Entropy | Connection to KL, log-loss | The most common classification loss |

### Section 10 — Numerical Methods

> *Why math on paper differs from math on silicon.*

| Topic | What You'll Learn | ML Connection |
|-------|------------------|---------------|
| Floating-Point Arithmetic | IEEE 754, precision, catastrophic cancellation | Mixed precision training (FP16/BF16) |
| Numerical Linear Algebra | Condition numbers, iterative solvers | Stable implementation of ML algorithms |
| Numerical Optimization | Line search, trust regions, convergence rates | Practical training considerations |
| Interpolation & Approximation | Polynomial, spline, Chebyshev | Data augmentation, function approximation |
| Numerical Integration | Quadrature, Monte Carlo integration | Expected value computation, Bayesian methods |

### Section 11 — Graph Theory

> *The math behind social networks, molecules, and knowledge graphs.*

| Topic | What You'll Learn | ML Connection |
|-------|------------------|---------------|
| Graph Basics | Vertices, edges, degree, connectivity | Data structure for relational data |
| Graph Representations | Adjacency matrix, Laplacian, incidence | Input format for graph algorithms |
| Graph Algorithms | BFS, DFS, shortest path, minimum spanning tree | Pathfinding, clustering |
| Spectral Graph Theory | Laplacian eigenvalues, Cheeger inequality | Spectral clustering, graph partitioning |
| Graph Neural Networks | Message passing, GCN, GAT, pooling | Node/edge/graph classification |

### Section 12 — Functional Analysis

> *The rigorous theory behind kernel methods and infinite-dimensional ML.*

| Topic | What You'll Learn | ML Connection |
|-------|------------------|---------------|
| Vector Spaces | Axioms, function spaces, completeness | Theory of hypothesis spaces |
| Normed Spaces | Lp norms, Banach spaces, operator norms | Regularization theory |
| Hilbert Spaces | Inner products, Riesz representation, projections | RKHS foundation |
| Kernel Methods | Mercer's theorem, representer theorem, kernel trick | SVMs, Gaussian processes, kernel PCA |

### Section 13 — ML-Specific Math

> *Mathematical concepts purpose-built for machine learning.*

| Topic | What You'll Learn | ML Connection |
|-------|------------------|---------------|
| Loss Functions | Design principles, properties, custom losses | Choosing the right objective |
| Regularization Theory | Bias-variance, structural risk minimization | Generalization theory |
| Kernel Methods (Applied) | RBF, polynomial, string kernels | Practical kernel selection |
| Information Geometry | Fisher information, natural gradient | Advanced optimization on manifolds |
| Sampling Methods | MCMC, Metropolis-Hastings, HMC, variational inference | Bayesian inference, generative models |

### Section 14 — Math for Specific Models

> *Detailed mathematical derivations for each major ML model family.*

| Topic | What You'll Learn | ML Connection |
|-------|------------------|---------------|
| Linear Models | Normal equations, ridge/lasso derivation | Linear/logistic regression, GLMs |
| Neural Networks | Universal approximation, backprop calculus | MLPs, CNNs, initialization theory |
| Probabilistic Models | EM algorithm, GMMs, HMMs, Gaussian processes | Bayesian ML, uncertainty quantification |
| Sequence Models | RNN gradients, LSTM/GRU gates, attention math | NLP, time series, speech |
| Generative Models | VAE ELBO, GAN minimax, diffusion SDEs | Image generation, density estimation |

---

## 🚀 How to Use This Repository

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/math_for_ai.git
cd math_for_ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Recommended Study Approach

```
For each topic:

    ┌──────────────────────────────────┐
    │  1. 📖 Read the README.md        │ ← Understand the theory & intuition
    └──────────────┬───────────────────┘
                   ▼
    ┌──────────────────────────────────┐
    │  2. 📓 Run theory.ipynb          │ ← Theory + code in action, tweak parameters
    └──────────────┬───────────────────┘
                   ▼
    ┌──────────────────────────────────┐
    │  3. ✏️  Solve exercises.ipynb     │ ← Try problems WITHOUT looking at solutions
    └──────────────┬───────────────────┘
                   ▼
    ┌──────────────────────────────────┐
    │  4. ✅ Check solutions & review   │ ← Understand mistakes, fill gaps
    └──────────────┬───────────────────┘
                   ▼
    ┌──────────────────────────────────┐
    │  5. ➡️  Move to next topic        │ ← Only when comfortable
    └──────────────────────────────────┘
```

### Quick Start by Goal

| Your Goal | Start Here | Then |
|-----------|-----------|------|
| Understand deep learning math | 02 → 04 → 05 → 08 | 14-02 Neural Networks |
| Prepare for ML interviews | 02 → 06 → 08 → 09 | [Interview Prep](docs/INTERVIEW_PREP.md) |
| Read ML research papers | 01 → 03 → 05 → 09 | [Notation Guide](docs/NOTATION_GUIDE.md) |
| Implement models from scratch | 02 → 05 → 08 → 10 | 14 (all model math) |
| Work with graph data / GNNs | 02 → 11 → 12 | 11-05 Graph Neural Networks |
| Understand generative AI | 06 → 09 → 08 → 14-05 | VAE ELBO & GAN theory |

---

## � Prerequisites

Before starting, you should be comfortable with:

| Requirement | Level | Brush Up |
|-------------|-------|----------|
| **Basic Algebra** | Variables, equations, polynomials | [Khan Academy Algebra](https://www.khanacademy.org/math/algebra) |
| **High School Math** | Geometry, trigonometry basics | [Khan Academy Precalculus](https://www.khanacademy.org/math/precalculus) |
| **Python Basics** | Functions, loops, lists, classes | [Python Tutorial](https://docs.python.org/3/tutorial/) |
| **NumPy Basics** | Array operations (helpful, not required) | [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html) |

> **💡 Tip:** If you're unsure about prerequisites, start with **Section 01 (Mathematical Foundations)** — it's designed to bring you up to speed.

---

## 📋 Quick Reference Docs

Five companion documents provide searchable reference material:

| Document | Description | Best For |
|----------|-------------|----------|
| 📄 [Cheatsheet](docs/CHEATSHEET.md) | All key formulas on one page — linear algebra through functional analysis, plus NumPy & PyTorch reference | Quick formula lookup during coding |
| 📝 [Notation Guide](docs/NOTATION_GUIDE.md) | Every mathematical symbol decoded — sets, vectors, matrices, calculus, probability, info theory, graph theory | Reading ML papers, understanding notation |
| 🗺️ [ML Math Map](docs/ML_MATH_MAP.md) | Which math appears in which model/algorithm, with dependency diagrams and learning paths | Planning what to study for a specific model |
| 🎤 [Interview Prep](docs/INTERVIEW_PREP.md) | 26+ solved interview questions with derivations: linear algebra → deep learning → generative models, plus study plan | Preparing for ML engineer interviews |
| 📊 [Visualization Guide](docs/VISUALIZATION_GUIDE.md) | Ready-to-use matplotlib/plotly code for visualizing every concept: eigenvectors, gradients, distributions, attention maps | Making math intuitive through plots |

---

## ✅ Progress Tracker

Use this checklist to track your journey through the curriculum:

### Foundations & Core

- [ ] **01 Mathematical Foundations** — Number systems, sets, logic, functions, proofs
- [ ] **02 Linear Algebra Basics** — Vectors, matrices, systems of equations, rank
- [ ] **04 Calculus Fundamentals** — Limits, derivatives, integration, series
- [ ] **05 Multivariate Calculus** — Gradients, Jacobians, chain rule, backpropagation

### Intermediate

- [ ] **03 Advanced Linear Algebra** — Eigenvalues, SVD, PCA, matrix decompositions
- [ ] **06 Probability Theory** — Random variables, distributions, Bayes' theorem
- [ ] **07 Statistics** — Estimation, hypothesis testing, Bayesian inference

### Advanced

- [ ] **08 Optimization** — Gradient descent, Adam, constrained optimization, regularization
- [ ] **09 Information Theory** — Entropy, KL divergence, cross-entropy
- [ ] **10 Numerical Methods** — Floating point, numerical stability, iterative solvers

### Expert

- [ ] **11 Graph Theory** — Graph representations, algorithms, spectral theory, GNNs
- [ ] **12 Functional Analysis** — Normed spaces, Hilbert spaces, kernel methods
- [ ] **13 ML-Specific Math** — Loss functions, regularization theory, sampling methods
- [ ] **14 Math for Specific Models** — Neural networks, probabilistic models, generative models

---

## 🔗 Resources

### Video Courses

| Resource | Topics Covered | Level |
|----------|---------------|-------|
| 📺 [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Vectors, transformations, eigenvalues | Beginner |
| 📺 [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) | Derivatives, integrals, Taylor series | Beginner |
| 📺 [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | Backpropagation, gradient descent | Beginner |
| 📺 [MIT 18.06 — Linear Algebra (Gilbert Strang)](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/) | Complete linear algebra course | Intermediate |
| 📺 [Stanford CS229 — Machine Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) | ML theory and math | Intermediate |
| 📺 [StatQuest with Josh Starmer](https://www.youtube.com/c/joshstarmer) | Statistics, ML algorithms explained clearly | Beginner |
| 📺 [MIT 18.065 — Matrix Methods in Data Analysis](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) | SVD, PCA, neural nets from linear algebra | Advanced |

### Books

| Book | Authors | Best For |
|------|---------|----------|
| 📖 [Mathematics for Machine Learning](https://mml-book.github.io/) | Deisenroth, Faisal, Ong | Comprehensive ML math (free PDF) |
| 📖 [Deep Learning](https://www.deeplearningbook.org/) | Goodfellow, Bengio, Courville | Part I: Applied math for DL (free online) |
| 📖 [Pattern Recognition and ML](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) | Bishop | Probabilistic ML, Bayesian methods |
| 📖 [Linear Algebra Done Right](https://linear.axler.net/) | Axler | Proof-based linear algebra |
| 📖 [All of Statistics](https://www.stat.cmu.edu/~larry/all-of-statistics/) | Wasserman | Statistics for ML practitioners |
| 📖 [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) | Boyd, Vandenberghe | Optimization theory (free PDF) |
| 📖 [Information Theory, Inference, and Learning Algorithms](https://www.inference.org.uk/itprnn/book.pdf) | MacKay | Info theory + ML (free PDF) |

### Interactive Tools

| Tool | Use Case |
|------|----------|
| 🔧 [Desmos](https://www.desmos.com/calculator) | 2D graphing, function visualization |
| 🔧 [GeoGebra](https://www.geogebra.org/) | 3D geometry, linear transformations |
| 🔧 [Wolfram Alpha](https://www.wolframalpha.com/) | Symbolic computation, verification |
| 🔧 [Seeing Theory](https://seeing-theory.brown.edu/) | Interactive probability visualizations |
| 🔧 [Immersive Math](http://immersivemath.com/ila/) | Interactive linear algebra textbook |
| 🔧 [Distill.pub](https://distill.pub/) | Interactive ML research articles |

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute

- 🐛 Fix typos, errors, or broken links
- 📝 Improve explanations or add intuition
- 📓 Add new Jupyter notebook examples
- 📊 Create visualizations
- ✏️ Create new exercises or challenge problems
- 🌐 Translate content

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### 🌟 Star this repository if you find it useful!

**226 files · 14 sections · 72 topics · 144 interactive notebooks**

Built with ❤️ for the ML community

*"In mathematics the art of proposing a question must be held of higher value than solving it."* — Georg Cantor

</div>
