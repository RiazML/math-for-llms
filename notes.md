<div align="center">

# 🧮 Mathematics for AI/ML & LLM Mastery

### The Only Math Resource You Need — From Zero to Building LLMs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.24+-green.svg)](https://numpy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LLM Ready](https://img.shields.io/badge/LLM-Training%20Ready-8A2BE2.svg)]()
![Sections](https://img.shields.io/badge/sections-16-blueviolet)
![Topics](https://img.shields.io/badge/topics-85-orange)
![Notebooks](https://img.shields.io/badge/notebooks-170-red)

*From number systems to training your own LLMs — everything you need, in one place.*

> *"The only way to learn mathematics is to do mathematics."* — Paul Halmos

</div>

---

## ⚡ Non-Negotiable Principles

Every single topic in this repository follows these five rules. No exceptions.

1. **No formula without implementation** — Every equation has working NumPy + PyTorch code
2. **No implementation without measurable output** — Every notebook produces a plot, metric, or artifact
3. **No module without a failure case** — Every topic shows what breaks and exactly how to fix it
4. **Every topic produces training data** — Every concept generates JSONL-ready artifacts for LLM training
5. **Reproducibility is required** — Every notebook has a fixed seed, config file, and expected output

---

## 📋 Table of Contents

- [Why This Repository](#-why-this-repository)
- [Who This Is For](#-who-this-is-for)
- [What You Will Be Able to Do](#-what-you-will-be-able-to-do)
- [Repository Structure](#-repository-structure)
- [Learning Roadmap](#-learning-roadmap)
- [Detailed Curriculum](#-detailed-curriculum)
- [How to Use This Repository](#-how-to-use-this-repository)
- [Content Format Standard](#-content-format-standard)
- [JSONL Training Data Standard](#-jsonl-training-data-standard)
- [Prerequisites](#-prerequisites)
- [Quick Reference Docs](#-quick-reference-docs)
- [Build Phases](#-build-phases)
- [Progress Tracker](#-progress-tracker)
- [Done Criteria](#-done-criteria)
- [Resources](#-resources)
- [Contributing](#-contributing)

---

## 💡 Why This Repository

Most ML math resources fall into two traps: abstract textbooks that never connect to code, or shallow tutorials that skip the "why." This repository fixes both problems and goes further — every topic connects directly to how real models like GPT, Llama, and BERT actually work, and every concept generates training data you can use to build your own LLMs.

| Problem | Solution |
|---------|----------|
| Math textbooks feel disconnected from ML | Every formula is tied to a real model or algorithm |
| Code-only tutorials skip the theory | Every notebook starts with derivations before code |
| Hard to know what to learn first | Clear 16-section roadmap from foundations to LLM building |
| No way to test understanding | 85 exercise notebooks with progressive difficulty |
| Can't connect math to LLM training | Every topic generates JSONL training data artifacts |
| Scattered across many sites | One complete, self-contained repository |

---

## 🎯 Who This Is For

| Audience | What You Get |
|----------|--------------|
| 🎓 **Students** preparing for ML/AI careers | A structured curriculum from foundations to mastery |
| 💼 **ML Engineers** wanting deeper understanding | The "why" behind every algorithm, not just API calls |
| 📊 **Data Scientists** strengthening foundations | Fill gaps in linear algebra, probability, optimization |
| 🤖 **LLM Builders & Researchers** | Full transformer math, scaling laws, LoRA, RLHF, training dynamics |
| 🎤 **Interview Candidates** | 34+ solved derivation questions you will be asked |
| 🔄 **Career Switchers** transitioning to AI/ML | A clear learning path with time estimates |
| 📝 **Researchers** reading ML papers | Notation guide + every formula decoded |

---

## ✅ What You Will Be Able to Do

After completing this repository you will be able to:

**Core ML Math**
- Read any ML paper and understand every equation
- Derive backpropagation, PCA, SVD, and GMM-EM from scratch
- Implement optimization algorithms (SGD, Adam, L-BFGS) from math alone
- Explain probabilistic models mathematically
- Reason about numerical stability, convergence, and generalization

**LLM Specific**
- Derive scaled dot-product attention from first principles
- Understand transformer architecture math end to end
- Read LLM training papers (Chinchilla, GPT-4, LoRA) and follow every equation
- Implement attention, positional encodings, and LM sampling from scratch
- Calculate optimal model size and training tokens from a compute budget using Chinchilla laws
- Fine-tune models efficiently using LoRA math
- Build and format your own LLM training dataset in JSONL format

**Practical Engineering**
- Debug NaN losses using floating point knowledge
- Choose the right optimizer and learning rate schedule with mathematical justification
- Understand what BF16 vs FP16 means for training stability
- Answer ML math interview questions at depth

---

## 🏗️ Repository Structure

```
math_for_ai/
│
├── 📐 01-Mathematical-Foundations/
│   ├── 01-Number-Systems/
│   ├── 02-Sets-and-Logic/
│   ├── 03-Functions-and-Mappings/
│   ├── 04-Summation-and-Product-Notation/
│   ├── 05-Einstein-Summation-and-Index-Notation/     ← Critical for tensors
│   └── 06-Proof-Techniques/                          [Optional]
│
├── 📊 02-Linear-Algebra-Basics/                      ← Essential for all ML
│   ├── 01-Vectors-and-Spaces/
│   ├── 02-Matrix-Operations/
│   ├── 03-Systems-of-Equations/
│   ├── 04-Determinants/
│   ├── 05-Matrix-Rank/
│   └── 06-Vector-Spaces-Subspaces/
│
├── 🔬 03-Advanced-Linear-Algebra/                    ← PCA, SVD, spectral methods
│   ├── 01-Eigenvalues-and-Eigenvectors/
│   ├── 02-Singular-Value-Decomposition/
│   ├── 03-Principal-Component-Analysis/
│   ├── 04-Linear-Transformations/
│   ├── 05-Orthogonality-and-Orthonormality/
│   ├── 06-Matrix-Norms/
│   ├── 07-Positive-Definite-Matrices/
│   └── 08-Matrix-Decompositions/
│
├── 📈 04-Calculus-Fundamentals/                      ← Derivatives & integration
│   ├── 01-Limits-and-Continuity/
│   ├── 02-Derivatives-and-Differentiation/
│   ├── 03-Integration/
│   └── 04-Series-and-Sequences/
│
├── 🌊 05-Multivariate-Calculus/                      ← Gradients & backprop
│   ├── 01-Partial-Derivatives-and-Gradients/
│   ├── 02-Jacobians-and-Hessians/
│   ├── 03-Chain-Rule-and-Backpropagation/
│   └── 04-Optimization-Theory/
│
├── 🎲 06-Probability-Theory/                         ← Uncertainty & inference
│   ├── 01-Introduction-and-Random-Variables/
│   ├── 02-Common-Distributions/
│   ├── 03-Joint-Distributions/
│   └── 04-Expectation-and-Moments/
│
├── 📉 07-Statistics/                                 ← Estimation & testing
│   ├── 01-Descriptive-Statistics/
│   ├── 02-Estimation-Theory/
│   ├── 03-Hypothesis-Testing/
│   └── 04-Bayesian-Inference/
│
├── ⚡ 08-Optimization/                               ← Training algorithms
│   ├── 01-Convex-Optimization/
│   ├── 02-Gradient-Descent/
│   ├── 03-Second-Order-Methods/
│   ├── 04-Constrained-Optimization/
│   ├── 05-Stochastic-Optimization/
│   ├── 06-Optimization-Landscape/
│   ├── 07-Adaptive-Learning-Rate/
│   ├── 08-Regularization-Methods/
│   ├── 09-Hyperparameter-Optimization/
│   └── 10-Learning-Rate-Schedules/                   ← Warmup, cosine decay, LLM schedules
│
├── 📡 09-Information-Theory/                         ← Loss functions & metrics
│   ├── 01-Entropy/
│   ├── 02-KL-Divergence/
│   ├── 03-Mutual-Information/
│   └── 04-Cross-Entropy/
│
├── 🔢 10-Numerical-Methods/                          ← Stability & precision
│   ├── 01-Floating-Point-Arithmetic/                 ← FP32/FP16/BF16 + AMP
│   ├── 02-Numerical-Linear-Algebra/
│   ├── 03-Numerical-Optimization/
│   ├── 04-Interpolation-and-Approximation/
│   └── 05-Numerical-Integration/
│
├── 🕸️ 11-Graph-Theory/                              ← GNNs & structured data
│   ├── 01-Graph-Basics/
│   ├── 02-Graph-Representations/
│   ├── 03-Graph-Algorithms/
│   ├── 04-Spectral-Graph-Theory/
│   └── 05-Graph-Neural-Networks/
│
├── 🏛️ 12-Functional-Analysis/                       [Optional — Advanced Theory]
│   ├── 01-Vector-Spaces/
│   ├── 02-Normed-Spaces/
│   ├── 03-Hilbert-Spaces/
│   └── 04-Kernel-Methods/
│
├── 🤖 13-ML-Specific-Math/                           ← Applied ML concepts
│   ├── 01-Loss-Functions/
│   ├── 02-Activation-Functions/
│   ├── 03-Attention-Mechanisms/
│   ├── 04-Normalization-Techniques/
│   └── 05-Sampling-Methods/
│
├── 🧠 14-Math-for-Specific-Models/                   ← Full model derivations
│   ├── 01-Linear-Models/
│   ├── 02-Neural-Networks/
│   ├── 03-Probabilistic-Models/
│   ├── 04-Sequence-Models/
│   │   ├── 04a-RNN-and-LSTM-Math/
│   │   └── 04b-Transformer-Architecture-Math/        ← Full block math
│   └── 05-Generative-Models/
│
├── 🔥 15-Math-for-LLMs/                              ← NEW: Complete LLM math
│   ├── 01-Tokenization-Math/
│   ├── 02-Embedding-Space-Math/
│   ├── 03-Attention-Mechanism-Math/                  ← Highest priority
│   ├── 04-Positional-Encodings/
│   ├── 05-Language-Model-Probability/
│   ├── 06-Training-at-Scale/
│   ├── 07-Fine-Tuning-Math/
│   └── 08-Scaling-Laws/
│
├── 📦 16-LLM-Training-Data-Pipeline/                 ← NEW: Build your dataset
│   ├── 01-Data-Format-Standards/
│   ├── 02-JSONL-Generation/
│   ├── 03-Quality-Checks/
│   └── 04-Full-Dataset-Assembly/
│
├── 📚 docs/
│   ├── CHEATSHEET.md                                 ← All key formulas + LLM block
│   ├── NOTATION_GUIDE.md
│   ├── ML_MATH_MAP.md
│   ├── LLM_MATH_MAP.md                               ← NEW: Which math powers which model
│   ├── MATH_TO_CODE.md                               ← NEW: 50 formula → code mappings
│   ├── INTERVIEW_PREP.md                             ← 34+ solved questions
│   └── VISUALIZATION_GUIDE.md
│
├── training_data/                                    ← NEW: Generated JSONL dataset
│   ├── by_section/
│   │   ├── 01_foundations.jsonl
│   │   ├── 02_linear_algebra.jsonl
│   │   └── ... (one per section)
│   └── full_dataset.jsonl                            ← Merged, deduped, ready
│
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

**Each topic folder contains exactly 3 files:**

| File | Purpose |
|------|---------|
| `README.md` | Theory, derivations, intuition, ML connections, failure cases |
| `theory.ipynb` | Code + visualizations + measurable outputs |
| `exercises.ipynb` | Problems with full solutions, progressive difficulty |

---

## 🗺️ Learning Roadmap

```
┌─────────────────────────────────────────────────────────────────┐
│              MATHEMATICS FOR AI/ML & LLM MASTERY                │
│                16 Sections · 85 Topics · 170 Notebooks          │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌──────────────────┐     ┌───────────────────┐
│ 01 FOUNDATIONS│     │ 02 LINEAR ALGEBRA│     │ 04 CALCULUS       │
│ • Notation    │     │ • Vectors/Matrices│────▶│ • Derivatives     │
│ • Einsum      │────▶│ • Matrix Ops     │     │ • Integration     │
│ • Sets/Logic  │     │ • Systems/Rank   │     │ • Series          │
└───────────────┘     └──────────────────┘     └───────────────────┘
                                │                       │
                                ▼                       ▼
                    ┌───────────────────┐   ┌───────────────────────┐
                    │ 03 ADV. LIN ALG   │   │ 05 MULTIVARIATE CALC  │
                    │ • Eigenvalues/SVD │   │ • Gradients/Jacobians │
                    │ • PCA             │   │ • Chain Rule          │
                    │ • Decompositions  │   │ • Backpropagation     │
                    └───────────────────┘   └───────────────────────┘
                                │                       │
              ┌─────────────────┴───────────────────────┘
              │
        ┌─────┴──────┐
        │             │
        ▼             ▼
┌──────────────┐ ┌───────────────┐
│ 06 PROB THEORY│ │ 07 STATISTICS │
│ • Distributions│ │ • MLE / MAP  │
│ • Bayes Theorem│─▶│ • Hypothesis │
│ • Expectations│ │ • Bayesian   │
└──────────────┘ └───────────────┘
        │
        ├──────────────────────────────────────────────┐
        │                       │                      │
        ▼                       ▼                      ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│ 08 OPTIMIZATION  │  │ 09 INFO THEORY   │  │ 10 NUMERICAL METHODS │
│ • Grad Descent   │  │ • Entropy        │  │ • FP32/FP16/BF16    │
│ • Adam/AdamW     │  │ • KL Divergence  │  │ • Stability         │
│ • LR Schedules   │  │ • Cross-Entropy  │  │ • Mixed Precision   │
│ • Regularization │  └──────────────────┘  └──────────────────────┘
└──────────────────┘
        │
        ├───────────────────────┬──────────────────────┐
        │                       │                      │
        ▼                       ▼                      ▼
┌──────────────┐  ┌────────────────────────┐  ┌──────────────────┐
│ 11 GRAPH     │  │ 13 ML-SPECIFIC MATH    │  │ 12 FUNCTIONAL    │
│ • GNNs       │  │ • Loss Functions       │  │    ANALYSIS      │
│ • Spectral   │  │ • Activations/Norms    │  │  [OPTIONAL]      │
└──────────────┘  │ • Attention Mechanisms │  └──────────────────┘
                  └────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │  14 MATH FOR SPECIFIC MODELS   │
              │  • Linear Models               │
              │  • Neural Networks (full math) │
              │  • Probabilistic Models        │
              │  • Transformers (04b)          │
              │  • Generative Models (VAE/GAN) │
              └────────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │  15 MATH FOR LLMs         🔥   │
              │  • Tokenization Math           │
              │  • Attention Derivation (QKV)  │
              │  • Positional Encodings        │
              │  • Perplexity / Sampling       │
              │  • Training at Scale           │
              │  • LoRA / RLHF Math            │
              │  • Scaling Laws (Chinchilla)   │
              └────────────────────────────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │  16 LLM TRAINING DATA PIPELINE │
              │  • JSONL Format Standard       │
              │  • 6-Type Generator Pipeline   │
              │  • Quality Checks + Dedup      │
              │  • full_dataset.jsonl (680+)   │
              │                                │
              │       🎓 LLM BUILDER!          │
              └────────────────────────────────┘
```

---

## 📖 Detailed Curriculum

---

### Section 01 — Mathematical Foundations
> *Build the mathematical language that everything else uses.*

| Topic | What You Learn | ML Connection |
|-------|---------------|---------------|
| Number Systems | Real, complex, integers, floating point | Data types, precision limits |
| Sets and Logic | Set operations, logical quantifiers | Data pipeline logic, conditions |
| Functions and Mappings | Injective, surjective, composition | Layer functions, activations |
| Summation and Product Notation | Σ, Π, index manipulation | Loss sums, likelihood products |
| **Einstein Summation** | Repeated index convention, tensor contraction, einsum | Tensor ops in every neural network |
| Proof Techniques *(Optional)* | Induction, contradiction, direct proof | Reading theorems in papers |

**⭐ Do not skip Einstein Summation.** `torch.einsum` and `np.einsum` appear in attention, convolutions, and matrix ops everywhere.

---

### Section 02 — Linear Algebra Basics
> *The language of data. Every input is a vector, every layer is a matrix.*

| Topic | What You Learn | ML Connection |
|-------|---------------|---------------|
| Vectors and Spaces | Vector ops, norms (L1, L2), dot product | Feature representations, similarity |
| Matrix Operations | Multiply, transpose, inverse | Forward pass: output = Wx + b |
| Systems of Equations | Ax = b, overdetermined systems | Least squares regression |
| Determinants | Geometric meaning, cofactor expansion | Volume scaling, invertibility |
| Matrix Rank | Column space, null space, rank-nullity | Dimensionality, LoRA rank |
| Vector Spaces | Basis, span, linear independence | Representation capacity |

---

### Section 03 — Advanced Linear Algebra
> *PCA, SVD, and the decompositions that power compression and dimensionality reduction.*

| Topic | What You Learn | ML Connection |
|-------|---------------|---------------|
| Eigenvalues & Eigenvectors | Characteristic equation, diagonalization | PCA, graph Laplacian, covariance |
| SVD | U Σ Vᵀ decomposition, truncated SVD | Compression, recommender systems, LoRA |
| PCA | Covariance matrix, projection, variance explained | Dimensionality reduction, visualization |
| Linear Transformations | Rotation, scaling, shear as matrices | Data augmentation, geometry |
| Orthogonality | Gram-Schmidt, QR decomposition | Stable numerical computation |
| Matrix Norms | Frobenius, spectral, nuclear norm | Regularization, LoRA nuclear norm |
| Positive Definite Matrices | Definiteness tests, Cholesky | Covariance matrices, Gaussian processes |
| Matrix Decompositions | LU, QR, Cholesky, Schur | Numerical solvers, stability |

---

### Section 04 — Calculus Fundamentals
> *Derivatives are the engine of learning. Integration handles probability.*

| Topic | What You Learn | ML Connection |
|-------|---------------|---------------|
| Limits and Continuity | Epsilon-delta, continuity conditions | Loss functions must be smooth |
| Derivatives | Rules: power, chain, product, quotient | Sensitivity of loss to any parameter |
| Integration | Definite/indefinite, techniques | Expected values, area under curves |
| Series and Sequences | Taylor series, convergence | Function approximation, numerical methods |

---

### Section 05 — Multivariate Calculus
> *The math behind how neural networks actually learn.*

| Topic | What You Learn | ML Connection |
|-------|---------------|---------------|
| Partial Derivatives & Gradients | Gradient vector, directional derivatives | Direction to update every weight |
| Jacobians and Hessians | Matrix of derivatives, curvature | Sensitivity, second-order optimizers |
| Chain Rule & Backpropagation | Multivariate chain rule, computational graphs | **This IS backpropagation** |
| Optimization Theory | Critical points, convexity, Lagrange multipliers | Finding loss minima |

---

### Section 06 — Probability Theory
> *Models output probabilities. This section explains what that means mathematically.*

| Topic | What You Learn | ML Connection |
|-------|---------------|---------------|
| Random Variables | Discrete/continuous, PMF, PDF, CDF | Model output distributions |
| Common Distributions | Gaussian, Bernoulli, Categorical, Poisson | Noise modeling, classification heads |
| Joint Distributions | Marginal, conditional, independence, Bayes | Bayesian inference, graphical models |
| Expectation and Moments | Mean, variance, covariance, MGF | Loss expectations, variance reduction |

---

### Section 07 — Statistics
> *How you estimate model parameters and know when one model beats another.*

| Topic | What You Learn | ML Connection |
|-------|---------------|---------------|
| Descriptive Statistics | Central tendency, dispersion, correlation | EDA, feature engineering |
| Estimation Theory | MLE, MAP, bias-variance, confidence intervals | **Training IS maximum likelihood** |
| Hypothesis Testing | t-test, p-values, multiple comparisons | A/B testing, model comparison |
| Bayesian Inference | Priors, posteriors, conjugacy, MCMC | Bayesian NNs, uncertainty quantification |

---

### Section 08 — Optimization
> *The largest section — because training a model IS an optimization problem.*

| Topic | What You Learn | ML Connection |
|-------|---------------|---------------|
| Convex Optimization | Convex sets/functions, duality, KKT conditions | SVM formulation, theoretical guarantees |
| Gradient Descent | Batch vs SGD vs mini-batch, convergence analysis | Core training algorithm |
| Second-Order Methods | Newton's method, BFGS, L-BFGS | Faster convergence, Hessian-free |
| Constrained Optimization | Lagrangian, KKT, duality | SVMs, constrained learning |
| Stochastic Optimization | SGD variants, variance reduction | Large-scale training |
| Optimization Landscape | Saddle points, loss surfaces, basins | Why deep networks train successfully |
| Adaptive Learning Rate | Adam, AdamW, RMSProp, AdaGrad | Modern deep learning optimizers |
| Regularization Methods | L1, L2, dropout, weight decay | Preventing overfitting |
| Hyperparameter Optimization | Grid search, Bayesian opt, Hyperband | Tuning model performance |
| **Learning Rate Schedules** | Linear warmup, cosine decay, cyclic LR | **How LLMs are trained** |

---

### Section 09 — Information Theory
> *The mathematical foundation of loss functions and measuring uncertainty.*

| Topic | What You Learn | ML Connection |
|-------|---------------|---------------|
| Entropy | Shannon entropy, information content | Decision trees, uncertainty measurement |
| KL Divergence | Asymmetric distance, forward vs reverse | VAE loss, knowledge distillation |
| Mutual Information | Dependence measurement | Feature selection, self-supervised learning |
| Cross-Entropy | Connection to KL, log-loss | **Most common classification loss** |

---

### Section 10 — Numerical Methods
> *Why math on paper behaves differently on silicon — and how to fix it.*

| Topic | What You Learn | ML Connection |
|-------|---------------|---------------|
| **Floating-Point Arithmetic** | FP32/FP16/BF16 ranges, precision, overflow | Mixed precision training, NaN losses |
| Numerical Linear Algebra | Condition numbers, iterative solvers | Stable ML algorithm implementation |
| Numerical Optimization | Line search, trust regions, convergence | Practical training considerations |
| Interpolation & Approximation | Polynomial, spline, Chebyshev | Data augmentation, function approx |
| Numerical Integration | Quadrature, Monte Carlo | Expected value computation |

**BF16 vs FP16 — why it matters:** BF16 has the same exponent range as FP32 so it never overflows during training. FP16 overflows easily and needs loss scaling. Modern LLMs all train in BF16.

---

### Section 11 — Graph Theory
> *The math behind GNNs, knowledge graphs, and social networks.*

| Topic | What You Learn | ML Connection |
|-------|---------------|---------------|
| Graph Basics | Vertices, edges, degree, connectivity | Relational data structure |
| Graph Representations | Adjacency matrix, Laplacian | Input format for graph algorithms |
| Graph Algorithms | BFS, DFS, shortest path | Pathfinding, clustering |
| Spectral Graph Theory | Laplacian eigenvalues, Cheeger inequality | Spectral clustering, partitioning |
| Graph Neural Networks | Message passing, GCN, GAT, pooling | Node/edge/graph classification |

---

### Section 12 — Functional Analysis *(Optional)*
> *Rigorous theory behind kernel methods. Skip unless you need RKHS or Gaussian Processes.*

| Topic | What You Learn | ML Connection |
|-------|---------------|---------------|
| Vector Spaces | Axioms, function spaces | Theory of hypothesis spaces |
| Normed Spaces | Lp norms, Banach spaces | Regularization theory |
| Hilbert Spaces | Inner products, Riesz representation | RKHS foundation |
| Kernel Methods | Mercer's theorem, representer theorem | SVMs, kernel PCA, GPs |

---

### Section 13 — ML-Specific Math
> *Mathematical concepts purpose-built for machine learning.*

| Topic | What You Learn | ML Connection |
|-------|---------------|---------------|
| Loss Functions | Design principles, properties, custom losses | Choosing the right objective |
| Activation Functions | Sigmoid, ReLU, GELU, SwiGLU — derivatives | Nonlinearity, gradient flow |
| Attention Mechanisms | QKV formulation, softmax temperature | Foundation for Section 15 |
| Normalization Techniques | BatchNorm, LayerNorm, RMSNorm — math | Training stability in deep networks |
| Sampling Methods | MCMC, Metropolis-Hastings, variational inference | Generative models, Bayesian inference |

---

### Section 14 — Math for Specific Models
> *Complete mathematical derivations for every major ML model family.*

| Topic | What You Learn | ML Connection |
|-------|---------------|---------------|
| Linear Models | Normal equations, ridge/lasso derivation, GLMs | Linear/logistic regression from scratch |
| Neural Networks | Universal approximation, backprop calculus, init math | MLPs, CNNs, residual networks |
| Probabilistic Models | EM algorithm, GMMs, HMMs, Gaussian processes | Bayesian ML, uncertainty |
| **04a: RNN and LSTM Math** | Vanishing gradient, LSTM gates, BPTT | Sequence modeling theory |
| **04b: Transformer Architecture** | Full block math, pre-norm, SwiGLU, residuals | Foundation for Section 15 |
| Generative Models | VAE ELBO, GAN minimax, diffusion SDEs | Image generation, density estimation |

---

### Section 15 — Math for Large Language Models 🔥
> *The complete mathematical foundation behind GPT, Llama, BERT, and every modern LLM.*

| Topic | What You Learn | ML Connection |
|-------|---------------|---------------|
| **Tokenization Math** | BPE algorithm, vocabulary construction, byte-level BPE | How GPT/Llama converts text to numbers |
| **Embedding Space Math** | Embedding matrix E ∈ ℝ^(V×d), cosine similarity, vector arithmetic | Token representations, semantic search |
| **Attention Mechanism Math** ⭐ | Full QKV derivation, why divide by √d_k, causal masking, multi-head | **Core of every transformer** |
| **Positional Encodings** | Sinusoidal formula, RoPE rotation math, ALiBi bias | How models understand token order |
| **Language Model Probability** | Autoregressive factorization, perplexity, temperature, top-k, top-p | Controlling LLM generation |
| **Training at Scale** | Gradient accumulation, BF16 vs FP16, gradient clipping, cosine warmup | How LLMs are actually trained |
| **Fine-Tuning Math** | LoRA: W = W₀ + BA, rank r tradeoff, QLoRA, RLHF + PPO objective | Fine-tune without full retraining |
| **Scaling Laws** | Chinchilla: N_opt ∝ C^0.5, compute-optimal training, emergent abilities | Plan your LLM training budget |

**Every topic in Section 15 includes:** Full derivation → NumPy implementation → PyTorch implementation → Visualization → Failure case + fix → 6 JSONL training pairs

---

### Section 16 — LLM Training Data Pipeline
> *Turn every math concept into high-quality training data to build your own LLMs.*

| Topic | What You Build | Output |
|-------|---------------|--------|
| Data Format Standards | JSONL schema, schema validator | `schema_validator.py` |
| JSONL Generation | 6-type generator pipeline for all 85 topics | `generators/` scripts |
| Quality Checks | Dedup, length filter, difficulty labeler, metadata tagger | `checkers/` scripts |
| Full Dataset Assembly | Merge all sections, train/val split, tokenization stats | `full_dataset.jsonl` |

**Minimum outputs:** 680+ training examples · 85 topics × 6 types · Schema-validated · Train/val split ready

---

## 🚀 How to Use This Repository

### Setup

```bash
# Clone
git clone https://github.com/yourusername/math_for_ai.git
cd math_for_ai

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Study Method for Every Topic

```
┌─────────────────────────────────────────┐
│  1. 📖 Read README.md                   │  ← Theory, intuition, ML connection
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│  2. 📓 Run theory.ipynb                 │  ← Code + plots + measurable output
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│  3. ✏️  Solve exercises.ipynb           │  ← Try WITHOUT looking at solutions
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│  4. ✅ Check solutions + review errors  │  ← Understand mistakes, fill gaps
└────────────────────┬────────────────────┘
                     ▼
┌─────────────────────────────────────────┐
│  5. 🤖 Review JSONL pairs generated     │  ← See the training data artifact
└─────────────────────────────────────────┘
```

### Quick Start by Goal

| Your Goal | Start Here | Then |
|-----------|-----------|------|
| Understand deep learning math | 02 → 04 → 05 → 08 | 14-02 Neural Networks |
| Build or fine-tune an LLM | 02 → 05 → 08 → 09 | 15 (all) → 16 |
| Train an LLM on custom data | 15 → 16 | full_dataset.jsonl |
| Understand transformer architecture | 02 → 05 → 13 → 14-04b | 15-03 Attention |
| Prepare for ML interviews | 02 → 06 → 08 → 09 | INTERVIEW_PREP.md |
| Read ML research papers | 01 → 03 → 05 → 09 | NOTATION_GUIDE.md |
| Implement models from scratch | 02 → 05 → 08 → 10 | Section 14 |
| Work with graph data / GNNs | 02 → 11 | 11-05 GNNs |
| Understand generative AI | 06 → 09 → 14-05 | 15-07 Fine-Tuning Math |

---

## 📐 Content Format Standard

Every single topic `README.md` follows this exact structure. No exceptions.

```markdown
# [Topic Name]

> Section: [Name] | Difficulty: [Beginner/Intermediate/Advanced] | Time: [X hrs]

## Why This Matters in ML
[1 paragraph. Concrete. Name a real model or algorithm.]

## The Core Idea (Intuition First)
[Plain English. No jargon. Use an analogy if needed.]

## Mathematical Foundation

### [Sub-concept]
[Definition]

**Formula:**
$$[LaTeX]$$

**What each symbol means:**
- symbol: meaning

## Worked Numeric Example
[Step-by-step by hand with real numbers]

## Implementation

### NumPy
```python
# code
```

### PyTorch
```python
# code
```

## ❌ Common Mistakes → ✅ Fixes
- ❌ Wrong: [mistake]  →  ✅ Right: [fix]
- ❌ Wrong: [mistake]  →  ✅ Right: [fix]

## Where This Appears in Real Models
| Model | How [Concept] Is Used |
|-------|-----------------------|
| GPT   | ...                   |
| BERT  | ...                   |

## Key Takeaways
- Point 1
- Point 2
- Point 3

## References
- [Resource]
```

---

## 📦 JSONL Training Data Standard

Every topic generates 6 training pair types. This is how you build your own LLM training dataset.

### Schema

```json
{
  "id":          "sec02_t03_exp_001",
  "section":     "02-Linear-Algebra-Basics",
  "topic":       "Matrix Multiplication",
  "type":        "explanation",
  "difficulty":  "beginner",
  "prompt":      "Explain matrix multiplication and why it matters in neural networks.",
  "completion":  "...",
  "tags":        ["linear-algebra", "matrix", "neural-networks"],
  "has_code":    false,
  "has_math":    true,
  "word_count":  120
}
```

### 6 Required Types Per Topic

| Type | Prompt Pattern | Purpose |
|------|---------------|---------|
| `explanation` | "Explain [concept] and how it is used in ML." | Core understanding |
| `qa` | "Q: [specific question about concept]" | Fact retrieval |
| `derivation` | "Derive [formula] step by step." | Reasoning depth |
| `code_math` | "Show the math for [concept] and implement in NumPy + PyTorch." | Code grounding |
| `error_correction` | "What is wrong with this reasoning about [concept]? [wrong example]" | Error detection |
| `concept_connection` | "How does [concept A] connect to [concept B] in practice?" | Cross-topic links |

### ID Naming Convention

```
[section]_[topic]_[type_code]_[number]

sec02_t03_exp_001   → Section 02, Topic 03, Explanation, #1
sec15_t03_der_002   → Section 15, Topic 03, Derivation, #2
sec08_t07_cod_001   → Section 08, Topic 07, Code+Math, #1
```

### Dataset Targets

| Metric | Target |
|--------|--------|
| Total examples | 680+ |
| Types per topic | 6 |
| Topics covered | All 85 |
| Output file | `training_data/full_dataset.jsonl` |
| Validated by | `schema_validator.py` |
| Split | 90% train / 10% validation |

---

## 📋 Prerequisites

| Requirement | Level | Brush Up |
|-------------|-------|----------|
| Basic Algebra | Variables, equations, polynomials | [Khan Academy Algebra](https://www.khanacademy.org/math/algebra) |
| High School Math | Geometry, trigonometry basics | [Khan Academy Precalculus](https://www.khanacademy.org/math/precalculus) |
| Python Basics | Functions, loops, lists, classes | [Python Tutorial](https://docs.python.org/3/tutorial/) |
| NumPy Basics | Array operations (helpful, not required) | [NumPy Quickstart](https://numpy.org/doc/stable/user/quickstart.html) |

> If unsure, start with **Section 01** — it brings you up to speed fast.

---

## 📚 Quick Reference Docs

| Document | What It Contains | Best For |
|----------|-----------------|----------|
| 📄 [Cheatsheet](docs/CHEATSHEET.md) | All key formulas including full LLM block | Quick formula lookup |
| 📝 [Notation Guide](docs/NOTATION_GUIDE.md) | Every symbol decoded | Reading ML papers |
| 🗺️ [ML Math Map](docs/ML_MATH_MAP.md) | Which math appears in which algorithm | Planning study path |
| 🤖 [LLM Math Map](docs/LLM_MATH_MAP.md) | Which math powers GPT/BERT/Llama/Mistral | Building or reading LLMs |
| ⚡ [Math to Code](docs/MATH_TO_CODE.md) | 50 formulas → NumPy + PyTorch code | Translating papers to code |
| 🎤 [Interview Prep](docs/INTERVIEW_PREP.md) | 34+ solved derivation questions | ML engineer interviews |
| 📊 [Visualization Guide](docs/VISUALIZATION_GUIDE.md) | Ready-to-use plots for every concept | Building intuition |

---

## 🔨 Build Phases

Build this repository in this exact order:

### Phase 1 — Foundation *(Do First)*
- [ ] README global updates (badges, counts, new sections)
- [ ] Add `01-05-Einstein-Summation-and-Index-Notation/`
- [ ] Update `10-01-Floating-Point-Arithmetic/` with FP32/FP16/BF16/AMP content

### Phase 2 — Core LLM Math
- [ ] `15-01` Tokenization Math
- [ ] `15-02` Embedding Space Math
- [ ] `15-03` Attention Mechanism Math ⭐ **highest priority**
- [ ] `15-04` Positional Encodings
- [ ] `15-05` Language Model Probability

### Phase 3 — Advanced LLM
- [ ] `15-06` Training at Scale
- [ ] `15-07` LoRA / QLoRA / RLHF Math
- [ ] `15-08` Scaling Laws
- [ ] `14-04b` Transformer Architecture Math
- [ ] `08-10` Learning Rate Schedules

### Phase 4 — Documentation
- [ ] Create `docs/LLM_MATH_MAP.md`
- [ ] Create `docs/MATH_TO_CODE.md` (50 mappings)
- [ ] Update `docs/CHEATSHEET.md` with LLM formula block
- [ ] Update `docs/INTERVIEW_PREP.md` with 8 new LLM derivation questions

### Phase 5 — Training Data Pipeline
- [ ] Build `16-01` Data Format Standards + `schema_validator.py`
- [ ] Build `16-02` All 6 generator scripts
- [ ] Build `16-03` Quality check pipeline (dedup, filter, label, tag)
- [ ] Build `16-04` Assembly scripts + train/val split
- [ ] Generate all 680+ JSONL examples
- [ ] Validate with schema validator
- [ ] Assemble `training_data/full_dataset.jsonl`

### Phase 6 — Quality Gate *(Final)*
- [ ] All internal links resolve (no 404s)
- [ ] All notebooks run end to end without errors
- [ ] Notebook metadata normalized (kernelspec, Python version)
- [ ] JSONL schema validation passes for all examples
- [ ] README section/topic/notebook counts match actual filesystem
- [ ] `requirements.txt` includes all packages used

---

## ✅ Progress Tracker

### Core Sections

- [ ] **01 Mathematical Foundations** — Notation, sets, functions, einsum *(~8 hrs)*
- [ ] **02 Linear Algebra Basics** — Vectors, matrices, systems, rank *(~15 hrs)*
- [ ] **03 Advanced Linear Algebra** — Eigenvalues, SVD, PCA *(~18 hrs)*
- [ ] **04 Calculus Fundamentals** — Limits, derivatives, integration *(~12 hrs)*
- [ ] **05 Multivariate Calculus** — Gradients, Jacobians, backprop *(~10 hrs)*
- [ ] **06 Probability Theory** — Distributions, Bayes, expectation *(~14 hrs)*
- [ ] **07 Statistics** — MLE, MAP, hypothesis testing, Bayesian *(~12 hrs)*
- [ ] **08 Optimization** — Gradient descent, Adam, LR schedules *(~25 hrs)*
- [ ] **09 Information Theory** — Entropy, KL, cross-entropy *(~8 hrs)*
- [ ] **10 Numerical Methods** — FP32/FP16/BF16, stability, AMP *(~10 hrs)*
- [ ] **11 Graph Theory** — GNNs, spectral, message passing *(~15 hrs)*
- [ ] **12 Functional Analysis** *(Optional)* — Hilbert spaces, kernels *(~20 hrs)*
- [ ] **13 ML-Specific Math** — Losses, activations, norms, sampling *(~12 hrs)*
- [ ] **14 Math for Specific Models** — Full model derivations *(~20 hrs)*

### LLM Specialization

- [ ] **15-01** Tokenization Math *(~4 hrs)*
- [ ] **15-02** Embedding Space Math *(~4 hrs)*
- [ ] **15-03** Attention Mechanism Math ⭐ *(~6 hrs)*
- [ ] **15-04** Positional Encodings *(~4 hrs)*
- [ ] **15-05** Language Model Probability *(~4 hrs)*
- [ ] **15-06** Training at Scale *(~5 hrs)*
- [ ] **15-07** Fine-Tuning Math (LoRA/RLHF) *(~5 hrs)*
- [ ] **15-08** Scaling Laws *(~4 hrs)*

### Pipeline

- [ ] **16** LLM Training Data Pipeline *(~10 hrs)*
- [ ] 680+ JSONL examples generated and validated

---

**Total: ~245 hours (~6 months at 10 hrs/week)**

---

## 🏁 Done Criteria

The project is complete when all of these are true:

1. Sections 15 and 16 fully exist with all required files
2. All new docs are present and linked from this README
3. All existing section upgrades are merged and verified
4. `training_data/full_dataset.jsonl` exists and passes schema validation
5. Minimum 680+ examples generated across all 6 types
6. All internal links resolve without errors
7. README section/topic/notebook counts match the actual filesystem
8. All notebooks run clean from top to bottom with fixed seeds

---

## 🔗 Resources

### Video Courses

| Resource | Topics | Level |
|----------|--------|-------|
| [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Vectors, transformations, eigenvalues | Beginner |
| [3Blue1Brown — Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) | Derivatives, integrals | Beginner |
| [3Blue1Brown — Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) | Backpropagation, gradient descent | Beginner |
| [MIT 18.06 — Linear Algebra (Strang)](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/) | Complete linear algebra | Intermediate |
| [Stanford CS229](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU) | ML theory and math | Intermediate |
| [StatQuest](https://www.youtube.com/c/joshstarmer) | Statistics, ML algorithms | Beginner |
| [Andrej Karpathy — makemore / nanoGPT](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) | LLM building from scratch | Intermediate |

### Books

| Book | Authors | Best For |
|------|---------|----------|
| [Mathematics for Machine Learning](https://mml-book.github.io/) | Deisenroth et al. | Comprehensive ML math (free PDF) |
| [Deep Learning](https://www.deeplearningbook.org/) | Goodfellow et al. | Applied math for DL (free online) |
| [The Matrix Calculus You Need for DL](https://explained.ai/matrix-calculus/) | Parr & Howard | Backprop math (free) |
| [Pattern Recognition and ML](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) | Bishop | Probabilistic ML |
| [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) | Boyd & Vandenberghe | Optimization theory (free PDF) |
| [All of Statistics](https://www.stat.cmu.edu/~larry/all-of-statistics/) | Wasserman | Statistics for ML practitioners |

### Interactive Tools

| Tool | Use Case |
|------|----------|
| [Desmos](https://www.desmos.com/calculator) | 2D function visualization |
| [GeoGebra](https://www.geogebra.org/) | 3D geometry, linear transformations |
| [Wolfram Alpha](https://www.wolframalpha.com/) | Symbolic computation, verification |
| [Seeing Theory](https://seeing-theory.brown.edu/) | Interactive probability |
| [Distill.pub](https://distill.pub/) | Interactive ML research |
| [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) | Visual transformer walkthrough |

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

**Most wanted contributions:**

- 🤖 **JSONL training data pairs** — Q&A, derivations, code↔math pairs for LLM training (see Section 16)
- ❌✅ **Wrong → corrected examples** — Common mistake followed by exact fix
- 🔥 **PyTorch implementations** alongside existing NumPy ones
- 📐 **Einstein summation examples** — More einsum patterns for tensor operations
- 📓 **New notebook examples** with measurable outputs and fixed seeds
- 📊 **Visualizations** — matplotlib/manim animations especially welcome
- 🐛 **Bug fixes** — Typos, broken links, incorrect derivations

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

### 🌟 Star this repository if it helps you build!

**270 files · 16 sections · 85 topics · 170 notebooks · 680+ training examples**

Built for people who want to understand the math, write the code, and build the models — not just use them.

*"In mathematics the art of proposing a question must be held of higher value than solving it."* — Georg Cantor

</div>
