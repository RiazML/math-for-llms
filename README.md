<div align="center">

# Mathematics for AI / ML / LLM

### The Complete Math Foundation You Need to Master AI

[![GitHub stars](https://img.shields.io/github/stars/RiazML/math-for-llms?style=for-the-badge&logo=github&color=yellow)](https://github.com/RiazML/math-for-llms/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/RiazML/math-for-llms?style=for-the-badge&logo=github&color=blue)](https://github.com/RiazML/math-for-llms/network/members)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Notebooks](https://img.shields.io/badge/Jupyter_Notebooks-229-orange?style=for-the-badge&logo=jupyter)](.)
[![Chapters](https://img.shields.io/badge/Chapters-25-brightgreen?style=for-the-badge&logo=bookstack)](.)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](CONTRIBUTING.md)

**A structured, open-source curriculum covering 25 domains of mathematics essential for AI, Machine Learning, and Large Language Models — from foundational concepts to cutting-edge research.**

[Get Started](#-quick-start) · [Roadmap](#-learning-roadmap) · [Chapters](#-chapters) · [Resources](#-resources) · [Contributing](#-contributing)

---

<br>

<img src="https://img.shields.io/badge/115-Interactive_Theory_Notebooks-blue?style=flat-square&logo=jupyter" alt="Theory notebooks" />
<img src="https://img.shields.io/badge/114-Exercise_Notebooks-green?style=flat-square&logo=jupyter" alt="Exercise notebooks" />
<img src="https://img.shields.io/badge/115-Detailed_Notes-red?style=flat-square&logo=markdown" alt="Notes files" />

</div>

<br>

## Why This Repo?

Most AI/ML learners hit the **math wall** — papers full of symbols that feel alien, optimization steps that seem like magic, and model architectures that assume deep mathematical fluency.

This repository bridges that gap with a **learn-by-doing** approach:

- **Structured path** from high school math to research-level topics
- **Notes → Theory → Exercises** flow for every topic
- **Interactive Jupyter notebooks** with visualizations, not just formulas
- **Real ML connections** — every concept links to practical AI applications
- **Self-contained** — no prerequisites beyond basic algebra

> *"The math you need depends on what you're building — this repo helps you find exactly that."*

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/prmtkr/math_for_llms.git
cd math_for_ai

# Set up the environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

Each topic follows a **3-step learning flow**:

```
📖 notes.md          → Read the concepts and intuition
🔬 theory.ipynb      → Explore interactive demonstrations
✏️ exercises.ipynb   → Test your understanding
```

---

## 🗺️ Learning Roadmap

The curriculum covers **25 domains** organized in **8 phases**. Each phase builds on the previous one. Topics marked with ★ are critical for ML/AI.

```
  START HERE
      │
      ▼
 ┌─────────┐     ┌─────────────┐     ┌──────────┐     ┌──────────────┐
 │ Phase 1  │ ──▶ │   Phase 2   │ ──▶ │ Phase 3  │ ──▶ │   Phase 4    │
 │ Core     │     │ Probability │     │ Learning │     │ Deeper       │
 │ Math     │     │ & Stats     │     │ Engines  │     │ Theory       │
 └─────────┘     └─────────────┘     └──────────┘     └──────────────┘
                                                              │
      ┌───────────────────────────────────────────────────────┘
      ▼
 ┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────────┐
 │ Phase 5  │ ──▶ │ Phase 6  │ ──▶ │   Phase 7    │ ──▶ │   Phase 8    │
 │ ML Math  │     │ LLM Math │     │ Production   │     │ Research     │
 │          │     │          │     │ & Safety     │     │ Frontiers    │
 └──────────┘     └──────────┘     └──────────────┘     └──────────────┘
```

---

### Phase 1 — Core Math Foundations `Ch. 01 → 02 → 04 → 05`

> The mathematical language everything else is built on.

```
① MATHEMATICAL FOUNDATIONS (Ch.01)
├── Number Systems (ℕ ℤ ℚ ℝ ℂ)
├── Sets & Logic
├── Functions & Mappings
├── Σ Summation & Product Notation
├── Einstein Summation & Index Notation
└── Proof Techniques (induction, contradiction, direct)
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
② LINEAR ALGEBRA (Ch.02 + Ch.03)            ③ CALCULUS (Ch.04 + Ch.05)
├── Vectors & Spaces                        ├── Limits & Continuity
├── Matrix Operations                       ├── Derivatives & Differentiation
├── Systems of Equations                    ├── Integration & Series
├── Determinants & Rank                     ├── Partial Derivatives & Gradients
├── Eigenvalues & Eigenvectors ★            ├── Jacobians & Hessians ★
├── SVD ★                                   ├── Chain Rule → Backpropagation ★
├── PCA                                     ├── Optimality Conditions
├── Orthogonality & Norms                   └── Automatic Differentiation ★
├── Positive Definite Matrices
└── Matrix Decompositions (LU/QR/Cholesky)
```

---

### Phase 2 — Probabilistic Thinking `Ch. 06 → 07`

> How to reason about uncertainty — the foundation of all ML inference.

```
④ PROBABILITY & STATISTICS (Ch.06 + Ch.07)
├── Random Variables & Distributions
├── Joint Distributions
├── Expectation & Moments
├── Concentration Inequalities
├── Stochastic Processes
├── Markov Chains ★
├── Descriptive Statistics
├── Estimation Theory & MLE
├── Bayesian Inference ★
├── Hypothesis Testing
├── Time Series
└── Regression Analysis
```

---

### Phase 3 — Making Models Learn `Ch. 08 → 09`

> The algorithms that train every model and the theory behind loss functions.

```
⑤ OPTIMIZATION (Ch.08)                     ⑥ INFORMATION THEORY (Ch.09)
├── Convex Optimization ★                  ├── Entropy (Shannon) ★
├── Gradient Descent (SGD/Mini-batch) ★    ├── KL Divergence ★
├── Second-Order Methods (Newton/BFGS)     ├── Mutual Information ★
├── Constrained Optimization (KKT)         ├── Cross-Entropy ★
├── Stochastic Optimization ★              └── Fisher Information
├── Optimization Landscape
├── Adaptive LR (Adam / RMSProp) ★
├── Regularization (L1/L2/Dropout)
├── Hyperparameter Optimization
└── Learning Rate Schedules
```

---

### Phase 4 — Deeper Theory `Ch. 03 → 10 → 11 → 12`

> Specialized math that powers specific ML architectures.

```
⑦ NUMERICAL METHODS (Ch.10)                ⑧ GRAPH THEORY (Ch.11)
├── Floating-Point Arithmetic              ├── Graph Basics & Representations
├── Numerical Linear Algebra               ├── Graph Algorithms
├── Numerical Optimization                 ├── Spectral Graph Theory ★
├── Interpolation & Approximation          ├── Graph Neural Networks ★
└── Numerical Integration                  └── Random Graphs

⑨ FUNCTIONAL ANALYSIS (Ch.12)
├── Normed Spaces
├── Hilbert Spaces ★
└── Kernel Methods (SVM / GP) ★
```

---

### Phase 5 — ML Math in Practice `Ch. 13 → 14`

> The math that directly appears inside ML models.

```
                        ╔═══════════════════╗
                        ║  ML-SPECIFIC MATH  ║
                        ╚═══════════════════╝
                                 │
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
⑩ ML MATH CORE (Ch.13)   ⑪ DEEP LEARNING (Ch.14)   ⑫ RL (Ch.14)
├── Loss Functions ★      ├── Neural Net Math ★      ├── MDP (State/Action)
├── Activation Fns ★      ├── CNN & Convolution ★    ├── Bellman Equations ★
├── Normalization ★       ├── RNN & LSTM Math ★      ├── Policy Gradient ★
└── Sampling Methods      ├── Transformer ★          ├── Value Functions ★
                          ├── Generative (VAE/GAN)   └── Actor-Critic
                          └── Probabilistic Models
```

---

### Phase 6 — LLM Math `Ch. 15 → 16`

> Everything that makes Large Language Models work under the hood.

```
                         ╔═════════════════╗
                         ║  MATH FOR LLMs   ║
                         ╚═════════════════╝
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       ▼                          ▼                          ▼
⑬ ATTENTION & ARCH         ⑭ TRAINING AT SCALE       ⑮ DATA PIPELINE
   (Ch.15)                    (Ch.15)                    (Ch.16)
├── Tokenization            ├── Scaling Laws ★         ├── Data Format Standards
├── Embedding Space         ├── Training at Scale      ├── JSONL Generation
├── Attention Mech ★        ├── Efficient Attention    ├── Quality Checks
├── Positional Enc ★        ├── MoE & Routing          ├── Dataset Assembly
└── LM Probability ★       ├── Quantization           ├── Contamination & Dedup
                            ├── Distillation           ├── Documentation
                            └── RAG & Retrieval        └── Data Mixture Optimization
```

---

### Phase 7 — Production & Safety `Ch. 17 → 18 → 19`

> Ship models responsibly — evaluate, align, and monitor.

```
⑯ EVALUATION (Ch.17)           ⑰ ALIGNMENT & SAFETY (Ch.18)     ⑱ PRODUCTION (Ch.19)
├── Capability Benchmarks      ├── SFT Math ★                    ├── Data Versioning & Lineage
├── Calibration & Uncertainty  ├── RLHF Math ★                   ├── Experiment Tracking
├── Robustness &               ├── DPO / Preference Opt ★        ├── Feature Stores &
│   Distribution Shift         ├── Policy & Guardrails           │   Data Contracts
├── Error Analysis &           └── Human-in-the-Loop             ├── Model Serving &
│   Ablations                      & Monitoring                  │   Inference Optimization
└── A/B Testing &                                                ├── Monitoring, Drift
    Experimentation                                              │   & Retraining
                                                                 └── LLM Observability
                                                                     & Guardrails
```

---

### Phase 8 — Research Frontiers `Ch. 20 → 21 → 22 → 23 → 24 → 25`

> Advanced theory for research-level understanding.

```
⑲ FOURIER & SIGNALS (Ch.20)    ⑳ STATISTICAL LEARNING (Ch.21)   ㉑ CAUSAL INFERENCE (Ch.22)
├── Fourier Series              ├── PAC Learning                  ├── Structural Causal Models
├── Fourier Transform           ├── VC Dimension                  ├── Do-Calculus
├── DFT & FFT                  ├── Bias-Variance Tradeoff        ├── Counterfactuals
├── Convolution Theorem         ├── Generalization Bounds         └── Causal Discovery
└── Wavelets                    └── Rademacher Complexity

㉒ GAME THEORY (Ch.23)          ㉓ MEASURE THEORY (Ch.24)         ㉔ DIFF. GEOMETRY (Ch.25)
├── Nash Equilibria             ├── Sigma-Algebras                ├── Manifolds
├── Minimax Theorem             ├── Lebesgue Integration          ├── Riemannian Geometry
├── Multi-Agent Systems         ├── Probability Measure Spaces    ├── Geodesics
└── Adversarial Game Theory     └── Radon-Nikodym Theorem         └── Optimization on Manifolds
```

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║        ★  YOU ARE NOW A MATH-FOR-AI WIZARD  ★                 ║
║                                                               ║
║        ★ = Critical for ML/AI — prioritize these topics       ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

### Quick Reference — Learning Order

| Phase | Chapters | Focus |
|:------|:---------|:------|
| **Phase 1** — Core Foundations | 01 → 02 → 04 → 05 | Numbers, vectors, matrices, derivatives, gradients |
| **Phase 2** — Probabilistic Thinking | 06 → 07 | Random variables, distributions, estimation, inference |
| **Phase 3** — Making Models Learn | 08 → 09 | Optimization algorithms, information-theoretic losses |
| **Phase 4** — Deeper Theory | 03 → 10 → 11 → 12 | Advanced linear algebra, numerical methods, graphs, kernels |
| **Phase 5** — ML Math in Practice | 13 → 14 | Loss functions, activations, architecture-specific math |
| **Phase 6** — LLM Math | 15 → 16 | Attention, embeddings, scaling laws, training pipelines |
| **Phase 7** — Production & Safety | 17 → 18 → 19 | Evaluation, alignment (RLHF/DPO), MLOps |
| **Phase 8** — Research Frontiers | 20 → 21 → 22 → 23 → 24 → 25 | Fourier analysis, learning theory, causality, geometry |

---

## 📚 Chapters

### Core Mathematics

<details>
<summary><b>01 · Mathematical Foundations</b> — Number systems, sets, logic, proofs</summary>
<br>

| Topic | Description |
|:------|:------------|
| Number Systems | Natural, integer, rational, real, and complex numbers (N Z Q R C) |
| Sets & Logic | Set operations, propositional logic, quantifiers |
| Functions & Mappings | Domain, range, injectivity, surjectivity, composition |
| Summation & Product Notation | Sigma/Pi notation, index manipulation |
| Einstein Summation | Index notation used in tensor operations |
| Proof Techniques | Induction, contradiction, direct proof, contrapositive |

</details>

<details>
<summary><b>02 · Linear Algebra Basics</b> — Vectors, matrices, systems of equations</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Vectors & Spaces | Feature representations, embeddings |
| Matrix Operations | Forward propagation, transformations |
| Systems of Equations | Linear regression (normal equations) |
| Determinants | Change of variables in normalizing flows |
| Matrix Rank | Model capacity, low-rank approximations |
| Vector Spaces & Subspaces | Dimensionality, feature spaces |

</details>

<details>
<summary><b>03 · Advanced Linear Algebra</b> — Eigen decomposition, SVD, PCA</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Eigenvalues & Eigenvectors | PCA, spectral clustering, stability analysis |
| Singular Value Decomposition | Recommender systems, dimensionality reduction |
| Principal Component Analysis | Feature extraction, data compression |
| Linear Transformations | Neural network layers as transforms |
| Orthogonality & Orthonormality | Gram-Schmidt, decorrelated features |
| Matrix Norms | Regularization, operator bounds |
| Positive Definite Matrices | Covariance matrices, kernel validity |
| Matrix Decompositions | LU, QR, Cholesky — efficient solvers |

</details>

<details>
<summary><b>04 · Calculus Fundamentals</b> — Limits, derivatives, integrals, series</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Limits & Continuity | Convergence guarantees, activation smoothness |
| Derivatives & Differentiation | Gradient computation for all parameters |
| Integration | Probability densities, normalization constants |
| Series & Sequences | Taylor approximations, convergence analysis |

</details>

<details>
<summary><b>05 · Multivariate Calculus</b> — Gradients, Jacobians, backpropagation</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Partial Derivatives & Gradients | Direction of steepest descent |
| Jacobians & Hessians | Multi-output functions, second-order methods |
| Chain Rule & Backpropagation | Training every neural network |
| Optimality Conditions | Convergence criteria, saddle points |
| Automatic Differentiation | PyTorch autograd, JAX |

</details>

### Probability, Statistics & Optimization

<details>
<summary><b>06 · Probability Theory</b> — Distributions, expectations, stochastic processes</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Random Variables | Output uncertainty, stochastic models |
| Common Distributions | Gaussian, Bernoulli, Poisson — model assumptions |
| Joint Distributions | Multi-variate modeling, copulas |
| Expectation & Moments | Loss functions, feature statistics |
| Concentration Inequalities | Generalization bounds, sample complexity |
| Stochastic Processes | Time series, diffusion models |
| Markov Chains | MCMC sampling, language modeling |

</details>

<details>
<summary><b>07 · Statistics</b> — Estimation, testing, Bayesian inference, regression</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Descriptive Statistics | EDA, feature engineering |
| Estimation Theory | MLE, MAP — training as estimation |
| Hypothesis Testing | A/B testing, model comparison |
| Bayesian Inference | Posterior updates, uncertainty quantification |
| Time Series | Sequence forecasting, temporal patterns |
| Regression Analysis | Baseline models, diagnostics |

</details>

<details>
<summary><b>08 · Optimization</b> — SGD, Adam, constrained optimization, regularization</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Convex Optimization | Global guarantees, convergence proofs |
| Gradient Descent | The engine behind all training |
| Second-Order Methods | Newton, BFGS — faster convergence |
| Constrained Optimization | Lagrange multipliers, KKT conditions |
| Stochastic Optimization | SGD, mini-batch — scaling to big data |
| Optimization Landscape | Local minima, saddle points, loss surfaces |
| Adaptive Learning Rate | Adam, RMSProp, AdaGrad |
| Regularization Methods | L1/L2, Dropout, weight decay |
| Hyperparameter Optimization | Grid search, Bayesian optimization |
| Learning Rate Schedules | Warmup, cosine annealing, step decay |

</details>

### Information Theory & Numerical Methods

<details>
<summary><b>09 · Information Theory</b> — Entropy, KL divergence, cross-entropy</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Entropy | Decision tree splits, uncertainty measurement |
| KL Divergence | VAE loss, knowledge distillation |
| Mutual Information | Feature selection, InfoGAN |
| Cross-Entropy | The most common classification loss |
| Fisher Information | Efficient estimation, natural gradient |

</details>

<details>
<summary><b>10 · Numerical Methods</b> — Floating-point, stability, numerical linear algebra</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Floating-Point Arithmetic | Mixed precision training (FP16/BF16) |
| Numerical Linear Algebra | Efficient matrix operations at scale |
| Numerical Optimization | Convergence in practice |
| Interpolation & Approximation | Data augmentation, function fitting |
| Numerical Integration | Monte Carlo methods, quadrature |

</details>

### Specialized Mathematics

<details>
<summary><b>11 · Graph Theory</b> — Graph algorithms, spectral methods, GNNs</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Graph Basics | Social networks, molecular graphs |
| Graph Representations | Adjacency/Laplacian matrices |
| Graph Algorithms | Shortest path, centrality, traversal |
| Spectral Graph Theory | Community detection, graph wavelets |
| Graph Neural Networks | Message passing, GCN, GAT |
| Random Graphs | Erdos-Renyi, network analysis |

</details>

<details>
<summary><b>12 · Functional Analysis</b> — Hilbert spaces, kernel methods</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Normed Spaces | Regularization theory |
| Hilbert Spaces | RKHS, function space learning |
| Kernel Methods | SVM, Gaussian processes, kernel trick |

</details>

### ML-Specific Mathematics

<details>
<summary><b>13 · ML-Specific Math</b> — Loss functions, activations, normalization, sampling</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Loss Functions | MSE, cross-entropy, hinge, contrastive |
| Activation Functions | ReLU, GELU, sigmoid, softmax — and their gradients |
| Normalization Techniques | BatchNorm, LayerNorm, RMSNorm |
| Sampling Methods | MCMC, rejection sampling, importance sampling |

</details>

<details>
<summary><b>14 · Math for Specific Models</b> — NNs, CNNs, RNNs, Transformers, GANs, RL</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Linear Models | Regression, classification foundations |
| Neural Networks | Universal approximation, backprop math |
| Probabilistic Models | GMMs, HMMs, variational inference |
| RNN & LSTM Math | Vanishing gradients, gating mechanisms |
| Transformer Architecture | Attention is all you need — the math |
| Reinforcement Learning | Bellman equations, policy gradients |
| Generative Models | VAEs, GANs, diffusion models |
| CNN & Convolution Math | Convolution theorem, pooling, receptive fields |

</details>

### LLM Mathematics

<details>
<summary><b>15 · Math for LLMs</b> — Attention, embeddings, scaling laws, inference</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Tokenization Math | BPE, WordPiece — information-theoretic foundations |
| Embedding Space Math | Geometric properties of learned representations |
| Attention Mechanism Math | Scaled dot-product, multi-head, causal masking |
| Positional Encodings | Sinusoidal, RoPE, ALiBi |
| Language Model Probability | Next-token prediction, perplexity |
| Training at Scale | Distributed training, gradient accumulation |
| Fine-Tuning Math | LoRA, adapters, parameter-efficient methods |
| Scaling Laws | Chinchilla, compute-optimal training |
| Efficient Attention & Inference | FlashAttention, KV-cache, speculative decoding |
| Mixture of Experts & Routing | Sparse gating, load balancing |
| Quantization & Distillation | INT8/INT4, knowledge distillation |
| RAG Math & Retrieval | Retrieval-augmented generation |
| Serving & Systems Tradeoffs | Latency, throughput, batching strategies |

</details>

<details>
<summary><b>16 · LLM Training Data Pipeline</b> — Data quality, deduplication, mixture optimization</summary>
<br>

| Topic | Description |
|:------|:------------|
| Data Format Standards | JSONL, tokenized formats, schema validation |
| JSONL Generation | Efficient serialization for training |
| Quality Checks | Filtering, decontamination, toxicity |
| Full Dataset Assembly | Combining and balancing data sources |
| Contamination & Dedup Audits | Preventing benchmark leakage |
| Documentation & Governance | Data cards, provenance tracking |
| Data Mixture Optimization | Optimal domain ratios for training |

</details>

### Evaluation, Safety & Production

<details>
<summary><b>17 · Evaluation & Reliability</b> — Benchmarks, calibration, A/B testing</summary>
<br>

| Topic | Description |
|:------|:------------|
| Capability Benchmarks | MMLU, HumanEval, evaluation methodology |
| Calibration & Uncertainty | Confidence vs. accuracy alignment |
| Robustness & Distribution Shift | Out-of-distribution detection |
| Error Analysis & Ablations | Systematic debugging |
| Online Experimentation & A/B Testing | Statistical rigor in deployment |

</details>

<details>
<summary><b>18 · Alignment & Safety</b> — SFT, RLHF, DPO, red-teaming</summary>
<br>

| Topic | Description |
|:------|:------------|
| Instruction Tuning & SFT | Supervised fine-tuning mathematics |
| Preference Optimization (RLHF & DPO) | Reward modeling, Bradley-Terry, DPO objective |
| Red-Teaming & Safety Evaluations | Adversarial robustness testing |
| Policy & Guardrails | Constitutional AI, rule-based filtering |
| Human-in-the-Loop & Monitoring | Active learning, feedback loops |

</details>

<details>
<summary><b>19 · Production ML & MLOps</b> — Serving, monitoring, drift detection</summary>
<br>

| Topic | Description |
|:------|:------------|
| Data Versioning & Lineage | Reproducibility at scale |
| Experiment Tracking | MLflow, W&B — systematic experimentation |
| Feature Stores & Data Contracts | Consistent feature engineering |
| Model Serving & Inference Optimization | Latency, batching, hardware |
| Monitoring, Drift & Retraining | Detecting degradation |
| LLM Evaluation, Observability & Guardrails | LLM-specific ops |

</details>

### Advanced Theory

<details>
<summary><b>20 · Fourier Analysis & Signal Processing</b> — FFT, wavelets, convolution theorem</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Fourier Series | Periodic signal decomposition |
| Fourier Transform | Frequency domain analysis |
| DFT & FFT | Efficient spectral computation |
| Convolution Theorem | CNNs in frequency domain |
| Wavelets | Multi-resolution analysis, time-frequency |

</details>

<details>
<summary><b>21 · Statistical Learning Theory</b> — PAC learning, VC dimension, generalization</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| PAC Learning | Learnability guarantees |
| VC Dimension | Model complexity measurement |
| Bias-Variance Tradeoff | The fundamental modeling tension |
| Generalization Bounds | Why models work on unseen data |
| Rademacher Complexity | Data-dependent complexity measures |

</details>

<details>
<summary><b>22 · Causal Inference</b> — SCMs, do-calculus, counterfactuals</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Structural Causal Models | Beyond correlation |
| Do-Calculus | Interventional reasoning |
| Counterfactuals | "What if" reasoning |
| Causal Discovery | Learning causal structure from data |

</details>

<details>
<summary><b>23 · Game Theory</b> — Nash equilibria, minimax, adversarial methods</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Nash Equilibria | GAN training dynamics |
| Minimax Theorem | Adversarial robustness |
| Multi-Agent Systems | Cooperative/competitive learning |
| Adversarial Game Theory | Security and robustness |

</details>

<details>
<summary><b>24 · Measure Theory</b> — Sigma-algebras, Lebesgue integration, probability spaces</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Sigma-Algebras | Rigorous probability foundations |
| Lebesgue Integration | Expectation in continuous spaces |
| Probability Measure Spaces | Formal probability theory |
| Radon-Nikodym Theorem | Density ratios, importance sampling |

</details>

<details>
<summary><b>25 · Differential Geometry</b> — Manifolds, Riemannian geometry, geodesics</summary>
<br>

| Topic | ML Connection |
|:------|:-------------|
| Manifolds | Data lies on low-dimensional manifolds |
| Riemannian Geometry | Natural gradient, information geometry |
| Geodesics | Shortest paths in curved spaces |
| Optimization on Manifolds | Constrained optimization on curved surfaces |

</details>

---

## 📊 What's Inside Each Topic

Every topic folder follows a consistent structure:

```
📂 02-Linear-Algebra-Basics/
├── 📂 01-Vectors-and-Spaces/
│   ├── 📖 notes.md           ← Concepts, intuition, key formulas
│   ├── 🔬 theory.ipynb       ← Interactive demos with visualizations
│   └── ✏️ exercises.ipynb    ← Practice problems with solutions
├── 📂 02-Matrix-Operations/
│   ├── 📖 notes.md
│   ├── 🔬 theory.ipynb
│   └── ✏️ exercises.ipynb
└── ...
```

---

## 📖 Resources

The [`docs/`](docs/) folder contains supplementary references:

| Document | Description |
|:---------|:------------|
| [ML Math Map](docs/ML_MATH_MAP.md) | Visual guide — which math is used where in ML |
| [Notation Guide](docs/NOTATION_GUIDE.md) | Consistent notation conventions across the repo |
| [Cheatsheet](docs/CHEATSHEET.md) | Quick-reference formula sheet |
| [Interview Prep](docs/INTERVIEW_PREP.md) | Common ML math interview questions with solutions |
| [Visualization Guide](docs/VISUALIZATION_GUIDE.md) | Tips for building mathematical intuition visually |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|:-----|:--------|
| Python 3.8+ | Primary language |
| NumPy / SciPy | Numerical computing |
| Matplotlib / Seaborn / Plotly | Visualizations |
| SymPy | Symbolic mathematics |
| Jupyter Lab | Interactive notebooks |
| scikit-learn | ML examples and demos |

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

- **Fix errors** — found a typo or incorrect formula? Open a PR
- **Add exercises** — more practice problems are always valuable
- **Improve explanations** — clearer intuition helps everyone
- **Add visualizations** — interactive plots make concepts click

```bash
# Fork the repo, create a branch, make changes, then:
git checkout -b feature/your-improvement
git commit -m "Add: description of your change"
git push origin feature/your-improvement
# Open a Pull Request
```

---

## ⭐ Star History

If this repo helped you, consider giving it a star — it helps others find it too.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built for learners, researchers, and engineers who believe understanding the math makes you a better AI practitioner.**

<br>

*"In God we trust. All others must bring data."* — W. Edwards Deming

<br>

[Back to Top](#mathematics-for-ai--ml--llm)

</div>
