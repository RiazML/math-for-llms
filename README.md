<div align="center">

# Mathematics for AI / ML / LLM

### The Complete Math Foundation You Need to Master AI

[![GitHub stars](https://img.shields.io/github/stars/GizzZmo/math-for-llms?style=for-the-badge&logo=github&color=yellow)](https://github.com/GizzZmo/math-for-llms/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/GizzZmo/math-for-llms?style=for-the-badge&logo=github&color=blue)](https://github.com/GizzZmo/math-for-llms/network/members)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Notebooks](https://img.shields.io/badge/Jupyter_Notebooks-229-orange?style=for-the-badge&logo=jupyter)](.)
[![Chapters](https://img.shields.io/badge/Chapters-25-brightgreen?style=for-the-badge&logo=bookstack)](.)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](CONTRIBUTING.md)

**A structured, open-source curriculum covering 25 domains of mathematics essential for AI, Machine Learning, and Large Language Models — from foundational concepts to cutting-edge research.**

[Get Started](#-quick-start) · [Roadmap](#-learning-roadmap) · [Chapters](#-chapters) · [Applications](#financial--causal-applications) · [Contributing](#-contributing)

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

## Financial & Causal Applications

Graph Neural Networks (Chapter 11) and Causal Inference (Chapter 22) come together for high-stakes financial applications where relational structure and interventional effects both matter.

### Default Contagion and Credit Risk
- Construct firm or counterparty graphs with accounting features on nodes and payment / supply-chain / interbank exposures on edges.
- GATv2 (or GAT) layers learn dynamic attention so that a node’s embedding mixes its own fundamentals with *k*-hop spillover risk.
- Downstream PD classifiers or risk scores improve over i.i.d. models; attention weights and GraphSHAP provide the explanations required by the EU AI Act (high-risk systems) and GDPR Article 22.

See the full treatment in [Chapter 11 — Graph Neural Networks §12.6](11-Graph-Theory/05-Graph-Neural-Networks/notes.md#126-financial-networks-credit-risk-and-default-contagion).

### Causal Policy Evaluation
- Use the same GNN embeddings as rich covariates inside Double/Debiased Machine Learning estimators.
- Estimate heterogeneous treatment effects of liquidity support, credit-line adjustments, or regulatory interventions while controlling for selection bias induced by network position.
- Cross-reference: [Chapter 22 — Structural Causal Models](22-Causal-Inference/01-Structural-Causal-Models/notes.md).

### Why It Matters
These techniques turn relational data into actionable, auditable risk signals and let institutions quantify the real incremental effect of interventions rather than mere correlations.

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/GizzZmo/math-for-llms.git
cd math-for-llms

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
 ┌─────────┐     ┌──────────────┐     ┌──────────┐     ┌──────────────┐
 │ Phase 1  │ ──▶ │   Phase 2   │ ──▶ │ Phase 3  │ ──▶ │   Phase 4    │
 │ Core     │     │ Probability │     │ Learning │     │ Deeper       │
 │ Math     │     │ & Stats     │     │ Engines  │     │ Theory       │
 └─────────┘     └──────────────┘     └──────────┘     └──────────────┘
                                                              │
      ┌───────────────────────────────────────────────────────────────────────────┘
      ▼
 ┌──────────┐     ┌──────────┐     ┌──────────────┐     ┌──────────────┐
 │ Phase 5  │ ──▶ │ Phase 6  │ ──▶ │   Phase 7    │ ──▶ │   Phase 8    │
 │ ML Math  │     │ LLM Math │     │ Production   │     │ Research     │
 │          │     │          │     │ & Safety     │     │ Frontiers    │
 └──────────┘     └──────────┘     └──────────────┘     └──────────────┘
```

### Quick Reference — Learning Order

| Phase                                | Chapters                    | Focus                                                       |
| :----------------------------------- | :-------------------------- | :---------------------------------------------------------- |
| **Phase 1** — Core Foundations       | 01 → 02 → 04 → 05           | Numbers, vectors, matrices, derivatives, gradients          |
| **Phase 2** — Probabilistic Thinking | 06 → 07                     | Random variables, distributions, estimation, inference      |
| **Phase 3** — Making Models Learn    | 08 → 09                     | Optimization algorithms, information-theoretic losses       |
| **Phase 4** — Deeper Theory          | 03 → 10 → 11 → 12           | Advanced linear algebra, numerical methods, graphs, kernels |
| **Phase 5** — ML Math in Practice    | 13 → 14                     | Loss functions, activations, architecture-specific math     |
| **Phase 6** — LLM Math               | 15 → 16                     | Attention, embeddings, scaling laws, training pipelines     |
| **Phase 7** — Production & Safety    | 17 → 18 → 19                | Evaluation, alignment (RLHF/DPO), MLOps                     |
| **Phase 8** — Research Frontiers     | 20 → 21 → 22 → 23 → 24 → 25 | Fourier analysis, learning theory, causality, geometry      |

---

## 📚 Chapters

The repository is organized into 25 numbered chapters. Each chapter contains self-contained sections with `notes.md`, `theory.ipynb`, and `exercises.ipynb`.

<details>
<summary><b>Core Foundations (01–05)</b></summary>

- **01 Mathematical Foundations** — Number systems, sets, logic, functions, summation, Einstein notation, proofs
- **02 Linear Algebra Basics** — Vectors, matrices, systems of equations, rank, determinants
- **03 Advanced Linear Algebra** — Eigenvalues, SVD, PCA, positive definite matrices, decompositions
- **04 Calculus Fundamentals** — Limits, derivatives, integration, series
- **05 Multivariate Calculus** — Gradients, Jacobians, Hessians, chain rule, automatic differentiation

</details>

<details>
<summary><b>Probability, Statistics & Optimization (06–09)</b></summary>

- **06 Probability Theory** — Random variables, distributions, expectation, concentration, stochastic processes, Markov chains
- **07 Statistics** — Estimation, MLE, Bayesian inference, hypothesis testing, time series, regression
- **08 Optimization** — Convex optimization, gradient descent, second-order methods, Adam, regularization
- **09 Information Theory** — Entropy, KL divergence, mutual information, cross-entropy

</details>

<details>
<summary><b>Deeper Theory & Graph Methods (10–12)</b></summary>

- **10 Numerical Methods** — Floating-point, numerical linear algebra, interpolation, integration
- **11 Graph Theory** — Basics, representations, algorithms, spectral theory, **Graph Neural Networks**, random graphs
- **12 Functional Analysis** — Normed spaces, Hilbert spaces, kernel methods

</details>

<details>
<summary><b>ML & LLM Mathematics (13–16)</b></summary>

- **13 ML-Specific Math** — Loss functions, activations, normalization, sampling
- **14 Math for Specific Models** — Neural nets, CNNs, RNNs/LSTMs, Transformers, generative models, RL
- **15 Math for LLMs** — Attention, embeddings, positional encodings, scaling laws, efficient attention, MoE
- **16 LLM Training Data Pipeline** — Formats, quality, deduplication, mixture optimization

</details>

<details>
<summary><b>Production, Safety & Research Frontiers (17–25)</b></summary>

- **17 Evaluation and Reliability** — Benchmarks, calibration, robustness, A/B testing
- **18 Alignment and Safety** — SFT, RLHF, DPO, guardrails
- **19 Production ML and MLOps** — Versioning, tracking, serving, drift, observability
- **20 Fourier Analysis and Signal Processing**
- **21 Statistical Learning Theory** — PAC, VC dimension, bias-variance, generalization
- **22 Causal Inference** — SCMs, do-calculus, counterfactuals, discovery
- **23 Game Theory**
- **24 Measure Theory**
- **25 Differential Geometry**

</details>

---

## Resources

- [Notation Guide](docs/NOTATION_GUIDE.md) — authoritative notation used throughout
- [Cheatsheet](docs/CHEATSHEET.md) — every formula that actually matters
- [ML Math Map](docs/ML_MATH_MAP.md) — concept → model / paper mapping
- [Contributing Guide](CONTRIBUTING.md) — quality standards for notes, notebooks, and exercises
- [Code of Conduct](CODE_OF_CONDUCT.md)

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. The quality bar is high: graduate-level notes, interactive notebooks, and graded exercises.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This curriculum builds on the excellent structure and content originally developed in the RiazML/math-for-llms repository and has been extended with additional applications in financial networks and causal inference.
