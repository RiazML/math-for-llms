# Mathematics for AI/ML

This repo is a structured math curriculum for AI/ML: concise notes, interactive theory notebooks, and practice exercises.

## Start here

- `docs/ML_MATH_MAP.md` - high-level topic map and suggested learning paths
- `docs/NOTATION_GUIDE.md` - notation used across the repo
- `docs/CHEATSHEET.md` - formulas at a glance
- `docs/INTERVIEW_PREP.md` - common interview questions with solutions
- `docs/VISUALIZATION_GUIDE.md` - plotting tips and visual intuition

## How it is organized

Most topic folders contain:

- `notes.md` - narrative explanation, intuition, and references
- `theory.ipynb` - interactive demonstrations and experiments
- `exercises.ipynb` - practice problems (often with worked solutions)

Recommended flow: `notes.md` -> `theory.ipynb` -> `exercises.ipynb`.

## Running locally

You can read the Markdown files directly, or run the notebooks via Jupyter.

```bash
python3 -m venv .venv  # use `python` on Windows
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
jupyter lab
```

To open a specific notebook:

```bash
jupyter lab 02-Linear-Algebra-Basics/01-Vectors-and-Spaces/theory.ipynb
```

## Repository map

<details>
<summary>Full directory tree</summary>

```text
math_for_ai/
├── 01-Mathematical-Foundations/
│   ├── 01-Number-Systems/
│   ├── 02-Sets-and-Logic/
│   ├── 03-Functions-and-Mappings/
│   ├── 04-Summation-and-Product-Notation/
│   ├── 05-Einstein-Summation-and-Index-Notation/
│   └── 06-Proof-Techniques/
├── 02-Linear-Algebra-Basics/
│   ├── 01-Vectors-and-Spaces/
│   ├── 02-Matrix-Operations/
│   ├── 03-Systems-of-Equations/
│   ├── 04-Determinants/
│   ├── 05-Matrix-Rank/
│   └── 06-Vector-Spaces-Subspaces/
├── 03-Advanced-Linear-Algebra/
│   ├── 01-Eigenvalues-and-Eigenvectors/
│   ├── 02-Singular-Value-Decomposition/
│   ├── 03-Principal-Component-Analysis/
│   ├── 04-Linear-Transformations/
│   ├── 05-Orthogonality-and-Orthonormality/
│   ├── 06-Matrix-Norms/
│   ├── 07-Positive-Definite-Matrices/
│   └── 08-Matrix-Decompositions/
├── 04-Calculus-Fundamentals/
│   ├── 01-Limits-and-Continuity/
│   ├── 02-Derivatives-and-Differentiation/
│   ├── 03-Integration/
│   └── 04-Series-and-Sequences/
├── 05-Multivariate-Calculus/
│   ├── 01-Partial-Derivatives-and-Gradients/
│   ├── 02-Jacobians-and-Hessians/
│   ├── 03-Chain-Rule-and-Backpropagation/
│   ├── 04-Optimality-Conditions/
│   └── 05-Automatic-Differentiation/
├── 06-Probability-Theory/
│   ├── 01-Introduction-and-Random-Variables/
│   ├── 02-Common-Distributions/
│   ├── 03-Joint-Distributions/
│   ├── 04-Expectation-and-Moments/
│   ├── 05-Concentration-Inequalities/
│   ├── 06-Stochastic-Processes/
│   └── 07-Markov-Chains/
├── 07-Statistics/
│   ├── 01-Descriptive-Statistics/
│   ├── 02-Estimation-Theory/
│   ├── 03-Hypothesis-Testing/
│   ├── 04-Bayesian-Inference/
│   ├── 05-Time-Series/
│   └── 06-Regression-Analysis/
├── 08-Optimization/
│   ├── 01-Convex-Optimization/
│   ├── 02-Gradient-Descent/
│   ├── 03-Second-Order-Methods/
│   ├── 04-Constrained-Optimization/
│   ├── 05-Stochastic-Optimization/
│   ├── 06-Optimization-Landscape/
│   ├── 07-Adaptive-Learning-Rate/
│   ├── 08-Regularization-Methods/
│   ├── 09-Hyperparameter-Optimization/
│   └── 10-Learning-Rate-Schedules/
├── 09-Information-Theory/
│   ├── 01-Entropy/
│   ├── 02-KL-Divergence/
│   ├── 03-Mutual-Information/
│   ├── 04-Cross-Entropy/
│   └── 05-Fisher-Information/
├── 10-Numerical-Methods/
│   ├── 01-Floating-Point-Arithmetic/
│   ├── 02-Numerical-Linear-Algebra/
│   ├── 03-Numerical-Optimization/
│   ├── 04-Interpolation-and-Approximation/
│   └── 05-Numerical-Integration/
├── 11-Graph-Theory/
│   ├── 01-Graph-Basics/
│   ├── 02-Graph-Representations/
│   ├── 03-Graph-Algorithms/
│   ├── 04-Spectral-Graph-Theory/
│   ├── 05-Graph-Neural-Networks/
│   └── 06-Random-Graphs/
├── 12-Functional-Analysis/
│   ├── 01-Normed-Spaces/
│   ├── 02-Hilbert-Spaces/
│   └── 03-Kernel-Methods/
├── 13-ML-Specific-Math/
│   ├── 01-Loss-Functions/
│   ├── 02-Activation-Functions/
│   ├── 03-Normalization-Techniques/
│   └── 04-Sampling-Methods/
├── 14-Math-for-Specific-Models/
│   ├── 01-Linear-Models/
│   ├── 02-Neural-Networks/
│   ├── 03-Probabilistic-Models/
│   ├── 04-RNN-and-LSTM-Math/
│   ├── 05-Transformer-Architecture/
│   ├── 06-Reinforcement-Learning/
│   ├── 07-Generative-Models/
│   └── 08-CNN-and-Convolution-Math/
├── 15-Math-for-LLMs/
│   ├── 01-Tokenization-Math/
│   ├── 02-Embedding-Space-Math/
│   ├── 03-Attention-Mechanism-Math/
│   ├── 04-Positional-Encodings/
│   ├── 05-Language-Model-Probability/
│   ├── 06-Training-at-Scale/
│   ├── 07-Fine-Tuning-Math/
│   ├── 08-Scaling-Laws/
│   ├── 09-Efficient-Attention-and-Inference/
│   ├── 10-Mixture-of-Experts-and-Routing/
│   ├── 11-Quantization-and-Distillation/
│   ├── 12-RAG-Math-and-Retrieval/
│   └── 13-Serving-and-Systems-Tradeoffs/
├── 16-LLM-Training-Data-Pipeline/
│   ├── 01-Data-Format-Standards/
│   ├── 02-JSONL-Generation/
│   ├── 03-Quality-Checks/
│   ├── 04-Full-Dataset-Assembly/
│   ├── 05-Contamination-and-Dedup-Audits/
│   ├── 06-Documentation-and-Governance/
│   └── 07-Data-Mixture-Optimization/
├── 17-Evaluation-and-Reliability/
│   ├── 01-Capability-Benchmarks/
│   ├── 02-Calibration-and-Uncertainty/
│   ├── 03-Robustness-and-Distribution-Shift/
│   ├── 04-Error-Analysis-and-Ablations/
│   └── 05-Online-Experimentation-and-AB-Testing/
├── 18-Alignment-and-Safety/
│   ├── 01-Instruction-Tuning-and-SFT/
│   ├── 02-Preference-Optimization-RLHF-and-DPO/
│   ├── 03-Red-Teaming-and-Safety-Evaluations/
│   ├── 04-Policy-and-Guardrails/
│   └── 05-Human-in-the-Loop-and-Monitoring/
├── 19-Production-ML-and-MLOps/
│   ├── 01-Data-Versioning-and-Lineage/
│   ├── 02-Experiment-Tracking-and-Reproducibility/
│   ├── 03-Feature-Stores-and-Data-Contracts/
│   ├── 04-Model-Serving-and-Inference-Optimization/
│   ├── 05-Monitoring-Drift-and-Retraining/
│   └── 06-LLM-Evaluation-Observability-and-Guardrails/
├── 20-Fourier-Analysis-and-Signal-Processing/
│   ├── 01-Fourier-Series/
│   ├── 02-Fourier-Transform/
│   ├── 03-Discrete-Fourier-Transform-and-FFT/
│   ├── 04-Convolution-Theorem/
│   └── 05-Wavelets/
├── 21-Statistical-Learning-Theory/
│   ├── 01-PAC-Learning/
│   ├── 02-VC-Dimension/
│   ├── 03-Bias-Variance-Tradeoff/
│   ├── 04-Generalization-Bounds/
│   └── 05-Rademacher-Complexity/
├── 22-Causal-Inference/
│   ├── 01-Structural-Causal-Models/
│   ├── 02-Do-Calculus/
│   ├── 03-Counterfactuals/
│   └── 04-Causal-Discovery/
├── 23-Game-Theory/
│   ├── 01-Nash-Equilibria/
│   ├── 02-Minimax-Theorem/
│   ├── 03-Multi-Agent-Systems/
│   └── 04-Adversarial-Game-Theory/
├── 24-Measure-Theory/
│   ├── 01-Sigma-Algebras/
│   ├── 02-Lebesgue-Integration/
│   ├── 03-Probability-Measure-Spaces/
│   └── 04-Radon-Nikodym-Theorem/
├── 25-Differential-Geometry/
│   ├── 01-Manifolds/
│   ├── 02-Riemannian-Geometry/
│   ├── 03-Geodesics/
│   └── 04-Optimization-on-Manifolds/
└── docs/
```

</details>













╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║            ███╗   ███╗ █████╗ ████████╗██╗  ██╗    ███████╗ ██████╗ ██████╗             ║
║            ████╗ ████║██╔══██╗╚══██╔══╝██║  ██║    ██╔════╝██╔═══██╗██╔══██╗            ║
║            ██╔████╔██║███████║   ██║   ███████║    █████╗  ██║   ██║██████╔╝            ║
║            ██║╚██╔╝██║██╔══██║   ██║   ██╔══██║    ██╔══╝  ██║   ██║██╔══██╗            ║
║            ██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║    ██║     ╚██████╔╝██║  ██║            ║
║            ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝    ╚═╝      ╚═════╝ ╚═╝  ╚═╝            ║
║                                                                                          ║
║                    ━━━  A I   R O A D M A P  ·  2 5  D O M A I N S  ━━━                 ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

                                         START HERE
                                             │
                          ┌──────────────────┴──────────────────┐
                          │                                      │
                   PURE MATH CORE                        APPLIED MATH CORE
                          │                                      │
        ┌─────────────────┼─────────────────┐         ┌─────────┴─────────┐
        │                 │                 │         │                   │
  ① FOUNDATIONS    ② LINEAR ALGEBRA   ③ CALCULUS   ④ PROBABILITY    ⑤ OPTIMIZATION
        │                 │                 │         │                   │
        │                 │                 │         │                   │


┌────────────────────────────────────────────────────────────────────────────────────────┐
│  ① MATHEMATICAL FOUNDATIONS                                                            │
│  ──────────────────────────                                                            │
│  ├── Number Systems (ℕ ℤ ℚ ℝ ℂ)                                                        │
│  ├── Sets & Logic                                                                      │
│  ├── Functions & Mappings                                                              │
│  ├── Σ Summation & Product Notation                                                    │
│  ├── Einstein Summation & Index Notation                                               │
│  └── Proof Techniques (induction, contradiction, direct)                               │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼

┌───────────────────────────────────────┐   ┌───────────────────────────────────────────┐
│  ② LINEAR ALGEBRA  (basics → advanced)│   │  ③ CALCULUS  (single → multivariate)      │
│  ─────────────────────────────────────│   │  ─────────────────────────────────────────│
│  ├── Vectors & Spaces                 │   │  ├── Limits & Continuity                  │
│  ├── Matrix Operations                │   │  ├── Derivatives & Differentiation        │
│  ├── Systems of Equations             │   │  ├── Integration & Series                 │
│  ├── Determinants & Rank              │   │  ├── Partial Derivatives & Gradients      │
│  ├── Eigenvalues & Eigenvectors ★     │   │  ├── Jacobians & Hessians ★               │
│  ├── SVD  ★                           │   │  ├── Chain Rule → Backpropagation ★       │
│  ├── PCA                              │   │  ├── Optimality Conditions                │
│  ├── Orthogonality & Norms            │   │  └── Automatic Differentiation ★          │
│  ├── Positive Definite Matrices       │   └───────────────────────────────────────────┘
│  └── Matrix Decompositions (LU/QR/Chol│
└───────────────────────────────────────┘
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼

┌───────────────────────────────────────┐   ┌───────────────────────────────────────────┐
│  ④ PROBABILITY & STATISTICS           │   │  ⑤ OPTIMIZATION                           │
│  ─────────────────────────────────────│   │  ─────────────────────────────────────────│
│  ├── Random Variables & Distributions │   │  ├── Convex Optimization ★                │
│  ├── Joint Distributions              │   │  ├── Gradient Descent (SGD/Mini-batch) ★  │
│  ├── Expectation & Moments            │   │  ├── Second-Order Methods (Newton/BFGS)   │
│  ├── Concentration Inequalities       │   │  ├── Constrained Optimization (KKT)       │
│  ├── Stochastic Processes             │   │  ├── Stochastic Optimization ★            │
│  ├── Markov Chains ★                  │   │  ├── Optimization Landscape               │
│  ├── Bayesian Inference ★             │   │  ├── Adaptive LR  (Adam / RMSProp) ★      │
│  ├── Estimation Theory & MLE          │   │  ├── Regularization (L1/L2/Dropout)       │
│  ├── Hypothesis Testing               │   │  ├── Hyperparameter Optimization          │
│  └── Regression Analysis              │   │  └── Learning Rate Schedules              │
└───────────────────────────────────────┘   └───────────────────────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼

┌───────────────────────────────────────┐   ┌───────────────────────────────────────────┐
│  ⑥ INFORMATION THEORY                 │   │  ⑦ NUMERICAL METHODS                      │
│  ─────────────────────────────────────│   │  ─────────────────────────────────────────│
│  ├── Entropy (Shannon) ★              │   │  ├── Floating-Point Arithmetic            │
│  ├── KL Divergence ★                  │   │  ├── Numerical Linear Algebra             │
│  ├── Mutual Information ★             │   │  ├── Numerical Optimization               │
│  ├── Cross-Entropy ★                  │   │  ├── Interpolation & Approximation        │
│  └── Fisher Information               │   │  └── Numerical Integration                │
└───────────────────────────────────────┘   └───────────────────────────────────────────┘
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼

┌───────────────────────────────────────┐   ┌───────────────────────────────────────────┐
│  ⑧ GRAPH THEORY                        │   │  ⑨ FUNCTIONAL ANALYSIS                    │
│  ─────────────────────────────────────│   │  ─────────────────────────────────────────│
│  ├── Graph Basics & Representations   │   │  ├── Normed Spaces                        │
│  ├── Graph Algorithms                 │   │  ├── Hilbert Spaces ★                     │
│  ├── Spectral Graph Theory ★          │   │  └── Kernel Methods (SVM / GP) ★          │
│  ├── Graph Neural Networks ★          │   └───────────────────────────────────────────┘
│  └── Random Graphs                    │
└───────────────────────────────────────┘
                                         │
                                         ▼
                              ╔═══════════════════╗
                              ║  ML-SPECIFIC MATH  ║
                              ╚═══════════════════╝
                                         │
             ┌───────────────────────────┼───────────────────────────┐
             ▼                           ▼                           ▼

┌─────────────────────────┐  ┌──────────────────────────┐  ┌─────────────────────────────┐
│  ⑩ ML MATH CORE          │  │  ⑪ DEEP LEARNING MATH     │  │  ⑫ REINFORCEMENT LEARNING   │
│  ───────────────────────│  │  ────────────────────────│  │  ───────────────────────────│
│  ├── Loss Functions ★   │  │  ├── Neural Net Math ★    │  │  ├── MDP (State/Action/Rew) │
│  ├── Activation Fns ★   │  │  ├── CNN & Convolution ★  │  │  ├── Bellman Equations ★    │
│  ├── Normalization ★    │  │  ├── RNN & LSTM Math ★    │  │  ├── Policy Gradient ★      │
│  └── Sampling Methods   │  │  ├── Transformer ★        │  │  ├── Value Functions ★      │
└─────────────────────────┘  │  ├── Generative (VAE/GAN) │  │  └── Actor-Critic Methods   │
                             │  └── Probabilistic Models  │  └─────────────────────────────┘
                             └──────────────────────────┘
                                         │
                                         ▼
                                ╔═════════════════╗
                                ║  MATH FOR LLMs   ║
                                ╚═════════════════╝
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         ▼                               ▼                               ▼

┌──────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────────┐
│  ⑬ ATTENTION & ARCH   │   │  ⑭ TRAINING AT SCALE      │   │  ⑮ ALIGNMENT & FINE-TUNING   │
│  ────────────────────│   │  ────────────────────────│   │  ────────────────────────────│
│  ├── Tokenization     │   │  ├── Scaling Laws ★       │   │  ├── SFT Math ★              │
│  ├── Embedding Space  │   │  ├── Training Dynamics    │   │  ├── RLHF Math ★             │
│  ├── Attention Mech ★ │   │  ├── Efficient Attention  │   │  ├── DPO / Preference Opt ★  │
│  ├── Positional Enc ★ │   │  ├── MoE & Routing        │   │  ├── Constitutional AI       │
│  └── LM Probability ★ │   │  ├── Quantization         │   │  └── Red-Teaming & Safety    │
└──────────────────────┘   │  ├── Distillation          │   └──────────────────────────────┘
                           │  └── RAG & Retrieval        │
                           └──────────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼

┌───────────────────────────────────────┐   ┌───────────────────────────────────────────┐
│  ⑯ ADVANCED THEORY                    │   │  ⑰ PRODUCTION & EVALUATION                │
│  ─────────────────────────────────────│   │  ─────────────────────────────────────────│
│  ├── PAC Learning & VC Dimension      │   │  ├── Capability Benchmarks                │
│  ├── Bias-Variance Tradeoff           │   │  ├── Calibration & Uncertainty            │
│  ├── Generalization Bounds            │   │  ├── Robustness & Distribution Shift      │
│  ├── Rademacher Complexity            │   │  ├── Error Analysis & Ablations           │
│  ├── Fourier & Signal Processing      │   │  ├── A/B Testing & Experimentation        │
│  ├── Causal Inference (Do-Calculus)   │   │  ├── Drift Monitoring & Retraining        │
│  ├── Game Theory & Nash Equilibria    │   │  └── LLM Observability & Guardrails       │
│  ├── Measure Theory                   │   └───────────────────────────────────────────┘
│  └── Differential Geometry            │
└───────────────────────────────────────┘
                                         │
                                         ▼
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║                         🎯  YOU ARE NOW A MATH-FOR-AI WIZARD  🎯                         ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

   LEGEND:  ★ = Must-Know for Practitioners   │  → = Feeds Into   │  ╔╗ = Major Milestone
   ─────────────────────────────────────────────────────────────────────────────────────
   TRACK A  [Foundations → Lin.Alg → Calculus → Prob → Optimization]   ← start here
   TRACK B  [+ Info Theory → ML Math → Deep Learning]                  ← practitioners
   TRACK C  [+ LLMs → Scaling → Alignment]                             ← LLM engineers
   TRACK D  [+ Adv. Theory → Game Theory → Causal Inference]           ← researchers