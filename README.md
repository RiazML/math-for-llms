# 🧮 Mathematics for AI/ML Mastery

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-1.20+-green.svg)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A comprehensive, structured guide to mastering the mathematics required for AI/ML, combining the best content from DeepLearning.AI, Khan Academy, MIT OCW, Stanford CS229, 3Blue1Brown, and StatQuest.

## 📋 Table of Contents

- [Overview](#-overview)
- [Learning Roadmap](#-learning-roadmap)
- [Repository Structure](#-repository-structure)
- [How to Use This Repository](#-how-to-use-this-repository)
- [Prerequisites](#-prerequisites)
- [Learning Guide](#-learning-guide)
- [Quick Reference](#-quick-reference)
- [Resources](#-resources)
- [Contributing](#-contributing)

---

## 🎯 Overview

This repository provides a **complete mathematical foundation** for understanding and implementing AI/ML algorithms. Each topic follows a **3-File System**:

| File           | Purpose                                     |
| -------------- | ------------------------------------------- |
| `README.md`    | Theory, intuition, visualizations, formulas |
| `examples.py`  | Working Python/NumPy implementations        |
| `exercises.py` | Practice problems with solutions            |

### Who Is This For?

- 🎓 **Students** preparing for ML/AI careers
- 💼 **ML Engineers** wanting deeper mathematical understanding
- 📊 **Data Scientists** strengthening foundations
- 🎤 **Interview Candidates** preparing for technical interviews
- 🔄 **Career Switchers** transitioning to AI/ML

### After Completing This Repository, You Will Be Able To:

✅ Read ML papers and understand the mathematics  
✅ Derive backpropagation from scratch  
✅ Understand optimization algorithms deeply  
✅ Explain probabilistic models mathematically  
✅ Answer ML math interview questions confidently

---

## 🗺️ Learning Roadmap

```
                                    ┌─────────────────────────────────────────┐
                                    │     MATHEMATICS FOR AI/ML MASTERY       │
                                    └─────────────────────────────────────────┘
                                                        │
                    ┌───────────────────────────────────┼───────────────────────────────────┐
                    │                                   │                                   │
                    ▼                                   ▼                                   ▼
         ┌───────────────────┐               ┌───────────────────┐               ┌───────────────────┐
         │     FOUNDATIONS   │──────────────▶│  CORE LINEAR ALG  │──────────────▶│     CALCULUS      │
         └───────────────────┘               └───────────────────┘               └───────────────────┘
                    │                                   │                                   │
                    │  • Number Systems                 │  • Eigenvalues                    │  • Single Variable
                    │  • Sets & Logic                   │  • SVD                            │  • Multivariable
                    │  • Vectors & Matrices             │  • PCA                            │  • Gradients
                    │  • Linear Transformations         │  • Matrix Factorizations          │  • Jacobian/Hessian
                    │                                   │                                   │
                    └───────────────────────────────────┴───────────────────────────────────┘
                                                        │
                    ┌───────────────────────────────────┼───────────────────────────────────┐
                    │                                   │                                   │
                    ▼                                   ▼                                   ▼
         ┌───────────────────┐               ┌───────────────────┐               ┌───────────────────┐
         │     PROBABILITY   │               │   OPTIMIZATION    │               │ INFORMATION THEORY│
         └───────────────────┘               └───────────────────┘               └───────────────────┘
                    │                                   │                                   │
                    │  • Probability Theory             │  • Gradient Descent               │  • Entropy
                    │  • Bayes Theorem                  │  • SGD, Adam                      │  • Cross-Entropy
                    │  • Distributions                  │  • Convex Optimization            │  • KL Divergence
                    │  • Statistical Inference          │  • Constrained Opt                │  • Mutual Information
                    │                                   │                                   │
                    └───────────────────────────────────┴───────────────────────────────────┘
                                                        │
                    ┌───────────────────────────────────┼───────────────────────────────────┐
                    │                                   │                                   │
                    ▼                                   ▼                                   ▼
         ┌───────────────────┐               ┌───────────────────┐               ┌───────────────────┐
         │  ADVANCED TOPICS  │               │  ML APPLICATIONS  │               │      COMPLETE     │
         └───────────────────┘               └───────────────────┘               └───────────────────┘
                    │                                   │
                    │  • Numerical Methods              │  • Backpropagation Math
                    │  • Graphical Models               │  • Attention/Transformers
                    │  • MCMC                           │  • Model-Specific Math
                    │  • Gaussian Processes             │
```

---

## 📁 Repository Structure

```
math_for_ai/
│
├── README.md                          # This file
├── CONTRIBUTING.md                    # How to contribute
├── requirements.txt                   # Python dependencies
│
├── docs/                              # Quick reference documents
│   ├── CHEATSHEET.md                  # All formulas in one page
│   ├── NOTATION_GUIDE.md              # Mathematical notation guide
│   ├── ML_MATH_MAP.md                 # Which math is used where in ML
│   ├── INTERVIEW_PREP.md              # Common math interview questions
│   └── VISUALIZATION_GUIDE.md         # How to visualize concepts
│
├── notebooks/                         # Interactive Jupyter notebooks
│   ├── 01_linear_algebra_visualizations.ipynb
│   ├── 02_calculus_intuition.ipynb
│   ├── 03_probability_simulations.ipynb
│   └── ...
│
 ├── 01-Mathematical-Foundations/
│   ├── 01-Number-Systems/
│   │   ├── README.md
│   │   ├── examples.py
│   │   └── exercises.py
│   ├── 02-Sets-and-Logic/
│   ├── 03-Functions-and-Mappings/
│   ├── 04-Summation-and-Product-Notation/
│   └── 05-Proof-Techniques/
│
 ├── 02-Linear-Algebra-Basics/
│   ├── 01-Vectors-and-Spaces/
│   ├── 02-Matrix-Operations/
│   ├── 03-Linear-Transformations/
│   ├── 04-Systems-of-Linear-Equations/
│   ├── 05-Gaussian-Elimination/
│   └── 06-Vector-Spaces-Subspaces/
│
 ├── 03-Advanced-Linear-Algebra/
│   ├── 01-Eigenvalues-Eigenvectors/
│   ├── 02-Diagonalization/
│   ├── 03-Singular-Value-Decomposition/
│   ├── 04-Principal-Component-Analysis/
│   ├── 05-Matrix-Factorizations/
│   ├── 06-Positive-Definite-Matrices/
│   ├── 07-Orthogonality-Projections/
│   └── 08-Least-Squares/
│
 ├── 04-Single-Variable-Calculus/
│   ├── 01-Limits-and-Continuity/
│   ├── 02-Derivatives/
│   ├── 03-Chain-Rule/
│   ├── 04-Optimization-1D/
│   ├── 05-Integration/
│   └── 06-Taylor-Series/
│
 ├── 05-Multivariable-Calculus/
│   ├── 01-Partial-Derivatives/
│   ├── 02-Gradients/
│   ├── 03-Jacobian-Matrix/
│   ├── 04-Hessian-Matrix/
│   ├── 05-Directional-Derivatives/
│   ├── 06-Multivariable-Chain-Rule/
│   ├── 07-Vector-Calculus/
│   └── 08-Lagrange-Multipliers/
│
 ├── 06-Probability-Theory/
│   ├── 01-Probability-Basics/
│   ├── 02-Conditional-Probability/
│   ├── 03-Bayes-Theorem/
│   ├── 04-Random-Variables/
│   ├── 05-Probability-Distributions/
│   ├── 06-Joint-Distributions/
│   ├── 07-Expectation-Variance/
│   ├── 08-Covariance-Correlation/
│   └── 09-Law-of-Large-Numbers/
│
 ├── 07-Statistical-Inference/
│   ├── 01-Parameter-Estimation/
│   ├── 02-Maximum-Likelihood-Estimation/
│   ├── 03-Maximum-A-Posteriori/
│   ├── 04-Confidence-Intervals/
│   ├── 05-Hypothesis-Testing/
│   ├── 06-Bayesian-Inference/
│   └── 07-Sampling-Methods/
│
 ├── 08-Optimization/
│   ├── 01-Convex-Sets-Functions/
│   ├── 02-Gradient-Descent/
│   ├── 03-Stochastic-Gradient-Descent/
│   ├── 04-Momentum-Methods/
│   ├── 05-Adam-and-Variants/
│   ├── 06-Learning-Rate-Schedules/
│   ├── 07-Constrained-Optimization/
│   ├── 08-Convex-Optimization/
│   └── 09-Second-Order-Methods/
│
 ├── 09-Information-Theory/
│   ├── 01-Entropy/
│   ├── 02-Cross-Entropy/
│   ├── 03-KL-Divergence/
│   ├── 04-Mutual-Information/
│   └── 05-Information-Gain/
│
 ├── 10-Numerical-Methods/
│   ├── 01-Numerical-Stability/
│   ├── 02-Floating-Point-Arithmetic/
│   ├── 03-Numerical-Differentiation/
│   ├── 04-Numerical-Integration/
│   └── 05-Iterative-Methods/
│
 ├── 11-Probabilistic-Graphical-Models/
│   ├── 01-Bayesian-Networks/
│   ├── 02-Markov-Random-Fields/
│   ├── 03-Hidden-Markov-Models/
│   └── 04-Inference-Algorithms/
│
 ├── 12-Advanced-Probability/
│   ├── 01-Markov-Chains/
│   ├── 02-Monte-Carlo-Methods/
│   ├── 03-MCMC/
│   ├── 04-Variational-Inference/
│   └── 05-Gaussian-Processes/
│
 ├── 13-Math-for-Neural-Networks/
│   ├── 01-Backpropagation-Math/
│   ├── 02-Activation-Functions/
│   ├── 03-Loss-Functions/
│   ├── 04-Regularization-Theory/
│   ├── 05-Batch-Normalization-Math/
│   ├── 06-Attention-Mechanism-Math/
│   └── 07-Transformer-Math/
│
 └── 14-Math-for-Specific-Models/
    ├── 01-Linear-Regression-Theory/
    ├── 02-Logistic-Regression-Theory/
    ├── 03-SVM-Math/
    ├── 04-Decision-Tree-Math/
    ├── 05-Clustering-Math/
    ├── 06-Dimensionality-Reduction/
    └── 07-Recommender-Systems-Math/
```

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
```

### Recommended Study Approach

1. **Read the README.md** first to understand the theory
2. **Run examples.py** to see concepts in action
3. **Complete exercises.py** without looking at solutions
4. **Check solutions** and understand any mistakes
5. **Review the notebook** for visualizations
6. **Move to next topic** only when comfortable

### For Each Topic

```bash
# Navigate to topic
cd 02-Linear-Algebra-Basics/01-Vectors-and-Spaces/

# Read theory
cat README.md

# Run examples
python examples.py

# Practice exercises
python exercises.py
```

---

## 📚 Prerequisites

Before starting, you should be comfortable with:

- **Basic Algebra**: Variables, equations, polynomials
- **High School Math**: Basic geometry, trigonometry
- **Python Basics**: Functions, loops, lists
- **NumPy Basics**: Array operations (helpful but not required)

If you need to brush up:

- [Khan Academy Algebra](https://www.khanacademy.org/math/algebra)
- [Khan Academy Precalculus](https://www.khanacademy.org/math/precalculus)

---

## 📖 Learning Guide

### Foundations

**Goal**: Build mathematical language and basic linear algebra intuition

Start here to establish notation, proof techniques, and vector/matrix fundamentals.

| Topic          | ML Application                       |
| -------------- | ------------------------------------ |
| Number Systems | Numerical precision in deep learning |
| Sets and Logic | Boolean operations in decision trees |
| Functions      | Activation functions, loss functions |
| Vectors        | Feature vectors, embeddings          |
| Matrices       | Weight matrices, transformations     |

### Core Linear Algebra

**Goal**: Master the linear algebra that powers ML

| Topic       | ML Application                   |
| ----------- | -------------------------------- |
| Eigenvalues | PCA, spectral clustering         |
| SVD         | Recommender systems, compression |
| Projections | Least squares, linear regression |

### Calculus

**Goal**: Understand how neural networks learn

| Topic       | ML Application            |
| ----------- | ------------------------- |
| Derivatives | Gradient computation      |
| Chain Rule  | Backpropagation           |
| Gradients   | Optimization direction    |
| Hessian     | Second-order optimization |

### Probability & Statistics

**Goal**: Reason about uncertainty in ML

| Topic         | ML Application           |
| ------------- | ------------------------ |
| Bayes Theorem | Bayesian ML, Naive Bayes |
| Distributions | Generative models, VAEs  |
| MLE/MAP       | Parameter estimation     |

### Optimization

**Goal**: Master how models are trained

| Topic            | ML Application             |
| ---------------- | -------------------------- |
| Gradient Descent | Training neural networks   |
| Adam             | State-of-the-art optimizer |
| Constrained Opt  | SVMs, regularization       |

### Information Theory

**Goal**: Understand loss functions and model comparison

| Topic         | ML Application               |
| ------------- | ---------------------------- |
| Cross-Entropy | Classification loss          |
| KL Divergence | VAEs, knowledge distillation |

### Advanced Topics

**Goal**: Prepare for research-level ML

| Topic              | ML Application             |
| ------------------ | -------------------------- |
| MCMC               | Bayesian inference         |
| Gaussian Processes | Uncertainty quantification |
| HMMs               | Sequence modeling          |

### ML Applications

**Goal**: Connect all math to real models

| Topic            | ML Application               |
| ---------------- | ---------------------------- |
| Backprop Math    | Implementing neural networks |
| Transformer Math | Understanding attention      |

---

## 📋 Quick Reference

| Document                                           | Description                    |
| -------------------------------------------------- | ------------------------------ |
| [Cheatsheet](docs/CHEATSHEET.md)                   | All formulas in one page       |
| [Notation Guide](docs/NOTATION_GUIDE.md)           | Mathematical symbols explained |
| [ML Math Map](docs/ML_MATH_MAP.md)                 | Which math is used where       |
| [Interview Prep](docs/INTERVIEW_PREP.md)           | Common interview questions     |
| [Visualization Guide](docs/VISUALIZATION_GUIDE.md) | How to visualize concepts      |

---

## 🔗 Resources

### Video Courses

- 📺 [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
- 📺 [3Blue1Brown - Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr)
- 📺 [MIT 18.06 Linear Algebra (Gilbert Strang)](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/)
- 📺 [Stanford CS229 - Machine Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rMiGQp3WXShtMGgzqpfVfbU)
- 📺 [StatQuest with Josh Starmer](https://www.youtube.com/c/joshstarmer)

### Books

- 📖 [Mathematics for Machine Learning](https://mml-book.github.io/) (Deisenroth et al.)
- 📖 [Deep Learning](https://www.deeplearningbook.org/) (Goodfellow et al.) - Part I
- 📖 [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) (Bishop)

### Interactive Tools

- 🔧 [Desmos Graphing Calculator](https://www.desmos.com/calculator)
- 🔧 [GeoGebra](https://www.geogebra.org/)
- 🔧 [Wolfram Alpha](https://www.wolframalpha.com/)

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Ways to Contribute

- Fix typos or errors
- Add more examples
- Improve explanations
- Add visualizations
- Create new exercises

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Happy Learning! 🎓**

_"The only way to learn mathematics is to do mathematics."_ - Paul Halmos

</div>
