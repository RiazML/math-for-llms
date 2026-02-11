# Hyperparameter Optimization

> **Navigation**: [← 08-Regularization-Methods](../08-Regularization-Methods/) | [Optimization](../) | [09-Information-Theory →](../../09-Information-Theory/)

**Files in this section:**
- [theory.ipynb](theory.ipynb) - 12 worked examples
- [exercises.ipynb](exercises.ipynb) - 10 practice problems with solutions

---

## Introduction

Hyperparameter optimization (HPO) is the process of finding the best configuration for machine learning algorithms. Unlike model parameters learned during training, hyperparameters must be set before training begins. Good hyperparameters can mean the difference between a mediocre model and state-of-the-art performance.

```
The Hyperparameter Optimization Challenge:
═══════════════════════════════════════════════════════════

  ┌─────────────┐    ┌──────────────────┐    ┌─────────────┐
  │ Hyperparams │ ─▶ │     Training     │ ─▶ │ Validation  │
  │  (λ)        │    │   Algorithm A    │    │    Loss     │
  └─────────────┘    └──────────────────┘    └─────────────┘
        ▲                                          │
        │                                          │
        └──────────────────────────────────────────┘
                    Optimize this loop!
                    
  Challenge: Each evaluation is EXPENSIVE (full training)
```

## Prerequisites

- Probability distributions
- Optimization basics
- Cross-validation
- Basic machine learning concepts

## Learning Objectives

- Master hyperparameter search strategies
- Understand Bayesian optimization principles
- Implement efficient hyperparameter tuning
- Apply early stopping techniques (Hyperband, ASHA)

## Prerequisites

- Probability distributions
- Optimization basics
- Cross-validation
- Basic machine learning concepts

---

## 1. Introduction to Hyperparameters

### What are Hyperparameters?

**Model Parameters vs Hyperparameters:**

```
Parameters (learned):
- Neural network weights
- Regression coefficients
- SVM support vectors

Hyperparameters (set before training):
- Learning rate
- Number of hidden units
- Regularization strength
- Batch size
- Number of trees in random forest
```

### The Hyperparameter Optimization Problem

$$\lambda^* = \arg\min_{\lambda \in \Lambda} \mathcal{L}_{val}(\mathcal{A}(\lambda, \mathcal{D}_{train}), \mathcal{D}_{val})$$

Where:

- $\lambda$ = hyperparameters
- $\Lambda$ = hyperparameter space
- $\mathcal{A}$ = learning algorithm
- $\mathcal{L}_{val}$ = validation loss

### Challenges

```
Hyperparameter Optimization Challenges:
┌─────────────────────────────────────────────────────────┐
│  1. Expensive Evaluations                               │
│     └── Each evaluation requires full training          │
│                                                         │
│  2. Black-box Objective                                 │
│     └── No gradient information available               │
│                                                         │
│  3. High-dimensional Space                              │
│     └── Many hyperparameters to tune                    │
│                                                         │
│  4. Mixed Types                                         │
│     └── Continuous, discrete, categorical               │
│                                                         │
│  5. Conditional Dependencies                            │
│     └── Some HPs only relevant given others             │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Grid Search

### Basic Grid Search

**Algorithm:**

1. Define grid of hyperparameter values
2. Evaluate all combinations
3. Return best combination

```python
# Grid Search Example
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'hidden_units': [32, 64, 128],
    'dropout': [0.1, 0.3, 0.5]
}
# Total: 3 × 3 × 3 = 27 evaluations
```

### Curse of Dimensionality

$$\text{Number of evaluations} = \prod_{i=1}^{d} n_i$$

```
Grid Search Scaling:
┌─────────────────────────────────────────┐
│  d=2, n=10:    100 evaluations          │
│  d=3, n=10:    1,000 evaluations        │
│  d=5, n=10:    100,000 evaluations      │
│  d=10, n=10:   10,000,000,000 evals     │
└─────────────────────────────────────────┘
Exponential growth makes grid search infeasible!
```

### When to Use Grid Search

- Few hyperparameters (d ≤ 3)
- All hyperparameters roughly equally important
- Cheap model evaluations
- Want complete coverage

---

## 3. Random Search

### Key Insight (Bergstra & Bengio, 2012)

**Not all hyperparameters are equally important!**

```
Hyperparameter Importance Example:
┌───────────────────────────────────────────────────────┐
│                                                       │
│  Learning Rate: ████████████████████████  Very High   │
│                                                       │
│  Hidden Units:  ████████████              Medium      │
│                                                       │
│  Batch Size:    ████                      Low         │
│                                                       │
│  Momentum:      ████████                  Medium      │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### Grid vs Random Search

```
Grid Search (9 trials):        Random Search (9 trials):
┌─────────────────────┐        ┌─────────────────────┐
│ x   x   x           │        │   x       x         │
│                     │        │       x         x   │
│ x   x   x    HP2    │        │ x         x    HP2  │
│                     │        │     x       x       │
│ x   x   x           │        │         x           │
└─────────────────────┘        └─────────────────────┘
      HP1 (important)                HP1 (important)

Grid: 3 unique HP1 values      Random: 9 unique HP1 values!
```

**If only HP1 matters, random search explores 3x more values!**

### Random Search Algorithm

```
Algorithm: Random Search
─────────────────────────────────────
Input: Search space S, budget n
Output: Best configuration λ*

1. for i = 1 to n:
2.     Sample λ_i ~ Uniform(S)
3.     Evaluate f(λ_i)
4. Return λ* = argmin f(λ_i)
```

### Probability of Finding Good Region

$$P(\text{finding top } p\%) = 1 - (1-p)^n$$

```
Probability of finding top 5% region:
┌─────────────────────────────────────────┐
│  n=10:   P = 40%                        │
│  n=20:   P = 64%                        │
│  n=50:   P = 92%                        │
│  n=100:  P = 99.4%                      │
└─────────────────────────────────────────┘
```

---

## 4. Bayesian Optimization

### The Key Idea

**Use past evaluations to model the objective function!**

```
Bayesian Optimization Loop:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │ Observations │───>│ Surrogate    │───>│ Acquisition│ │
│  │ {(λ,f(λ))}  │    │ Model p(f|D) │    │ Function  │  │
│  └──────────────┘    └──────────────┘    └─────┬────┘  │
│         ^                                       │       │
│         │            ┌──────────────┐          │       │
│         └────────────│ Evaluate     │<─────────┘       │
│                      │ f(λ_next)    │                  │
│                      └──────────────┘                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Surrogate Models

#### Gaussian Process (GP)

$$f(\lambda) \sim \mathcal{GP}(m(\lambda), k(\lambda, \lambda'))$$

**Properties:**

- Provides mean prediction AND uncertainty
- Non-parametric (flexible)
- $O(n^3)$ complexity (limits to ~1000 observations)

**Common kernels:**
$$k_{RBF}(\lambda, \lambda') = \sigma^2 \exp\left(-\frac{||\lambda - \lambda'||^2}{2\ell^2}\right)$$

$$k_{Matern}(\lambda, \lambda') = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}r}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}r}{\ell}\right)$$

#### Tree Parzen Estimator (TPE)

Instead of modeling $p(y|\lambda)$, model:
$$p(\lambda|y) = \begin{cases} \ell(\lambda) & \text{if } y < y^* \\ g(\lambda) & \text{if } y \geq y^* \end{cases}$$

**Advantages:**

- Handles categorical variables naturally
- Scales better than GP
- Used in Hyperopt, Optuna

### Acquisition Functions

#### Expected Improvement (EI)

$$\text{EI}(\lambda) = \mathbb{E}[\max(0, f(\lambda^*) - f(\lambda))]$$

For Gaussian posterior:
$$\text{EI}(\lambda) = (\mu^* - \mu(\lambda))\Phi(Z) + \sigma(\lambda)\phi(Z)$$

where $Z = \frac{\mu^* - \mu(\lambda)}{\sigma(\lambda)}$

```
Expected Improvement Visualization:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Objective   ──────────────────────────────────         │
│  Function                  ╱╲                           │
│              ────────────╱────╲─────────────────        │
│                         ╱      ╲         ╱╲             │
│              ─────────╱──────────╲──────╱──╲────        │
│                                                         │
│  Surrogate   ═══════╦════════════╦════════════          │
│  Mean        ═══════╝            ╚═══════════           │
│                                                         │
│  Uncertainty ░░░░░▓▓▓▓▓▓▓▓▓▓▓░░░░░░░▓▓▓▓▓▓▓▓           │
│              (low)  (high)    (low)   (high)            │
│                        │                                │
│  EI peaks where      ◄─┘                                │
│  mean is low AND uncertainty is high                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Upper Confidence Bound (UCB)

$$\text{UCB}(\lambda) = -\mu(\lambda) + \kappa\sigma(\lambda)$$

- $\kappa$ controls exploration-exploitation tradeoff
- Higher $\kappa$ → more exploration

#### Probability of Improvement (PI)

$$\text{PI}(\lambda) = P(f(\lambda) < f(\lambda^*)) = \Phi\left(\frac{f(\lambda^*) - \mu(\lambda)}{\sigma(\lambda)}\right)$$

### Exploration-Exploitation Tradeoff

```
Acquisition Function Behavior:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Exploitation (low μ):    Exploration (high σ):         │
│  ───────────────────      ───────────────────           │
│  "Sample where model      "Sample where model           │
│   predicts good values"    is uncertain"                │
│                                                         │
│  Risk: Local optima       Risk: Wasted samples          │
│                                                         │
│  EI balances both:                                      │
│  ─────────────────                                      │
│  High EI = good prediction OR high uncertainty          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Sequential Model-Based Optimization (SMBO)

### General Algorithm

```
Algorithm: SMBO
─────────────────────────────────────
Input: Search space S, budget n, prior p(f)
Output: Best configuration λ*

1. Initialize D_0 with random samples
2. for i = 1 to n:
3.     Fit surrogate model M to D_{i-1}
4.     Select λ_i = argmax α(λ; M)  [acquisition]
5.     Evaluate y_i = f(λ_i)
6.     Update D_i = D_{i-1} ∪ {(λ_i, y_i)}
7. Return λ* = argmin_{(λ,y) ∈ D_n} y
```

### Practical Considerations

**Initialization:**

- Use Latin Hypercube Sampling or Sobol sequences
- Typically 5-10 random samples before optimization

**Batch Parallel Evaluation:**

- Kriging Believer: assume mean prediction for pending evaluations
- Constant Liar: assume worst-case for pending evaluations
- qEI: Jointly optimize batch acquisition

---

## 6. Hyperband and ASHA

### Early Stopping for Hyperparameter Search

**Key Insight:** Most configurations can be identified as bad early in training!

```
Training Curves for Different Hyperparameters:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Val    ▲                                               │
│  Loss   │  Bad config                                   │
│         │  ═══════════════════════════════════════      │
│         │                                               │
│         │       Medium config                           │
│         │       ───────────────────────────────         │
│         │                                               │
│         │              Good config                      │
│         │              ────────────────────────         │
│         │                                               │
│         └───────────────────────────────────────> Epoch │
│           │           │                                 │
│           Early       Full                              │
│           stopping    training                          │
│                                                         │
└─────────────────────────────────────────────────────────┘

We can identify bad configs early → save compute!
```

### Successive Halving

**Algorithm:**

1. Start with $n$ configurations, each with $r$ resources
2. After each round, keep top half, double resources
3. Continue until one configuration remains

```
Successive Halving Example (n=16, r=1):
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Round 1: 16 configs × 1 epoch  = 16 epochs            │
│           Keep top 8                                    │
│                                                         │
│  Round 2:  8 configs × 2 epochs = 16 epochs            │
│           Keep top 4                                    │
│                                                         │
│  Round 3:  4 configs × 4 epochs = 16 epochs            │
│           Keep top 2                                    │
│                                                         │
│  Round 4:  2 configs × 8 epochs = 16 epochs            │
│           Keep top 1                                    │
│                                                         │
│  Total: 64 epochs (vs 128 for full training of 8)      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Hyperband

**Problem with Successive Halving:** What if good configs need more resources early?

**Solution:** Run multiple brackets with different n/r tradeoffs

```
Hyperband Example (R=81, η=3):
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Bracket 4 (s=4): Many configs, aggressive pruning      │
│  ┌────────────────────────────────────────────────┐     │
│  │ n=81, r=1 → n=27, r=3 → n=9, r=9 → n=3, r=27  │     │
│  │ → n=1, r=81                                    │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  Bracket 3 (s=3): Moderate configs/pruning              │
│  ┌────────────────────────────────────────────────┐     │
│  │ n=27, r=3 → n=9, r=9 → n=3, r=27 → n=1, r=81  │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  Bracket 2 (s=2):                                       │
│  ┌────────────────────────────────────────────────┐     │
│  │ n=9, r=9 → n=3, r=27 → n=1, r=81              │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  Bracket 1 (s=1):                                       │
│  ┌────────────────────────────────────────────────┐     │
│  │ n=3, r=27 → n=1, r=81                          │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  Bracket 0 (s=0): Few configs, full training            │
│  ┌────────────────────────────────────────────────┐     │
│  │ n=1, r=81                                      │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### ASHA (Asynchronous Successive Halving)

**Problem:** Hyperband is synchronous—must wait for slowest trial

**Solution:** Asynchronous promotion

- Promote trials as soon as enough are ready
- Better GPU utilization

```
ASHA vs Hyperband:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Hyperband (synchronous):                               │
│  Worker 1: [====trial====]          [===trial===]       │
│  Worker 2: [====trial====]          [===trial===]       │
│  Worker 3: [==trial==]----wait----  [=trial=]           │
│  Worker 4: [===trial===]--wait----  [====trial====]     │
│                                                         │
│  ASHA (asynchronous):                                   │
│  Worker 1: [====trial====][====trial====][===trial===]  │
│  Worker 2: [====trial====][===trial===][====trial====]  │
│  Worker 3: [==trial==][====promoted====][===trial===]   │
│  Worker 4: [===trial===][===trial===][====trial====]    │
│                                                         │
│  ASHA: No idle time, better utilization                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Population-Based Training (PBT)

### Concept

**Combine hyperparameter optimization with training!**

```
PBT Overview:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Population of agents training in parallel              │
│                                                         │
│  Agent 1: ─────────┬────────┬────────────────           │
│                    │exploit │                           │
│  Agent 2: ─────────▼────────┼────────────────           │
│                             │exploit                    │
│  Agent 3: ──────────────────▼────────────────           │
│                                                         │
│  Agent 4: ─────────┬─────────────────────────           │
│                    │explore (mutate)                    │
│           ─────────▼─────────────────────────           │
│                                                         │
│  Legend: ─── training   explore: perturb HPs           │
│                         exploit: copy from better agent│
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Algorithm

```
Algorithm: PBT
─────────────────────────────────────
1. Initialize population of N agents with random HPs
2. for each training step:
3.     Train all agents in parallel
4.     if ready_to_exploit(agent):
5.         if performance(agent) in bottom 20%:
6.             Copy weights from top 20% agent
7.             Randomly perturb hyperparameters
8. Return best agent
```

### Advantages of PBT

- Adaptive hyperparameter schedules
- No separate tuning phase
- Can discover complex schedules

---

## 8. Multi-Fidelity Optimization

### Fidelity Parameters

```
Fidelity Approximations:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Low Fidelity (cheap):       High Fidelity (expensive): │
│  ───────────────────         ────────────────────────   │
│  • Fewer epochs              • Full training            │
│  • Subset of data            • Full dataset             │
│  • Smaller model             • Full model               │
│  • Lower resolution          • Full resolution          │
│                                                         │
│  Assumption: Low-fidelity ranking ≈ High-fidelity       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Multi-Fidelity Bayesian Optimization

**Model performance as function of both HP and fidelity:**

$$f(\lambda, z) \sim \mathcal{GP}(m(\lambda, z), k((\lambda, z), (\lambda', z')))$$

**Acquisition with fidelity:**
$$\alpha(\lambda, z) = \frac{\text{EI}(\lambda, z)}{\text{cost}(z)}$$

---

## 9. Practical Considerations

### Hyperparameter Importance

Use sensitivity analysis or ANOVA to identify important HPs:

```
Typical Importance Ranking (Neural Networks):
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Most Important:                                        │
│  • Learning rate                                        │
│  • Batch size                                          │
│  • Number of layers/units                               │
│                                                         │
│  Moderately Important:                                  │
│  • Dropout rate                                         │
│  • Weight decay                                         │
│  • Optimizer choice                                     │
│                                                         │
│  Less Important:                                        │
│  • Momentum                                             │
│  • Learning rate schedule details                       │
│  • Initialization scheme                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Search Space Design

```python
# Good search space design
search_space = {
    # Log scale for learning rate
    'learning_rate': loguniform(1e-5, 1e-1),

    # Log scale for regularization
    'weight_decay': loguniform(1e-6, 1e-2),

    # Integer for discrete values
    'num_layers': randint(1, 5),

    # Categorical
    'activation': choice(['relu', 'tanh', 'elu']),

    # Conditional: only if using dropout
    'dropout_rate': uniform(0.1, 0.5),  # if use_dropout
}
```

### Reproducibility

**Essential for fair comparison:**

1. Fix random seeds
2. Use same train/val splits
3. Report variance across multiple runs
4. Document compute budget

### Warm Starting

**Use previous knowledge:**

- Start from previously good configurations
- Transfer from similar tasks
- Use meta-learning

---

## 10. Hyperparameter Optimization Libraries

### Popular Tools

```
Library Comparison:
┌────────────────┬────────────────────────────────────────┐
│ Library        │ Features                               │
├────────────────┼────────────────────────────────────────┤
│ Optuna         │ TPE, pruning, distributed, dashboard   │
│ Ray Tune       │ All algorithms, scalable, integrations │
│ Hyperopt       │ TPE, random, tree-based               │
│ SMAC           │ Random forest surrogate               │
│ BoTorch        │ GP-based, multi-objective             │
│ Weights&Biases │ Sweeps, visualization, logging        │
└────────────────┴────────────────────────────────────────┘
```

---

## 11. Summary

### Method Selection Guide

```
Decision Tree for HPO Method:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  How many evaluations can you afford?                   │
│  ├── < 10: Random search                                │
│  ├── 10-100: Bayesian optimization (Optuna, BO)        │
│  └── > 100: Hyperband/ASHA                              │
│                                                         │
│  Is training expensive?                                 │
│  ├── Yes: Multi-fidelity (Hyperband, ASHA)             │
│  └── No: Standard BO                                    │
│                                                         │
│  Do you have parallel resources?                        │
│  ├── Yes: ASHA, PBT, parallel BO                       │
│  └── No: Sequential BO                                  │
│                                                         │
│  Is the HP space categorical/structured?                │
│  ├── Yes: TPE, SMAC                                     │
│  └── No: GP-based BO                                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Key Takeaways

1. **Random > Grid** for most problems
2. **Bayesian optimization** is the gold standard for few evaluations
3. **Early stopping** (Hyperband/ASHA) saves massive compute
4. **Not all hyperparameters matter equally** — identify important ones
5. **Use log scales** for learning rates and regularization
6. **Multi-fidelity** methods are essential for expensive models
7. **Reproducibility** requires careful seed management

---

## Exercises

1. **Grid vs Random**: For a 2D hyperparameter space where only one hyperparameter matters, calculate how many unique values each method explores with 16 evaluations.

2. **Bayesian Optimization**: Implement Expected Improvement (EI) for a 1D Gaussian Process. Given observations at $x = [0, 1]$ with $y = [1, 0.5]$, compute EI at $x = 0.5$.

3. **Successive Halving**: With budget for 64 total epochs and 8 initial configurations, design a successive halving schedule. How many epochs does the final configuration receive?

4. **GP Surrogate**: Explain why Gaussian Processes are preferred over neural networks as surrogates in Bayesian optimization when we have < 100 evaluations.

5. **Hyperband**: Compare the computational cost of Hyperband vs random search for finding a good learning rate among 81 candidates, where full training takes 81 epochs.

---

## References

1. Bergstra & Bengio (2012). "Random Search for Hyper-Parameter Optimization"
2. Snoek et al. (2012). "Practical Bayesian Optimization of ML Algorithms"
3. Li et al. (2018). "Hyperband: A Novel Bandit-Based Approach"
4. Jaderberg et al. (2017). "Population Based Training of Neural Networks"
5. Feurer & Hutter (2019). "Hyperparameter Optimization" (AutoML book)
