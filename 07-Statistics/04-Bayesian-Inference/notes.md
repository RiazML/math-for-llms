# Bayesian Inference

> **Navigation**: [← 03-Hypothesis-Testing](../03-Hypothesis-Testing/) | [Statistics](../) | [08-Optimization →](../../08-Optimization/)

## Introduction

Bayesian inference provides a principled framework for updating beliefs in light of evidence. Unlike frequentist methods, Bayesian approaches treat parameters as random variables with probability distributions, enabling uncertainty quantification and principled incorporation of prior knowledge.

```
The Bayesian Perspective:
══════════════════════════════════════════════════════════════════

   Frequentist View                    Bayesian View
   ─────────────────                   ─────────────────
   Parameters are                      Parameters are
   FIXED but unknown                   RANDOM VARIABLES

   Data is random                      Data is observed
   (hypothetically                     (fixed once seen)
    repeatable)

   ┌────────────────┐                 ┌────────────────────────┐
   │ θ = ???        │                 │     P(θ | data)        │
   │ (one true      │                 │         ╱╲             │
   │  value)        │                 │        ╱  ╲            │
   └────────────────┘                 │       ╱    ╲           │
                                      │      ╱______╲          │
                                      │   (distribution        │
                                      │    of belief)          │
                                      └────────────────────────┘
```

## Prerequisites

- Probability theory (Bayes' theorem)
- Common distributions
- Maximum Likelihood Estimation

## Learning Objectives

1. Understand the Bayesian paradigm
2. Compute posterior distributions
3. Use conjugate priors
4. Compare Bayesian and frequentist approaches
5. Apply Bayesian methods to ML

---

## 1. Bayesian Framework

### 1.1 Bayes' Theorem

$$P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)}$$

| Term              | Name       | Meaning                    |
| ----------------- | ---------- | -------------------------- |
| $P(\theta)$       | Prior      | Belief before seeing data  |
| $P(D | \theta)$   | Likelihood | How likely data given θ    |
| $P(\theta | D)$   | Posterior  | Updated belief after data  |
| $P(D)$            | Evidence   | Normalizing constant       |

### 1.2 Posterior ∝ Likelihood × Prior

$$P(\theta|D) \propto P(D|\theta) P(\theta)$$

```
Bayesian Update Process:
══════════════════════════════════════════════════════════════════

    PRIOR              LIKELIHOOD           POSTERIOR
 (What we knew)    × (What data says)  =  (What we know now)

       ╱╲                  ╱╲                  ╱╲
      ╱  ╲                ╱  ╲                ╱  ╲
     ╱    ╲              ╱    ╲              ╱    ╲
    ╱      ╲      ×     ╱      ╲      =     ╱      ╲
   ╱        ╲          ╱        ╲          ╱        ╲
  ╱__________╲        ╱__________╲        ╱__________╲
  
    Wide               Peaked at           Narrower,
  (uncertain)            MLE                shifted
  
  
  ┌──────────────────────────────────────────────────────────────┐
  │ More data → Likelihood dominates → Posterior ≈ Likelihood   │
  │ Less data → Prior dominates → Posterior ≈ Prior             │
  └──────────────────────────────────────────────────────────────┘
```

> 💡 **Key Insight**: Bayesian inference is just a formal way of updating beliefs. Start with what you know (prior), observe evidence (likelihood), update your belief (posterior). Then the posterior becomes the prior for the next observation!

---

## 2. Prior Distributions

### 2.1 Types of Priors

| Type                   | Description                | Example                         |
| ---------------------- | -------------------------- | ------------------------------- |
| **Informative**        | Strong prior knowledge     | Expert belief, previous studies |
| **Weakly informative** | Mild regularization        | Normal(0, 10) for coefficients  |
| **Non-informative**    | Minimal influence          | Uniform, Jeffreys               |
| **Conjugate**          | Same family as posterior   | Beta for Bernoulli              |

```
Prior Selection Impact:
══════════════════════════════════════════════════════════════════

                      Few Data Points

   Strong Prior                         Weak Prior
   ┌───────────────┐                   ┌───────────────┐
   │     ╱╲        │                   │ ──────────    │
   │    ╱  ╲       │                   │ (flat)        │
   │   ╱    ╲      │                   │               │
   └───────────────┘                   └───────────────┘
          │                                   │
          ▼                                   ▼
   ┌───────────────┐                   ┌───────────────┐
   │     ╱╲        │                   │     ╱╲        │
   │    ╱  ╲       │                   │    ╱  ╲       │
   │   ╱    ╲__    │                   │   ╱    ╲      │
   └───────────────┘                   └───────────────┘
   Posterior near prior               Posterior follows data
   
   
                      Many Data Points
   
   Both priors → Similar posteriors (data dominates)
```

### 2.2 Jeffreys Prior

Non-informative prior based on Fisher Information:

$$p(\theta) \propto \sqrt{I(\theta)}$$

Invariant to reparameterization.

> ⚠️ **Caution**: "Non-informative" priors ARE still informative in some sense. There is no truly uninformative prior. Choose priors carefully and check sensitivity.

---

## 3. Conjugate Priors

### 3.1 Definition

A prior is **conjugate** to a likelihood if the posterior is in the same family as the prior.

### 3.2 Common Conjugate Pairs

| Likelihood           | Conjugate Prior   | Posterior                   |
| -------------------- | ----------------- | --------------------------- |
| Bernoulli/Binomial   | Beta(α, β)        | Beta(α+k, β+n-k)            |
| Poisson              | Gamma(α, β)       | Gamma(α+Σx, β+n)            |
| Normal (μ, σ² known) | Normal(μ₀, τ²)    | Normal(μ_post, τ²_post)     |
| Normal (μ known)     | Inverse-Gamma     | Inverse-Gamma               |
| Multinomial          | Dirichlet         | Dirichlet                   |
| Exponential          | Gamma             | Gamma                       |

> 💡 **Why Conjugate Priors?** They give closed-form posteriors, making computation tractable. Before MCMC was common, conjugate priors were almost required!

### 3.3 Beta-Binomial Example

**Prior:** $p \sim \text{Beta}(\alpha_0, \beta_0)$

**Data:** k successes in n trials

**Posterior:** $p | D \sim \text{Beta}(\alpha_0 + k, \beta_0 + n - k)$

```
Beta-Binomial Update Example:
══════════════════════════════════════════════════════════════════

Context: Estimating click-through rate

Prior: Beta(2, 2)  →  "We're unsure, centered around 0.5"

Data: 70 clicks out of 100 impressions

Posterior: Beta(2+70, 2+30) = Beta(72, 32)


          Prior Beta(2,2)              Posterior Beta(72,32)
    
              ╱────╲                           ╱╲
             ╱      ╲                         ╱  ╲
            ╱        ╲                       ╱    ╲
           ╱          ╲                     ╱      ╲
          ╱            ╲                   ╱        ╲
    ─────────────────────────        ─────────────────────────
    0                     1 p        0         0.7         1 p
    
    Prior mean: 0.5                   Posterior mean: 72/104 ≈ 0.69
    Prior samples: 4 (α+β)            Posterior samples: 104 (α+β)
    
    "Prior acts like 4 pseudo-observations: 2 successes, 2 failures"
```

---

## 4. Normal-Normal Conjugacy

### 4.1 Setup

- Prior: $\mu \sim N(\mu_0, \tau_0^2)$
- Likelihood: $X_i | \mu \sim N(\mu, \sigma^2)$ (σ² known)

### 4.2 Posterior

$$\mu | X_1, \ldots, X_n \sim N(\mu_{post}, \tau_{post}^2)$$

where:

$$\mu_{post} = \frac{\frac{\mu_0}{\tau_0^2} + \frac{n\bar{x}}{\sigma^2}}{\frac{1}{\tau_0^2} + \frac{n}{\sigma^2}}$$

$$\frac{1}{\tau_{post}^2} = \frac{1}{\tau_0^2} + \frac{n}{\sigma^2}$$

### 4.3 Interpretation

```
Normal-Normal Update as Precision-Weighted Average:
══════════════════════════════════════════════════════════════════

Precision = 1/Variance  (how "sure" we are)

Prior precision:     τ₀⁻² 
Data precision:      n/σ²  (more data = more precision)
Posterior precision: τ₀⁻² + n/σ²  (precisions ADD!)


Posterior mean = Weighted average of prior mean and data mean

         (Prior precision)×(Prior mean) + (Data precision)×(Sample mean)
μ_post = ──────────────────────────────────────────────────────────────────
                    Prior precision + Data precision


       ┌──────────────────────────────────────────────────────────┐
       │ More data (larger n) → Posterior closer to sample mean  │
       │ Stronger prior (smaller τ₀²) → Posterior closer to μ₀   │
       └──────────────────────────────────────────────────────────┘
```

---

## 5. Point Estimates

### 5.1 Maximum A Posteriori (MAP)

$$\hat{\theta}_{MAP} = \arg\max_\theta P(\theta|D) = \arg\max_\theta [P(D|\theta)P(\theta)]$$

### 5.2 Posterior Mean

$$\hat{\theta}_{PM} = E[\theta|D] = \int \theta \cdot P(\theta|D) d\theta$$

### 5.3 Comparison

| Estimate          | Definition                   | Optimal for            |
| ----------------- | ---------------------------- | ---------------------- |
| MLE               | $\arg\max P(D|\theta)$       | Squared error (asymp.) |
| MAP               | $\arg\max P(\theta|D)$       | 0-1 loss               |
| Posterior Mean    | $E[\theta|D]$                | Squared error loss     |
| Posterior Median  | Median of $P(\theta|D)$      | Absolute error loss    |

```
When Point Estimates Differ:
══════════════════════════════════════════════════════════════════

For symmetric, unimodal posteriors: 
    MAP ≈ Mean ≈ Median

For skewed posteriors:
    
        ╱╲
       ╱  ╲
      ╱    ╲_______________
     ╱                     ╲
    ╱_________________________╲
    │    │          │
   MAP  Med       Mean
   
Use posterior mean for prediction (minimizes squared error)
Use MAP when you want a single "most likely" value
Use full posterior to quantify uncertainty!
```

> 💡 **Best Practice**: Don't reduce the posterior to a point estimate unless necessary. The full posterior captures uncertainty, which is often more valuable than a single number.

---

## 6. Credible Intervals

### 6.1 Definition

A $(1-\alpha)$ **credible interval** $[a, b]$ satisfies:

$$P(a \leq \theta \leq b | D) = 1 - \alpha$$

### 6.2 Types

**Equal-tailed:** $P(\theta < a | D) = P(\theta > b | D) = \alpha/2$

**Highest Posterior Density (HPD):** Smallest interval containing $(1-\alpha)$ probability

```
Credible Intervals: Equal-tailed vs HPD:
══════════════════════════════════════════════════════════════════

         Symmetric Posterior              Skewed Posterior
    
              ╱╲                               ╱╲
             ╱  ╲                             ╱  ╲____
            ╱    ╲                           ╱        ╲
           ╱      ╲                         ╱          ╲
          ╱        ╲                       ╱            ╲
    ────|████████████|────           ────|███████████████|────
       Equal-tailed                      Equal-tailed
       = HPD                        
                                    ────|████████████|────────
                                         HPD (narrower!)
                                         
    For symmetric: Both are identical
    For skewed: HPD gives shortest interval
```

### 6.3 Interpretation

Unlike frequentist CIs, we CAN say:

> "There is a 95% probability that θ lies in this interval (given our data and prior)"

This is what people intuitively WANT to say but can't with frequentist CIs!

---

## 7. Bayesian vs Frequentist

| Aspect         | Frequentist                | Bayesian                       |
| -------------- | -------------------------- | ------------------------------ |
| Parameters     | Fixed but unknown          | Random variables               |
| Probability    | Long-run frequency         | Degree of belief               |
| Prior info     | Not used                   | Formally incorporated          |
| Uncertainty    | Confidence intervals       | Credible intervals             |
| Interpretation | Repeated sampling          | Given this specific data       |
| Computation    | Often analytical           | May need MCMC                  |

```
The Two Paradigms - Different Questions:
══════════════════════════════════════════════════════════════════

FREQUENTIST:
"If I repeated this experiment infinitely, what would happen?"

    Exp 1 ──▶ CI₁ ──────[---|---]────
    Exp 2 ──▶ CI₂ ───────[--|--]──────
    Exp 3 ──▶ CI₃ ─────[---|----]─────    95% contain θ
    Exp 4 ──▶ CI₄ ────────────[--|--]─    (this one missed!)
                              ▲
                              θ (fixed, unknown)


BAYESIAN:
"Given THIS data and my prior, what do I believe about θ?"

    This experiment ──▶ Posterior P(θ|D)
    
              ┌─────────────────────────┐
              │         ╱╲              │
              │        ╱  ╲             │
              │       ╱    ╲            │
              │ 95% ╱________╲          │
              │     ▲        ▲          │
              │     a        b          │
              │   P(a≤θ≤b|D) = 0.95     │
              └─────────────────────────┘
```

### 7.1 When to Use Each

**Bayesian advantages:**

- Incorporating prior knowledge
- Quantifying parameter uncertainty
- Small sample sizes  
- Sequential updating (online learning)
- Principled model comparison

**Frequentist advantages:**

- No prior specification needed
- Simpler computation
- Well-understood theoretical guarantees
- Standard in many scientific fields

---

## 8. Bayesian Model Comparison

### 8.1 Bayes Factor

$$BF_{12} = \frac{P(D|M_1)}{P(D|M_2)} = \frac{\int P(D|\theta_1, M_1)P(\theta_1|M_1)d\theta_1}{\int P(D|\theta_2, M_2)P(\theta_2|M_2)d\theta_2}$$

| $BF_{12}$    | Evidence for $M_1$    |
| ------------ | --------------------- |
| 1-3          | Weak                  |
| 3-10         | Moderate              |
| 10-30        | Strong                |
| > 30         | Very strong           |

### 8.2 Model Posterior

$$P(M_k|D) = \frac{P(D|M_k)P(M_k)}{\sum_j P(D|M_j)P(M_j)}$$

> 💡 **Bayesian Occam's Razor**: Bayes factors naturally penalize complex models! Complex models spread probability over more parameter space, reducing the marginal likelihood. Simpler models that fit well are preferred.

---

## 9. ML Applications

### 9.1 Bayesian Linear Regression

- Prior on weights: $\mathbf{w} \sim N(\mathbf{0}, \alpha^{-1}\mathbf{I})$
- Posterior: $\mathbf{w}|D \sim N(\mathbf{m}_N, \mathbf{S}_N)$

$$\mathbf{m}_N = \beta \mathbf{S}_N \mathbf{X}^T \mathbf{y}$$
$$\mathbf{S}_N^{-1} = \alpha \mathbf{I} + \beta \mathbf{X}^T \mathbf{X}$$

```
Bayesian Linear Regression Benefits:
══════════════════════════════════════════════════════════════════

Standard Regression:              Bayesian Regression:
Point prediction only             Prediction with uncertainty!

y                                 y
│     ____•____                   │     ╱═══════╲__
│    /         \                  │    ╱░░░░░░░░░░╲
│   / •         \                 │   ╱░░░░░░░░░░░░╲
│  / • •         \                │  ╱░░░░░░░░░░░░░░╲
│ /•   •          \               │ ╱░░░░░░░░░░░░░░░░╲
│/____•____________\              │░░░░░░░░░░░░░░░░░░░╲
└───────────────────▶ x           └─────────────────────▶ x

Single line ŷ = wx + b           Distribution of predictions
                                 ░░░ = Uncertainty region
                                 Wider where data is sparse!
```

### 9.2 Connection to Regularization

| Prior Type       | Regularization        | Penalty               |
| ---------------- | --------------------- | --------------------- |
| Gaussian $N(0,σ²)$ | L2 (Ridge)          | $\lambda\|\theta\|^2$ |
| Laplace          | L1 (Lasso)            | $\lambda\|\theta\|_1$ |
| Spike-and-Slab   | Best subset selection | Sparsity              |
| Horseshoe        | Adaptive shrinkage    | Heavy-tailed sparsity |

```
Regularization = Bayesian Prior:
══════════════════════════════════════════════════════════════════

L2 Regularization              Gaussian Prior

min Σ(y - Xw)² + λ||w||²   =   max P(y|X,w)·P(w)
                                        └─────┘
                                     N(0, σ²) prior
                                     
The λ parameter corresponds to the prior variance!
Large λ = Strong prior (small variance) = More regularization
```

### 9.3 Bayesian Neural Networks

- Place priors on all weights
- Posterior gives uncertainty in predictions
- Predict using: $p(y|x, D) = \int p(y|x, \mathbf{w})p(\mathbf{w}|D)d\mathbf{w}$

**Benefits:**

- Uncertainty quantification (crucial for safety-critical applications)
- Automatic model complexity control
- Robust to overfitting

### 9.4 Bayesian Optimization

- Prior over functions (Gaussian Process)
- Update posterior with observations
- Select next point using acquisition function

```
Bayesian Optimization Flow:
══════════════════════════════════════════════════════════════════

For hyperparameter tuning:

1. Start with GP prior over validation accuracy as f(hyperparams)

2. Evaluate a few random hyperparameter settings

3. Update posterior: P(f | observations)

4. Choose next hyperparams by maximizing acquisition function
   (balance exploration vs exploitation)

5. Evaluate, update posterior, repeat

        ┌─────────────────────────────────────────────────────┐
        │  Much more sample-efficient than grid search!       │
        │  Especially good when evaluations are expensive     │
        └─────────────────────────────────────────────────────┘
```

---

## 10. Computational Methods

### 10.1 Conjugate Priors

Analytical solutions when available. Fast but limited.

### 10.2 Laplace Approximation

Approximate posterior as Gaussian centered at MAP:

$$P(\theta|D) \approx N(\hat{\theta}_{MAP}, \mathbf{H}^{-1})$$

where $\mathbf{H}$ is the Hessian of negative log-posterior.

### 10.3 MCMC (Markov Chain Monte Carlo)

- **Metropolis-Hastings**: General purpose
- **Gibbs Sampling**: When conditionals are tractable
- **Hamiltonian Monte Carlo**: Uses gradients, more efficient

Draw samples from posterior to approximate integrals.

### 10.4 Variational Inference

Approximate posterior with simpler distribution by minimizing KL divergence.

```
Computational Methods Comparison:
══════════════════════════════════════════════════════════════════

Method              │ Accuracy  │ Speed   │ When to Use
────────────────────┼───────────┼─────────┼─────────────────────
Conjugate           │ Exact     │ ★★★★★   │ When available
Laplace Approx      │ Moderate  │ ★★★★☆   │ Unimodal posteriors
MCMC                │ Excellent │ ★★☆☆☆   │ Small-medium data
Variational         │ Good      │ ★★★★☆   │ Large data, NNs
```

---

## 11. Summary

### Key Formulas

| Concept            | Formula/Idea                                      |
| ------------------ | ------------------------------------------------- |
| Bayes' theorem     | $P(\theta|D) \propto P(D|\theta)P(\theta)$        |
| Conjugate prior    | Posterior in same family as prior                 |
| MAP                | $\arg\max P(\theta|D)$                            |
| Credible interval  | Direct probability statement about θ              |
| Bayes factor       | Model comparison via evidence ratio               |

### Key ML Connections

```
Bayesian View of Machine Learning:
══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Regularization  =  Prior distribution on parameters          │
│                                                                 │
│   MAP Estimation  =  Regularized Maximum Likelihood            │
│                                                                 │
│   Bayesian NN     =  Uncertainty quantification                 │
│                                                                 │
│   Dropout         ≈  Approximate Bayesian inference             │
│                                                                 │
│   Ensemble        ≈  Sampling from posterior                    │
│                                                                 │
│   Gaussian Process = Bayesian nonparametric regression          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Quick Reference

```
Cheat Sheet:
══════════════════════════════════════════════════════════════════

Posterior ∝ Likelihood × Prior

More data → Prior matters less
Less data → Prior matters more

Conjugate pairs (memorize these!):
  • Binomial + Beta → Beta
  • Normal + Normal → Normal  
  • Poisson + Gamma → Gamma
  • Multinomial + Dirichlet → Dirichlet

Point estimates:
  • MAP = mode of posterior
  • Posterior mean = expected value
  
Credible vs Confidence Intervals:
  • Credible: "95% probability θ is in here" (given data)
  • Confidence: "95% of intervals contain θ" (if repeated)
```

---

## Exercises

1. **Prior Sensitivity**: With a Beta(1,1) prior and 3 successes in 10 trials, compute the posterior. How does it change with Beta(10,10) prior?

2. **Conjugacy**: Derive the posterior for μ with a Normal prior and Normal likelihood (σ² known). Show it's a precision-weighted average.

3. **MAP vs MLE**: Show that MAP with a Gaussian prior on θ is equivalent to MLE with L2 regularization.

4. **Model Comparison**: You have two models: simple (1 parameter) and complex (10 parameters). Both fit the data equally well. Which would Bayesian model comparison favor and why?

---

## References

1. Gelman et al. - "Bayesian Data Analysis"
2. Bishop - "Pattern Recognition and Machine Learning"
3. Murphy - "Machine Learning: A Probabilistic Perspective"
4. McElreath - "Statistical Rethinking" (excellent intro)
