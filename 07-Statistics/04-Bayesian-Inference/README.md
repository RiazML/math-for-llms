# Bayesian Inference

## Introduction

Bayesian inference provides a principled framework for updating beliefs in light of evidence. Unlike frequentist methods, Bayesian approaches treat parameters as random variables with probability distributions, enabling uncertainty quantification and principled incorporation of prior knowledge.

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

| Term        | Name     | Meaning              |
| ----------- | -------- | -------------------- | ----------------------- |
| $P(\theta)$ | Prior    | Belief before data   |
| $P(D        | \theta)$ | Likelihood           | How likely data given θ |
| $P(\theta   | D)$      | Posterior            | Updated belief          |
| $P(D)$      | Evidence | Normalizing constant |

### 1.2 Posterior ∝ Likelihood × Prior

$$P(\theta|D) \propto P(D|\theta) P(\theta)$$

```
Bayesian Update:

Prior          ×    Likelihood     =    Posterior
(belief)           (data evidence)      (updated belief)

   ╱╲                   ╱╲                   ╱╲
  ╱  ╲                 ╱  ╲                 ╱  ╲
 ╱    ╲        ×      ╱    ╲        =      ╱    ╲
╱______╲             ╱______╲             ╱______╲
  Wide               Peaked at            Narrower,
(uncertain)          MLE                  shifted
```

---

## 2. Prior Distributions

### 2.1 Types of Priors

| Type                   | Description              | Example                        |
| ---------------------- | ------------------------ | ------------------------------ |
| **Informative**        | Strong prior knowledge   | Expert belief                  |
| **Weakly informative** | Mild regularization      | Normal(0, 10) for coefficients |
| **Non-informative**    | Minimal influence        | Uniform, Jeffreys              |
| **Conjugate**          | Same family as posterior | Beta for Bernoulli             |

### 2.2 Jeffreys Prior

Non-informative prior based on Fisher Information:

$$p(\theta) \propto \sqrt{I(\theta)}$$

Invariant to reparameterization.

---

## 3. Conjugate Priors

### 3.1 Definition

A prior is **conjugate** to a likelihood if the posterior is in the same family as the prior.

### 3.2 Common Conjugate Pairs

| Likelihood           | Conjugate Prior | Posterior               |
| -------------------- | --------------- | ----------------------- |
| Bernoulli/Binomial   | Beta(α, β)      | Beta(α+k, β+n-k)        |
| Poisson              | Gamma(α, β)     | Gamma(α+Σx, β+n)        |
| Normal (μ, σ² known) | Normal(μ₀, τ²)  | Normal(μ_post, τ²_post) |
| Normal (μ known)     | Inverse-Gamma   | Inverse-Gamma           |
| Multinomial          | Dirichlet       | Dirichlet               |
| Exponential          | Gamma           | Gamma                   |

### 3.3 Beta-Binomial Example

**Prior:** $p \sim \text{Beta}(\alpha_0, \beta_0)$

**Data:** k successes in n trials

**Posterior:** $p | D \sim \text{Beta}(\alpha_0 + k, \beta_0 + n - k)$

```
Beta-Binomial Update:

Beta(2,2) prior + 7 successes/3 failures = Beta(9,5) posterior

Prior:                         Posterior:
    ╱──────╲                      ╱╲
   ╱        ╲                    ╱  ╲
  ╱          ╲                  ╱    ╲
 ╱            ╲                ╱      ╲
0            1 p              0       1 p
 Uniform-ish                 Peaked near 0.7
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

- Posterior is precision-weighted average of prior and data
- More data → posterior closer to MLE
- More precise prior → posterior closer to prior mean

---

## 5. Point Estimates

### 5.1 Maximum A Posteriori (MAP)

$$\hat{\theta}_{MAP} = \arg\max_\theta P(\theta|D) = \arg\max_\theta [P(D|\theta)P(\theta)]$$

### 5.2 Posterior Mean

$$\hat{\theta}_{PM} = E[\theta|D] = \int \theta \cdot P(\theta|D) d\theta$$

### 5.3 Comparison

| Estimate     | Definition          | Optimal for |
| ------------ | ------------------- | ----------- | -------------------------- |
| MLE          | $\arg\max P(D       | \theta)$    | Squared error (asymptotic) |
| MAP          | $\arg\max P(\theta  | D)$         | 0-1 loss                   |
| Post. Mean   | $E[\theta           | D]$         | Squared error loss         |
| Post. Median | Median of $P(\theta | D)$         | Absolute error loss        |

---

## 6. Credible Intervals

### 6.1 Definition

A $(1-\alpha)$ **credible interval** $[a, b]$ satisfies:

$$P(a \leq \theta \leq b | D) = 1 - \alpha$$

### 6.2 Types

**Equal-tailed:** $P(\theta < a | D) = P(\theta > b | D) = \alpha/2$

**Highest Posterior Density (HPD):** Smallest interval containing $(1-\alpha)$ probability

### 6.3 Interpretation

Unlike frequentist CIs, we CAN say:
"There is a 95% probability that θ lies in this interval (given data and prior)"

---

## 7. Bayesian vs Frequentist

| Aspect         | Frequentist          | Bayesian              |
| -------------- | -------------------- | --------------------- |
| Parameters     | Fixed but unknown    | Random variables      |
| Probability    | Long-run frequency   | Degree of belief      |
| Prior info     | Not used             | Formally incorporated |
| Uncertainty    | Confidence intervals | Credible intervals    |
| Interpretation | Repeated sampling    | Given this data       |
| Computation    | Often analytical     | May need MCMC         |

### 7.1 When to Use Each

**Bayesian advantages:**

- Incorporating prior knowledge
- Quantifying parameter uncertainty
- Small sample sizes
- Sequential updating
- Principled model comparison

**Frequentist advantages:**

- No prior specification needed
- Simpler computation
- Well-understood guarantees
- Standard in many fields

---

## 8. Bayesian Model Comparison

### 8.1 Bayes Factor

$$BF_{12} = \frac{P(D|M_1)}{P(D|M_2)} = \frac{\int P(D|\theta_1, M_1)P(\theta_1|M_1)d\theta_1}{\int P(D|\theta_2, M_2)P(\theta_2|M_2)d\theta_2}$$

| $BF_{12}$ | Evidence for $M_1$ |
| --------- | ------------------ |
| 1-3       | Weak               |
| 3-10      | Moderate           |
| 10-30     | Strong             |
| > 30      | Very strong        |

### 8.2 Model Posterior

$$P(M_k|D) = \frac{P(D|M_k)P(M_k)}{\sum_j P(D|M_j)P(M_j)}$$

---

## 9. ML Applications

### 9.1 Bayesian Linear Regression

- Prior on weights: $\mathbf{w} \sim N(\mathbf{0}, \alpha^{-1}\mathbf{I})$
- Posterior: $\mathbf{w}|D \sim N(\mathbf{m}_N, \mathbf{S}_N)$

$$\mathbf{m}_N = \beta \mathbf{S}_N \mathbf{X}^T \mathbf{y}$$
$$\mathbf{S}_N^{-1} = \alpha \mathbf{I} + \beta \mathbf{X}^T \mathbf{X}$$

### 9.2 Connection to Regularization

| Prior            | Regularization |
| ---------------- | -------------- |
| $N(0, \sigma^2)$ | L2 (Ridge)     |
| Laplace          | L1 (Lasso)     |
| Spike-and-Slab   | Best subset    |

### 9.3 Bayesian Neural Networks

- Place priors on all weights
- Posterior gives uncertainty in predictions
- Predict using: $p(y|x, D) = \int p(y|x, \mathbf{w})p(\mathbf{w}|D)d\mathbf{w}$

### 9.4 Bayesian Optimization

- Prior over functions (Gaussian Process)
- Update posterior with observations
- Select next point using acquisition function

---

## 10. Computational Methods

### 10.1 Conjugate Priors

Analytical solutions when available.

### 10.2 Laplace Approximation

Approximate posterior as Gaussian centered at MAP:

$$P(\theta|D) \approx N(\hat{\theta}_{MAP}, \mathbf{H}^{-1})$$

where $\mathbf{H}$ is the Hessian of negative log-posterior.

### 10.3 MCMC (Markov Chain Monte Carlo)

- Metropolis-Hastings
- Gibbs Sampling
- Hamiltonian Monte Carlo

Draw samples from posterior to approximate integrals.

### 10.4 Variational Inference

Approximate posterior with simpler distribution by minimizing KL divergence.

---

## 11. Summary

| Concept           | Formula/Idea                         |
| ----------------- | ------------------------------------ | -------------- | ----------------- |
| Bayes' theorem    | $P(\theta                            | D) \propto P(D | \theta)P(\theta)$ |
| Conjugate prior   | Posterior in same family as prior    |
| MAP               | $\arg\max P(\theta                   | D)$            |
| Credible interval | Direct probability statement about θ |
| Bayes factor      | Model comparison via evidence ratio  |

**Key ML Connections:**

- Regularization = Prior
- MAP = Regularized MLE
- Bayesian NN = Uncertainty quantification
- GP = Bayesian nonparametrics

---

## References

1. Gelman et al. - "Bayesian Data Analysis"
2. Bishop - "Pattern Recognition and Machine Learning"
3. Murphy - "Machine Learning: A Probabilistic Perspective"
