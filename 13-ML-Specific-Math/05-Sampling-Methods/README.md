# Sampling Methods in Machine Learning

## Overview

Sampling methods are fundamental to probabilistic machine learning, enabling inference, generation, and optimization in complex probability distributions where analytical solutions are intractable.

## 1. Basic Sampling Methods

### Inverse Transform Sampling

For a continuous random variable with CDF $F$:
$$X = F^{-1}(U) \quad \text{where } U \sim \text{Uniform}(0, 1)$$

**Algorithm**:

1. Generate $u \sim \text{Uniform}(0, 1)$
2. Compute $x = F^{-1}(u)$

**Limitations**: Requires known, invertible CDF

### Box-Muller Transform

Generate standard normal samples from uniform samples:

Given $U_1, U_2 \sim \text{Uniform}(0, 1)$:
$$Z_1 = \sqrt{-2\ln U_1}\cos(2\pi U_2)$$
$$Z_2 = \sqrt{-2\ln U_1}\sin(2\pi U_2)$$

Then $Z_1, Z_2 \sim \mathcal{N}(0, 1)$ independently.

### Rejection Sampling

To sample from target $p(x)$ using proposal $q(x)$ where $Mq(x) \geq p(x)$ for all $x$:

1. Sample $x \sim q(x)$
2. Sample $u \sim \text{Uniform}(0, 1)$
3. Accept $x$ if $u < \frac{p(x)}{Mq(x)}$, else reject

**Acceptance rate**: $\frac{1}{M}$

**Challenge**: Finding tight bound $M$ in high dimensions

## 2. Markov Chain Monte Carlo (MCMC)

### Core Idea

Construct a Markov chain whose stationary distribution is the target $p(x)$.

### Detailed Balance

A sufficient condition for stationarity:
$$p(x)T(x'|x) = p(x')T(x|x')$$

where $T(x'|x)$ is the transition kernel.

### Metropolis-Hastings Algorithm

1. Initialize $x_0$
2. For $t = 1, 2, \ldots$:
   - Propose $x' \sim q(x'|x_t)$
   - Compute acceptance ratio:
     $$\alpha = \min\left(1, \frac{p(x')q(x_t|x')}{p(x_t)q(x'|x_t)}\right)$$
   - Accept: $x_{t+1} = x'$ with probability $\alpha$
   - Reject: $x_{t+1} = x_t$ otherwise

### Metropolis Algorithm (Symmetric Proposal)

When $q(x'|x) = q(x|x')$:
$$\alpha = \min\left(1, \frac{p(x')}{p(x_t)}\right)$$

### Gibbs Sampling

Sample each dimension conditioned on others:

For $x = (x_1, \ldots, x_d)$:

1. Sample $x_1^{(t+1)} \sim p(x_1 | x_2^{(t)}, \ldots, x_d^{(t)})$
2. Sample $x_2^{(t+1)} \sim p(x_2 | x_1^{(t+1)}, x_3^{(t)}, \ldots, x_d^{(t)})$
3. ...continue for all dimensions

**Advantages**: No rejection, always accepts
**Requirement**: Ability to sample from conditionals

## 3. Hamiltonian Monte Carlo (HMC)

### Physical Intuition

Treat the negative log-probability as potential energy:
$$U(x) = -\log p(x)$$

Introduce momentum $p$ with kinetic energy:
$$K(p) = \frac{1}{2}p^T M^{-1} p$$

Total Hamiltonian:
$$H(x, p) = U(x) + K(p)$$

### Leapfrog Integration

For step size $\epsilon$ and $L$ steps:

```
p ← p - (ε/2) ∇U(x)           # Half momentum update
for i = 1 to L-1:
    x ← x + ε M⁻¹ p           # Full position update
    p ← p - ε ∇U(x)           # Full momentum update
x ← x + ε M⁻¹ p               # Final position update
p ← p - (ε/2) ∇U(x)           # Final half momentum update
```

### HMC Algorithm

1. Sample momentum: $p \sim \mathcal{N}(0, M)$
2. Run leapfrog to get $(x', p')$
3. Accept with probability:
   $$\alpha = \min(1, \exp(H(x, p) - H(x', p')))$$

### Properties

- Suppresses random walk behavior
- Explores state space more efficiently
- Requires gradient computation

## 4. No-U-Turn Sampler (NUTS)

### Motivation

Automatically tune HMC's trajectory length $L$.

### Idea

Extend trajectory until it starts "turning back":
$$(\theta^- - \theta^+) \cdot p^- < 0 \quad \text{or} \quad (\theta^- - \theta^+) \cdot p^+ < 0$$

### Building Tree

Recursively double trajectory length in forward/backward directions until U-turn detected.

## 5. Importance Sampling

### Basic Importance Sampling

Estimate expectation under $p$ using samples from $q$:
$$E_p[f(x)] = E_q\left[f(x)\frac{p(x)}{q(x)}\right] = E_q[f(x)w(x)]$$

where $w(x) = \frac{p(x)}{q(x)}$ are importance weights.

### Estimator

$$\hat{\mu} = \frac{1}{N}\sum_{i=1}^N f(x_i)w(x_i), \quad x_i \sim q$$

### Self-Normalized Importance Sampling

When $p(x)$ is known only up to a constant:
$$\hat{\mu} = \frac{\sum_{i=1}^N f(x_i)w(x_i)}{\sum_{i=1}^N w(x_i)}$$

### Effective Sample Size (ESS)

$$\text{ESS} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2} = \frac{N}{1 + \text{Var}(w)}$$

Low ESS indicates poor proposal distribution.

## 6. Sequential Monte Carlo (SMC)

### Particle Filtering

For state-space models $p(x_{1:T}, y_{1:T})$:

1. **Initialize**: Sample $x_1^{(i)} \sim p(x_1)$
2. **For $t = 2, \ldots, T$**:
   - Propagate: $\tilde{x}_t^{(i)} \sim p(x_t | x_{t-1}^{(i)})$
   - Weight: $w_t^{(i)} \propto p(y_t | \tilde{x}_t^{(i)})$
   - Resample: Select $x_t^{(i)}$ from $\{\tilde{x}_t^{(j)}\}$ with probabilities $\propto w_t^{(j)}$

### Systematic Resampling

Evenly spaced sampling of cumulative weights to reduce variance.

## 7. Variational Inference Connection

### Reparameterization Trick

For $z \sim q_\phi(z|x)$, write:
$$z = g_\phi(\epsilon, x), \quad \epsilon \sim p(\epsilon)$$

Example for Gaussian:
$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

This enables gradient computation through sampling.

### Gumbel-Softmax (Concrete)

For categorical sampling, approximate:
$$y_i = \frac{\exp((\log\pi_i + g_i)/\tau)}{\sum_j \exp((\log\pi_j + g_j)/\tau)}$$

where $g_i \sim \text{Gumbel}(0, 1)$ and $\tau$ is temperature.

## 8. Modern Sampling for Generative Models

### Ancestral Sampling (Autoregressive)

$$p(x) = \prod_{i=1}^d p(x_i | x_{<i})$$

Sample sequentially: $x_i \sim p(x_i | x_1, \ldots, x_{i-1})$

### Diffusion Sampling

**Forward process**: $q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$

**Reverse sampling**:
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

Start from $x_T \sim \mathcal{N}(0, I)$ and iterate backward.

### Score-Based Sampling

Using score function $s_\theta(x) \approx \nabla_x \log p(x)$:

**Langevin dynamics**:
$$x_{t+1} = x_t + \frac{\epsilon}{2}\nabla_x \log p(x_t) + \sqrt{\epsilon}\eta_t, \quad \eta_t \sim \mathcal{N}(0, I)$$

## 9. Sampling Diagnostics

### Convergence Diagnostics

**Trace plots**: Visual inspection of samples over iterations

**Gelman-Rubin statistic** ($\hat{R}$):
$$\hat{R} = \sqrt{\frac{\text{Var}^+(\theta)}{W}}$$
where $W$ is within-chain variance and $\text{Var}^+$ is pooled variance.
Target: $\hat{R} < 1.01$

### Autocorrelation

$$\rho_k = \frac{\text{Cov}(x_t, x_{t+k})}{\text{Var}(x_t)}$$

**Effective Sample Size**:
$$n_{eff} = \frac{n}{1 + 2\sum_{k=1}^\infty \rho_k}$$

### Acceptance Rate

- **Metropolis-Hastings**: Target ~23% for high dimensions, ~44% for 1D
- **HMC**: Target ~65%

## 10. Practical Considerations

### Burn-in / Warm-up

Discard initial samples before chain reaches stationarity.

### Thinning

Keep every $k$-th sample to reduce autocorrelation (often unnecessary).

### Multiple Chains

Run parallel chains with different initializations to assess convergence.

### Adaptation

- Tune proposal variance (Metropolis)
- Tune step size and mass matrix (HMC)
- Modern samplers: NUTS, adaptive Metropolis

## 11. Specialized Sampling Methods

### Slice Sampling

Sample from uniform distribution under the curve:

1. Sample $y \sim \text{Uniform}(0, f(x_t))$
2. Find slice $S = \{x : f(x) > y\}$
3. Sample $x_{t+1} \sim \text{Uniform}(S)$

### Elliptical Slice Sampling

For posteriors of form $p(x) \propto L(x)\mathcal{N}(x; 0, \Sigma)$:

Uses ellipses defined by prior covariance.

### Annealed Importance Sampling

Bridge between easy $p_0$ and target $p_T$ through intermediate distributions:
$$p_t(x) \propto p_0(x)^{1-\beta_t} p_T(x)^{\beta_t}$$

## 12. Sampling in High Dimensions

### Curse of Dimensionality

- Typical sets concentrate in thin shells
- Random walk inefficient: $O(d^2)$ steps needed
- HMC scales as $O(d^{5/4})$

### Strategies

1. **Gradient-based methods**: HMC, NUTS
2. **Rao-Blackwellization**: Analytically marginalize some variables
3. **Blocking**: Update correlated variables together
4. **Preconditioning**: Use mass matrix matching posterior geometry

## Key Takeaways

1. **Inverse transform/Box-Muller**: Simple distributions only
2. **MCMC**: General framework for sampling intractable distributions
3. **HMC/NUTS**: State-of-the-art for continuous distributions
4. **Importance sampling**: Unbiased estimates, beware of ESS
5. **Reparameterization**: Enables gradients through stochastic nodes
6. **Diagnostics**: Always check convergence ($\hat{R}$, ESS, trace plots)
7. **Modern samplers**: NUTS for most problems, SMC for sequences
