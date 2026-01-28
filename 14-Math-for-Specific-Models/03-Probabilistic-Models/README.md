# Probabilistic Models: Mathematical Foundations

## Overview

Probabilistic models provide a principled framework for reasoning under uncertainty, enabling rigorous treatment of noise, missing data, and model uncertainty. They form the foundation of Bayesian machine learning and generative modeling.

## 1. Probabilistic Framework

### Random Variables and Distributions

**Discrete**: $P(X = x)$ where $\sum_x P(X = x) = 1$

**Continuous**: $p(x)$ where $\int p(x) dx = 1$

**Joint**: $p(x, y)$

**Conditional**: $p(y|x) = \frac{p(x, y)}{p(x)}$

**Marginal**: $p(x) = \sum_y p(x, y)$ or $p(x) = \int p(x, y) dy$

### Bayes' Theorem

$$p(\theta | D) = \frac{p(D | \theta) p(\theta)}{p(D)}$$

- **Posterior**: $p(\theta | D)$
- **Likelihood**: $p(D | \theta)$
- **Prior**: $p(\theta)$
- **Evidence**: $p(D) = \int p(D | \theta) p(\theta) d\theta$

## 2. Exponential Family

### General Form

$$p(x | \eta) = h(x) \exp(\eta^T T(x) - A(\eta))$$

- $\eta$: Natural parameters
- $T(x)$: Sufficient statistics
- $A(\eta)$: Log-partition function (normalizer)
- $h(x)$: Base measure

### Properties

**Mean**: $\mathbb{E}[T(x)] = \nabla_\eta A(\eta)$

**Variance**: $\text{Var}[T(x)] = \nabla^2_\eta A(\eta)$

### Common Distributions

| Distribution | $\eta$                           | $T(x)$     | $A(\eta)$                              |
| ------------ | -------------------------------- | ---------- | -------------------------------------- |
| Bernoulli    | $\log \frac{p}{1-p}$             | $x$        | $\log(1 + e^\eta)$                     |
| Gaussian     | $[\mu/\sigma^2, -1/(2\sigma^2)]$ | $[x, x^2]$ | $\frac{\mu^2}{2\sigma^2} + \log\sigma$ |
| Poisson      | $\log \lambda$                   | $x$        | $e^\eta$                               |

## 3. Graphical Models

### Bayesian Networks

**Directed acyclic graph** encoding factorization:
$$p(x_1, \ldots, x_n) = \prod_{i=1}^n p(x_i | \text{Pa}(x_i))$$

### Markov Random Fields

**Undirected graph** with potential functions:
$$p(x) = \frac{1}{Z} \prod_{c \in \mathcal{C}} \psi_c(x_c)$$

where $Z = \sum_x \prod_c \psi_c(x_c)$ is the partition function.

### D-Separation (Bayesian Networks)

$X \perp Y | Z$ if all paths between $X$ and $Y$ are blocked by $Z$.

**Blocked paths**:

- Chain: $X \to Z \to Y$ (blocked if $Z$ observed)
- Fork: $X \leftarrow Z \to Y$ (blocked if $Z$ observed)
- Collider: $X \to Z \leftarrow Y$ (blocked if $Z$ NOT observed)

## 4. Latent Variable Models

### General Framework

$$p(x) = \int p(x | z) p(z) dz$$

- $z$: Latent (hidden) variables
- $p(z)$: Prior over latents
- $p(x|z)$: Observation model

### Gaussian Mixture Model

$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$$

**Latent**: $z \in \{1, \ldots, K\}$ with $P(z=k) = \pi_k$

### Factor Analysis

$$x = Wz + \mu + \epsilon$$

where $z \sim \mathcal{N}(0, I)$ and $\epsilon \sim \mathcal{N}(0, \Psi)$

**Covariance structure**: $\Sigma = WW^T + \Psi$

### Probabilistic PCA

Factor analysis with $\Psi = \sigma^2 I$:

$$x = Wz + \mu + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

## 5. Expectation-Maximization (EM)

### Lower Bound

$$\log p(x|\theta) \geq \mathcal{L}(q, \theta) = \mathbb{E}_q[\log p(x, z|\theta)] - \mathbb{E}_q[\log q(z)]$$

### E-Step

$$q^{(t+1)}(z) = p(z | x, \theta^{(t)})$$

### M-Step

$$\theta^{(t+1)} = \arg\max_\theta \mathbb{E}_{q^{(t+1)}}[\log p(x, z | \theta)]$$

### Convergence

EM monotonically increases log-likelihood:
$$\log p(x | \theta^{(t+1)}) \geq \log p(x | \theta^{(t)})$$

## 6. Variational Inference

### Evidence Lower Bound (ELBO)

$$\log p(x) \geq \mathbb{E}_{q(z)}[\log p(x, z)] - \mathbb{E}_{q(z)}[\log q(z)]$$
$$= \mathbb{E}_{q(z)}[\log p(x|z)] - D_{KL}(q(z) \| p(z))$$

### Mean-Field Approximation

$$q(z) = \prod_i q_i(z_i)$$

**Optimal updates**:
$$\log q_j^*(z_j) = \mathbb{E}_{q_{-j}}[\log p(x, z)] + \text{const}$$

### Reparameterization Trick

For $z \sim q_\phi(z|x)$:
$$z = g_\phi(\epsilon, x), \quad \epsilon \sim p(\epsilon)$$

**Gradient**:
$$\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon, x))]$$

## 7. Variational Autoencoders

### Model

**Encoder**: $q_\phi(z|x) = \mathcal{N}(z | \mu_\phi(x), \sigma^2_\phi(x))$

**Decoder**: $p_\theta(x|z)$ (Gaussian or Bernoulli)

**Prior**: $p(z) = \mathcal{N}(0, I)$

### ELBO

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

### KL for Gaussians

$$D_{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = -\frac{1}{2}\sum_j (1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2)$$

## 8. Markov Chain Monte Carlo

### Metropolis-Hastings

**Proposal**: $q(x'|x)$

**Acceptance**: $\alpha = \min\left(1, \frac{p(x')q(x|x')}{p(x)q(x'|x)}\right)$

### Gibbs Sampling

Sample each variable conditioned on others:
$$x_i^{(t+1)} \sim p(x_i | x_{-i}^{(t)})$$

### Hamiltonian Monte Carlo

**Dynamics**:
$$\frac{dq}{dt} = \frac{\partial H}{\partial p}, \quad \frac{dp}{dt} = -\frac{\partial H}{\partial q}$$

where $H(q, p) = U(q) + K(p) = -\log p(q) + \frac{1}{2}p^T M^{-1} p$

## 9. Gaussian Processes

### Definition

$$f(\cdot) \sim \mathcal{GP}(m(\cdot), k(\cdot, \cdot))$$

Any finite set $\{f(x_1), \ldots, f(x_n)\}$ is jointly Gaussian.

### Posterior

Given observations $(X, y)$ with $y = f(X) + \epsilon$:

**Mean**: $\mu_* = k(X_*, X)[k(X, X) + \sigma^2 I]^{-1}y$

**Covariance**: $\Sigma_* = k(X_*, X_*) - k(X_*, X)[k(X, X) + \sigma^2 I]^{-1}k(X, X_*)$

### Common Kernels

**RBF**: $k(x, x') = \sigma^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)$

**Matérn**: $k(x, x') = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}r}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}r}{\ell}\right)$

## 10. Hidden Markov Models

### Model

- **Initial**: $p(z_1) = \pi$
- **Transition**: $p(z_t | z_{t-1}) = A$
- **Emission**: $p(x_t | z_t) = B$

### Forward Algorithm

$$\alpha_t(j) = p(x_1, \ldots, x_t, z_t = j)$$
$$\alpha_t(j) = \left[\sum_i \alpha_{t-1}(i) A_{ij}\right] B_j(x_t)$$

### Backward Algorithm

$$\beta_t(i) = p(x_{t+1}, \ldots, x_T | z_t = i)$$
$$\beta_t(i) = \sum_j A_{ij} B_j(x_{t+1}) \beta_{t+1}(j)$$

### Viterbi Algorithm

**Most likely state sequence**:
$$\delta_t(j) = \max_{z_1, \ldots, z_{t-1}} p(z_1, \ldots, z_{t-1}, z_t = j, x_1, \ldots, x_t)$$

## 11. Bayesian Neural Networks

### Weight Uncertainty

$$p(y | x, D) = \int p(y | x, w) p(w | D) dw$$

### Variational Approximation

$$q_\phi(w) = \prod_i \mathcal{N}(w_i | \mu_i, \sigma_i^2)$$

**ELBO**:
$$\mathcal{L} = \mathbb{E}_{q(w)}[\log p(D|w)] - D_{KL}(q(w) \| p(w))$$

### MC Dropout

Dropout at test time approximates Bayesian inference:
$$p(y|x) \approx \frac{1}{T}\sum_{t=1}^T p(y | x, w^{(t)})$$

## 12. Normalizing Flows

### Change of Variables

If $z = f(x)$ is invertible:
$$p_X(x) = p_Z(f(x)) \left|\det \frac{\partial f}{\partial x}\right|$$

### Flow Composition

$$x = f_K \circ f_{K-1} \circ \cdots \circ f_1(z)$$

$$\log p(x) = \log p(z) - \sum_{k=1}^K \log \left|\det \frac{\partial f_k}{\partial h_{k-1}}\right|$$

### Common Flows

**RealNVP**: Affine coupling layers
**GLOW**: 1×1 convolutions
**Neural ODE**: Continuous normalizing flows

## ML Connections

| Probabilistic Concept | Application                           |
| --------------------- | ------------------------------------- |
| Bayes' theorem        | Posterior inference, updating beliefs |
| Graphical models      | Structured prediction, causality      |
| EM algorithm          | GMM clustering, missing data          |
| Variational inference | VAE, approximate Bayesian             |
| MCMC                  | Uncertainty quantification            |
| Gaussian processes    | Bayesian optimization                 |
| HMM                   | Speech recognition, sequences         |
| Normalizing flows     | Density estimation, generation        |

## Key Equations Summary

| Concept      | Equation                                                            |
| ------------ | ------------------------------------------------------------------- | -------------- | ----------------- |
| Bayes        | $p(\theta                                                           | D) \propto p(D | \theta)p(\theta)$ |
| ELBO         | $\mathcal{L} = \mathbb{E}_q[\log p(x,z)] - \mathbb{E}_q[\log q(z)]$ |
| VAE KL       | $-\frac{1}{2}\sum(1 + \log\sigma^2 - \mu^2 - \sigma^2)$             |
| MH accept    | $\alpha = \min(1, \frac{p(x')q(x                                    | x')}{p(x)q(x'  | x)})$             |
| GP posterior | $\mu_* = K_*K^{-1}y$                                                |
| Forward      | $\alpha_t(j) = [\sum_i \alpha_{t-1}(i)A_{ij}]B_j(x_t)$              |
| Flow         | $\log p(x) = \log p(z) - \sum_k \log                                | \det J_k       | $                 |

## References

1. Bishop - "Pattern Recognition and Machine Learning"
2. Murphy - "Machine Learning: A Probabilistic Perspective"
3. Koller & Friedman - "Probabilistic Graphical Models"
4. Blei, Kucukelbir, McAuliffe - "Variational Inference: A Review"
