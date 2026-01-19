# KL Divergence

## Introduction

Kullback-Leibler (KL) divergence measures how one probability distribution differs from another. It's fundamental to machine learning, appearing in variational inference, information bottleneck, loss functions, and regularization. Understanding KL divergence is essential for modern ML practitioners.

## Prerequisites

- Entropy basics
- Probability distributions
- Expectation and logarithms

## Learning Objectives

1. Understand KL divergence definition and properties
2. Compute KL divergence for common distributions
3. Apply KL divergence in ML contexts
4. Distinguish forward vs reverse KL

---

## 1. Definition

### 1.1 Discrete KL Divergence

For discrete distributions $P$ and $Q$:

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_P\left[\log \frac{P(X)}{Q(X)}\right]$$

### 1.2 Continuous KL Divergence

For continuous distributions with densities $p$ and $q$:

$$D_{KL}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} dx$$

### 1.3 Interpretation

| Interpretation        | Description                                            |
| --------------------- | ------------------------------------------------------ |
| Information loss      | Extra bits needed when using $Q$ to code data from $P$ |
| Surprise difference   | Expected extra surprise using $Q$ instead of $P$       |
| Distribution distance | How different $Q$ is from $P$ (not symmetric!)         |

---

## 2. Properties

### 2.1 Non-negativity (Gibbs' Inequality)

$$D_{KL}(P \| Q) \geq 0$$

with equality if and only if $P = Q$ almost everywhere.

**Proof sketch:** Uses Jensen's inequality and concavity of log:
$$-D_{KL}(P \| Q) = \mathbb{E}_P\left[\log \frac{Q(X)}{P(X)}\right] \leq \log \mathbb{E}_P\left[\frac{Q(X)}{P(X)}\right] = \log 1 = 0$$

### 2.2 Asymmetry

$$D_{KL}(P \| Q) \neq D_{KL}(Q \| P) \text{ in general}$$

KL divergence is **not** a true metric/distance.

```
Example of asymmetry:

P: ████████░░░░
Q: ░░░░████████████

D_KL(P || Q) measures "surprise" of Q when true is P
D_KL(Q || P) measures "surprise" of P when true is Q

These can be very different!
```

### 2.3 Not a Metric

KL divergence violates:

1. **Symmetry:** $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$
2. **Triangle inequality:** May fail

### 2.4 Additivity for Independent Variables

For independent $(X_1, X_2)$:

$$D_{KL}(P_{X_1, X_2} \| Q_{X_1, X_2}) = D_{KL}(P_{X_1} \| Q_{X_1}) + D_{KL}(P_{X_2} \| Q_{X_2})$$

### 2.5 Chain Rule

$$D_{KL}(P(X,Y) \| Q(X,Y)) = D_{KL}(P(X) \| Q(X)) + \mathbb{E}_{P(X)}[D_{KL}(P(Y|X) \| Q(Y|X))]$$

---

## 3. Relationship to Other Quantities

### 3.1 Cross-Entropy Decomposition

$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

where $H(P, Q) = -\sum_x P(x) \log Q(x)$ is cross-entropy.

**Implication:** Minimizing cross-entropy = minimizing KL divergence (when $P$ is fixed).

### 3.2 Mutual Information

$$I(X; Y) = D_{KL}(P(X,Y) \| P(X)P(Y))$$

Mutual information is KL divergence from joint to product of marginals.

### 3.3 Information Diagram

```
          ┌─────────────────────────┐
          │       H(X,Y)            │
          │  ┌─────────┬─────────┐  │
          │  │         │         │  │
          │  │  H(X|Y) │ I(X;Y)  │  │
          │  │         │         │  │
          │  └─────────┴─────────┘  │
          └─────────────────────────┘

I(X;Y) = H(X) - H(X|Y) = D_KL(P(X,Y) || P(X)P(Y))
```

---

## 4. Forward vs Reverse KL

### 4.1 Forward KL: $D_{KL}(P \| Q)$

Minimize over $Q$: "moment matching"

$$Q^* = \arg\min_Q D_{KL}(P \| Q)$$

- Expectation under **true** distribution $P$
- $Q$ must cover all of $P$'s support (zero-avoiding)
- Tends to produce **broad** approximations

### 4.2 Reverse KL: $D_{KL}(Q \| P)$

Minimize over $Q$: "mode seeking"

$$Q^* = \arg\min_Q D_{KL}(Q \| P)$$

- Expectation under **approximate** distribution $Q$
- $Q$ can ignore parts of $P$ (zero-forcing)
- Tends to produce **narrow** approximations

### 4.3 Visual Comparison

```
True distribution P (bimodal):
      ▄▄                    ▄▄
     ████                  ████
    ██████                ██████
   ────────────────────────────

Forward KL approximation Q (covers both modes):
           ▄▄▄▄▄▄▄▄▄▄▄▄
          ████████████████
         ██████████████████
   ────────────────────────────

Reverse KL approximation Q (picks one mode):
      ▄▄▄▄▄▄
     ████████
    ██████████
   ────────────────────────────
```

### 4.4 When to Use Which

| Scenario              | Preferred KL | Reason                    |
| --------------------- | ------------ | ------------------------- |
| Variational inference | Reverse      | Tractable optimization    |
| Density estimation    | Forward      | Match all modes           |
| Generative models     | Forward      | Diversity                 |
| Policy distillation   | Reverse      | Focus on important states |

---

## 5. KL Divergence for Common Distributions

### 5.1 Bernoulli

$$D_{KL}(\text{Bern}(p) \| \text{Bern}(q)) = p \log\frac{p}{q} + (1-p)\log\frac{1-p}{1-q}$$

### 5.2 Categorical

$$D_{KL}(\text{Cat}(\mathbf{p}) \| \text{Cat}(\mathbf{q})) = \sum_{i=1}^{k} p_i \log\frac{p_i}{q_i}$$

### 5.3 Gaussian (Univariate)

$$D_{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

### 5.4 Gaussian (Multivariate)

$$D_{KL}(\mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) \| \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)) = \frac{1}{2}\left[\log\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|} - d + \text{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}_2^{-1} (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)\right]$$

### 5.5 Exponential

$$D_{KL}(\text{Exp}(\lambda_1) \| \text{Exp}(\lambda_2)) = \log\frac{\lambda_1}{\lambda_2} + \frac{\lambda_2}{\lambda_1} - 1$$

---

## 6. Applications in Machine Learning

### 6.1 Variational Inference

**Goal:** Approximate intractable posterior $p(z|x)$ with tractable $q(z)$

**ELBO (Evidence Lower Bound):**
$$\log p(x) \geq \mathbb{E}_{q(z)}[\log p(x|z)] - D_{KL}(q(z) \| p(z))$$

Maximizing ELBO = minimizing reverse KL to posterior.

### 6.2 Variational Autoencoders (VAE)

$$\mathcal{L} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) \| p(z))$$

- First term: reconstruction
- Second term: regularization (keeps $q$ close to prior)

### 6.3 Information Bottleneck

Find representation $Z$ that:

- Preserves information about $Y$: maximize $I(Z; Y)$
- Compresses $X$: minimize $I(X; Z)$

$$\mathcal{L} = I(Z; Y) - \beta I(X; Z)$$

### 6.4 Knowledge Distillation

Transfer knowledge from teacher $T$ to student $S$:

$$\mathcal{L} = D_{KL}(p_T(y|x) \| p_S(y|x))$$

(with temperature scaling)

### 6.5 Regularization in Neural Networks

**KL regularization** toward prior:
$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda D_{KL}(q(w) \| p(w))$$

Used in Bayesian neural networks.

---

## 7. Estimating KL Divergence

### 7.1 From Samples (Same Distribution)

If samples from both $P$ and $Q$:

$$\hat{D}_{KL} = \frac{1}{n}\sum_{i=1}^{n} \log \frac{\hat{p}(x_i)}{\hat{q}(x_i)}$$

where $\hat{p}, \hat{q}$ are density estimates.

### 7.2 Monte Carlo Estimation

If we can sample from $P$ and evaluate $q(x)$:

$$\hat{D}_{KL}(P \| Q) = \frac{1}{n}\sum_{i=1}^{n} \log \frac{p(x_i)}{q(x_i)}, \quad x_i \sim P$$

### 7.3 Variational Lower Bound

For any function $f$:

$$D_{KL}(P \| Q) \geq \mathbb{E}_P[f(X)] - \log \mathbb{E}_Q[e^{f(X)}]$$

This is the Donsker-Varadhan representation.

---

## 8. Related Divergences

### 8.1 Jensen-Shannon Divergence

Symmetric version of KL:

$$D_{JS}(P \| Q) = \frac{1}{2}D_{KL}(P \| M) + \frac{1}{2}D_{KL}(Q \| M)$$

where $M = \frac{1}{2}(P + Q)$.

Properties:

- Symmetric: $D_{JS}(P \| Q) = D_{JS}(Q \| P)$
- Bounded: $0 \leq D_{JS} \leq 1$ (with log base 2)
- $\sqrt{D_{JS}}$ is a proper metric

### 8.2 f-Divergences

General family including KL:

$$D_f(P \| Q) = \int q(x) f\left(\frac{p(x)}{q(x)}\right) dx$$

| Divergence      | $f(t)$                             |
| --------------- | ---------------------------------- | --- | --- |
| KL              | $t \log t$                         |
| Reverse KL      | $-\log t$                          |
| JS              | $t\log t - (t+1)\log\frac{t+1}{2}$ |
| Total variation | $\frac{1}{2}                       | t-1 | $   |
| $\chi^2$        | $(t-1)^2$                          |

### 8.3 Rényi Divergence

$$D_\alpha(P \| Q) = \frac{1}{\alpha - 1} \log \sum_x p(x)^\alpha q(x)^{1-\alpha}$$

KL is limit as $\alpha \to 1$.

---

## 9. Practical Considerations

### 9.1 Numerical Stability

```python
# Unstable
kl = sum(p * log(p / q))

# Stable
kl = sum(p * (log(p + eps) - log(q + eps)))
# or
kl = sum(p * log(p + eps)) - sum(p * log(q + eps))
```

### 9.2 Support Mismatch

If $P(x) > 0$ but $Q(x) = 0$: $D_{KL}(P \| Q) = \infty$

**Solutions:**

- Add small $\epsilon$ to $Q$
- Use smoothed distributions
- Consider alternative divergences (e.g., JS)

### 9.3 High Dimensions

KL can be difficult to estimate in high dimensions due to:

- Density estimation challenges
- Curse of dimensionality
- Variance of estimators

---

## 10. Summary

| Property     | KL Divergence                  |
| ------------ | ------------------------------ |
| Definition   | $\mathbb{E}_P[\log(P/Q)]$      |
| Non-negative | Yes (Gibbs' inequality)        |
| Symmetric    | No                             |
| Metric       | No                             |
| Zero iff     | $P = Q$                        |
| Forward      | Zero-avoiding, moment matching |
| Reverse      | Zero-forcing, mode seeking     |

**Key insight:** KL divergence measures information loss when approximating $P$ with $Q$. Its asymmetry leads to different behaviors (forward vs reverse) crucial for choosing the right objective in ML applications.

---

## References

1. Cover & Thomas - "Elements of Information Theory"
2. Murphy - "Machine Learning: A Probabilistic Perspective"
3. Bishop - "Pattern Recognition and Machine Learning"
4. Blei et al. - "Variational Inference: A Review for Statisticians"
