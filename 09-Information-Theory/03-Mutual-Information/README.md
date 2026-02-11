# Mutual Information

> **Navigation**: [← 02-KL-Divergence](../02-KL-Divergence/) | [Information Theory](../) | [04-Cross-Entropy →](../04-Cross-Entropy/)

**Files in this section:**
- [theory.ipynb](theory.ipynb) - 12 worked examples
- [exercises.ipynb](exercises.ipynb) - 10 practice problems with solutions

---

## Introduction

Mutual information quantifies the amount of information obtained about one random variable through observing another. It's a fundamental concept connecting probability, information theory, and machine learning, appearing in feature selection, representation learning, generative models, and neural network analysis.

## Prerequisites
- Entropy fundamentals
- KL divergence
- Joint and conditional probability

## Learning Objectives
1. Understand mutual information definition and interpretations
2. Compute MI for discrete and continuous variables
3. Apply MI to feature selection and representation learning
4. Understand connections to entropy and KL divergence

---

## 1. Definition

### 1.1 Formal Definition

Mutual information between random variables $X$ and $Y$:

$$I(X; Y) = \sum_{x,y} P(X=x, Y=y) \log \frac{P(X=x, Y=y)}{P(X=x)P(Y=y)}$$

For continuous variables:

$$I(X; Y) = \int \int p(x,y) \log \frac{p(x,y)}{p(x)p(y)} dx \, dy$$

### 1.2 Equivalent Forms

| Form | Expression |
|------|------------|
| KL divergence | $D_{KL}(P(X,Y) \| P(X)P(Y))$ |
| Entropy difference 1 | $H(X) - H(X|Y)$ |
| Entropy difference 2 | $H(Y) - H(Y|X)$ |
| Joint entropy | $H(X) + H(Y) - H(X,Y)$ |

All forms are equivalent!

### 1.3 Interpretation

$$I(X; Y) = \text{Reduction in uncertainty about } X \text{ when } Y \text{ is known}$$

```
                  ┌─────────────────────────────────────┐
                  │           H(X, Y)                   │
                  │  ┌─────────────┬─────────────┐      │
                  │  │             │             │      │
                  │  │   H(X|Y)    │   I(X;Y)    │ H(Y|X)
                  │  │             │             │      │
                  │  └─────────────┴─────────────┘      │
                  └─────────────────────────────────────┘

                  ├─────── H(X) ────────┤
                                        ├─────── H(Y) ────────┤
```

---

## 2. Properties

### 2.1 Non-negativity

$$I(X; Y) \geq 0$$

with equality iff $X$ and $Y$ are independent.

### 2.2 Symmetry

$$I(X; Y) = I(Y; X)$$

Unlike KL divergence, mutual information is symmetric.

### 2.3 Relation to Entropy

$$I(X; Y) \leq \min(H(X), H(Y))$$

Mutual information cannot exceed the entropy of either variable.

### 2.4 Chain Rule

$$I(X; Y, Z) = I(X; Y) + I(X; Z | Y)$$

where $I(X; Z | Y)$ is conditional mutual information.

### 2.5 Data Processing Inequality

For Markov chain $X \to Y \to Z$:

$$I(X; Z) \leq I(X; Y)$$

Processing cannot create information about $X$.

---

## 3. Computing Mutual Information

### 3.1 Discrete Variables

Direct computation:
$$I(X; Y) = \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$$

### 3.2 Gaussian Variables

For jointly Gaussian $(X, Y)$ with correlation $\rho$:

$$I(X; Y) = -\frac{1}{2} \log(1 - \rho^2)$$

For multivariate Gaussian:

$$I(X; Y) = \frac{1}{2} \log \frac{|\Sigma_X| |\Sigma_Y|}{|\Sigma_{XY}|}$$

### 3.3 Estimation from Samples

**Challenges:**
- Requires density estimation
- High variance in high dimensions
- Bias in finite samples

**Methods:**
1. **Histogram estimator:** Discretize and count
2. **KNN estimator:** Kraskov-Stögbauer-Grassberger
3. **Kernel density estimation:** Smooth density estimates
4. **Variational bounds:** MINE, InfoNCE

---

## 4. Conditional Mutual Information

### 4.1 Definition

$$I(X; Y | Z) = \mathbb{E}_Z[I(X; Y | Z=z)]$$

$$= \sum_z P(z) \sum_{x,y} P(x,y|z) \log \frac{P(x,y|z)}{P(x|z)P(y|z)}$$

### 4.2 Equivalent Forms

$$I(X; Y | Z) = H(X|Z) - H(X|Y,Z)$$
$$= H(X|Z) + H(Y|Z) - H(X,Y|Z)$$

### 4.3 Chain Rule

$$I(X_1, X_2, ..., X_n; Y) = \sum_{i=1}^{n} I(X_i; Y | X_1, ..., X_{i-1})$$

---

## 5. Applications in Machine Learning

### 5.1 Feature Selection

**Mutual Information Feature Selection:**

Select features $X_i$ that maximize $I(X_i; Y)$:

$$\text{Score}(X_i) = I(X_i; Y)$$

**Advantages over correlation:**
- Captures nonlinear dependencies
- Works for categorical targets
- No distribution assumptions

**Mutual Information Maximization (mRMR):**

Balance relevance and redundancy:

$$\max_{X_i} \left[ I(X_i; Y) - \frac{1}{|S|} \sum_{X_j \in S} I(X_i; X_j) \right]$$

### 5.2 Information Bottleneck

Find optimal representation $Z$ of $X$ for predicting $Y$:

$$\mathcal{L}_{IB} = I(X; Z) - \beta I(Z; Y)$$

```
    X ──────► Z ──────► Y
    
    Compress X    Preserve info
    (minimize     about Y
    I(X;Z))       (maximize I(Z;Y))
```

### 5.3 Deep Neural Networks (Information Plane)

Track information flow through network layers:

$$\text{Layer } l: I(X; T_l) \text{ vs } I(T_l; Y)$$

where $T_l$ is the representation at layer $l$.

### 5.4 InfoMax Principle

Maximize mutual information between input and representation:

$$\max_\theta I(X; f_\theta(X))$$

Used in:
- Contrastive learning (SimCLR, MoCo)
- Self-supervised learning
- Variational autoencoders

### 5.5 Generative Models

**InfoGAN:** Maximize $I(c; G(z, c))$ where $c$ is latent code

**Variational Bounds:**
- $I(X; Z) \geq \mathbb{E}_{p(x,z)}[\log q(x|z)] + H(X)$
- $I(X; Z) \leq \mathbb{E}_{p(x)}[D_{KL}(p(z|x) \| p(z))]$

---

## 6. Pointwise Mutual Information

### 6.1 Definition

For specific outcomes:

$$\text{PMI}(x, y) = \log \frac{P(x, y)}{P(x)P(y)}$$

Note: $I(X; Y) = \mathbb{E}[\text{PMI}(X, Y)]$

### 6.2 Interpretation

| PMI Value | Meaning |
|-----------|---------|
| PMI > 0 | Co-occur more than expected |
| PMI = 0 | Independent |
| PMI < 0 | Co-occur less than expected |

### 6.3 Applications

- **Word embeddings:** Word2Vec approximates PMI
- **Association mining:** Find related items
- **Collocation extraction:** Find common phrases

---

## 7. Normalized Mutual Information

### 7.1 Definition

$$\text{NMI}(X; Y) = \frac{I(X; Y)}{\sqrt{H(X) \cdot H(Y)}}$$

or

$$\text{NMI}(X; Y) = \frac{2 I(X; Y)}{H(X) + H(Y)}$$

### 7.2 Properties

- Range: $[0, 1]$
- NMI = 1 iff perfect correspondence
- NMI = 0 iff independent

### 7.3 Use in Clustering

Compare clustering results:

$$\text{NMI}(\text{Clustering}_1, \text{Clustering}_2)$$

---

## 8. Estimating Mutual Information

### 8.1 Histogram Estimator

```python
# Discretize continuous variables
bins_x, bins_y = np.histogram2d(X, Y, bins=k)
# Estimate MI from counts
MI = compute_discrete_MI(bins)
```

**Issues:**
- Choice of bins affects result
- Poor in high dimensions

### 8.2 KNN Estimator (KSG)

Kraskov-Stögbauer-Grassberger estimator:

$$\hat{I}(X; Y) = \psi(k) - \frac{1}{n}\sum_{i=1}^{n}[\psi(n_{x,i} + 1) + \psi(n_{y,i} + 1)] + \psi(n)$$

where $\psi$ is digamma function, $n_{x,i}$ is number of points within distance to $k$-th neighbor.

### 8.3 MINE (Mutual Information Neural Estimation)

Use neural network to estimate via Donsker-Varadhan bound:

$$I(X; Y) \geq \sup_\theta \mathbb{E}_{P(X,Y)}[T_\theta] - \log \mathbb{E}_{P(X)P(Y)}[e^{T_\theta}]$$

### 8.4 InfoNCE

Contrastive estimator:

$$I(X; Y) \geq \log N - \mathcal{L}_{\text{NCE}}$$

where:
$$\mathcal{L}_{\text{NCE}} = -\mathbb{E}\left[\log \frac{e^{f(x_i, y_i)}}{\frac{1}{N}\sum_{j=1}^{N} e^{f(x_i, y_j)}}\right]$$

---

## 9. Multivariate Extensions

### 9.1 Total Correlation

$$TC(X_1, ..., X_n) = D_{KL}(P(X_1, ..., X_n) \| \prod_i P(X_i))$$

$$= \sum_{i=1}^{n} H(X_i) - H(X_1, ..., X_n)$$

Measures total redundancy among all variables.

### 9.2 Interaction Information

For three variables:

$$II(X; Y; Z) = I(X; Y | Z) - I(X; Y)$$

Can be negative (synergy) or positive (redundancy).

### 9.3 Co-Information

$$CI(X_1; ...; X_n) = -\sum_{T \subseteq \{X_1, ..., X_n\}} (-1)^{|T|} H(T)$$

---

## 10. Practical Considerations

### 10.1 Computational Complexity

| Method | Complexity | Accuracy |
|--------|------------|----------|
| Histogram | O(n) | Low (bias) |
| KNN (KSG) | O(n² log n) | Medium |
| MINE | O(epochs × n) | High |
| InfoNCE | O(batch_size²) | High |

### 10.2 Finite Sample Bias

MI estimators are often biased:
- Histogram: positive bias
- KSG: lower bias but higher variance

**Correction:** Miller-Madow correction, bootstrap

### 10.3 High Dimensions

Challenges:
- Curse of dimensionality
- Density estimation fails
- Use variational bounds (MINE, InfoNCE)

---

## 11. Summary

| Concept | Definition/Formula |
|---------|---------------------|
| Mutual Information | $I(X;Y) = H(X) + H(Y) - H(X,Y)$ |
| As KL divergence | $D_{KL}(P(X,Y) \| P(X)P(Y))$ |
| Symmetry | $I(X;Y) = I(Y;X)$ |
| Non-negativity | $I(X;Y) \geq 0$ |
| Independence | $I(X;Y) = 0$ iff independent |
| Data processing | $X \to Y \to Z \Rightarrow I(X;Z) \leq I(X;Y)$ |
| Gaussian | $I(X;Y) = -\frac{1}{2}\log(1-\rho^2)$ |

**Key ML Applications:**
- Feature selection (relevance scoring)
- Information bottleneck (representation learning)
- Contrastive learning (InfoNCE objective)
- Analyzing neural networks (information plane)

---

## Exercises

1. **MI Computation**: For the joint distribution with $P(0,0)=0.3, P(0,1)=0.2, P(1,0)=0.1, P(1,1)=0.4$, compute $I(X;Y)$ using all four equivalent forms.

2. **Independence**: Prove that $I(X;Y) = 0$ if and only if $X$ and $Y$ are independent.

3. **Gaussian MI**: For bivariate Gaussian with correlation $\rho$, verify that $I(X;Y) = -\frac{1}{2}\log(1-\rho^2)$.

4. **Data Processing**: Prove the data processing inequality: if $X \to Y \to Z$ forms a Markov chain, then $I(X;Z) \leq I(X;Y)$.

5. **Feature Selection**: Given 3 features and a target, compute MI for each feature. Which would you select? How does this compare to correlation?

---

## References

1. Cover & Thomas - "Elements of Information Theory"
2. Kraskov et al. - "Estimating Mutual Information"
3. Belghazi et al. - "MINE: Mutual Information Neural Estimation"
4. Tishby & Zaslavsky - "Deep Learning and the Information Bottleneck Principle"
5. Oord et al. - "Representation Learning with Contrastive Predictive Coding"
