# Kernel Methods

[← Previous: Hilbert Spaces](../03-Hilbert-Spaces) | [Next: ML-Specific Math →](../../13-ML-Specific-Math)

## Overview

Kernel methods are a powerful class of algorithms that enable learning in high-dimensional feature spaces without explicitly computing the feature transformations. This "kernel trick" is fundamental to SVMs, Gaussian processes, and many other ML algorithms.

## Why This Matters for Machine Learning

Kernel methods represent one of the most elegant ideas in machine learning: transforming hard nonlinear problems into easy linear ones. The kernel trick lets you work in spaces of enormous (even infinite) dimension while only ever computing dot products—a $O(n)$ operation instead of the exponential cost of explicit feature computation. This mathematical sleight of hand powered the SVM revolution of the 2000s.

But kernels aren't just historical artifacts. Gaussian processes—the Bayesian approach to function learning—are entirely kernel-based, and they provide something deep learning struggles with: principled uncertainty estimates. When a GP says "I don't know," it means it. The choice of kernel encodes your prior beliefs about function smoothness, periodicity, and other properties in a mathematically rigorous way.

Most surprisingly, kernels have re-emerged at the heart of deep learning theory. The neural tangent kernel (NTK) shows that infinitely wide neural networks are equivalent to kernel machines, explaining why overparameterized networks generalize. Understanding kernels is now essential for understanding why deep learning works at all.

## Chapter Roadmap

- **Section 1-2**: Foundations—the kernel trick, feature maps, and positive definite kernels with Mercer's theorem
- **Section 3-4**: Kernel catalog—linear, polynomial, RBF, Matérn, string kernels, and kernel composition rules
- **Section 5-6**: Core algorithms—SVMs (hard/soft margin) and kernel ridge regression with the representer theorem
- **Section 7-8**: Unsupervised methods—kernel PCA and Gaussian processes with posterior prediction
- **Section 9-10**: Scalability—random Fourier features, Nyström approximation, and multiple kernel learning
- **Section 11-12**: Deep connections—neural tangent kernel, deep kernel learning, and computational tricks

## Files in This Section

| File | Description |
|------|-------------|
| [theory.ipynb](theory.ipynb) | Interactive examples with visualizations |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions |

## Prerequisites

- Linear algebra
- Hilbert spaces and inner products
- Optimization basics
- Basic machine learning concepts

## Learning Objectives

1. Understand the kernel trick and its mathematical foundation
2. Master common kernels and their properties
3. Implement kernel-based algorithms from scratch
4. Apply Mercer's theorem and kernel design
5. Connect kernels to Gaussian processes

---

## 1. The Kernel Trick

### Motivation

Many ML algorithms depend only on inner products $\langle x, x' \rangle$:

- Perceptron: $w = \sum_i \alpha_i y_i x_i$, decision: $\text{sign}(\sum_i \alpha_i y_i \langle x_i, x \rangle)$
- PCA: Uses covariance matrix $X^TX$
- Ridge regression: $(X^TX + \lambda I)^{-1}X^Ty$

### Feature Maps

Transform data: $\phi: \mathcal{X} \to \mathcal{H}$ (feature space)

Example: Polynomial features for $x \in \mathbb{R}^2$:
$$\phi(x) = (x_1^2, \sqrt{2}x_1x_2, x_2^2, \sqrt{2}x_1, \sqrt{2}x_2, 1)$$

### The Kernel Function

Instead of computing $\phi(x)$ explicitly:
$$K(x, x') = \langle \phi(x), \phi(x') \rangle$$

For the polynomial example:
$$K(x, x') = (x^T x' + 1)^2$$

### Key Insight

Compute $K(x, x')$ directly in $O(d)$ time, even when $\phi(x)$ is infinite-dimensional!

> 💡 **Insight:** The kernel trick is about computational complexity, not mathematical generality. Any algorithm that only uses inner products can be "kernelized." This includes perceptrons, PCA, CCA, k-means, and many more. When you see $x^T x'$ in an algorithm, you can replace it with $K(x, x')$ and suddenly work in an infinite-dimensional space.

---

## 2. Positive Definite Kernels

### Definition

$K: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is a **positive definite kernel** if:

1. **Symmetric**: $K(x, x') = K(x', x)$
2. **Positive semi-definite**: For any $\{x_1, ..., x_n\}$ and $c \in \mathbb{R}^n$:
   $$\sum_{i,j} c_i c_j K(x_i, x_j) \geq 0$$

Equivalently: Gram matrix $K_{ij} = K(x_i, x_j)$ is PSD.

### Mercer's Theorem

If $K$ is continuous and positive definite on compact $\mathcal{X}$:
$$K(x, x') = \sum_{i=1}^\infty \lambda_i \phi_i(x) \phi_i(x')$$

where $\lambda_i > 0$ and $\phi_i$ are eigenfunctions.

### Moore-Aronszajn Theorem

Every PD kernel defines a unique RKHS $\mathcal{H}_K$ with:
$$K(x, \cdot) \in \mathcal{H}_K \quad \text{and} \quad f(x) = \langle f, K(x, \cdot) \rangle_{\mathcal{H}_K}$$

---

## 3. Common Kernels

### Linear Kernel

$$K(x, x') = x^T x'$$

- Feature space: $\mathcal{H} = \mathbb{R}^d$
- Use when: Data is linearly separable

### Polynomial Kernel

$$K(x, x') = (x^T x' + c)^d$$

- Features: All monomials up to degree $d$
- Dimension: $\binom{d + p}{p}$ for $x \in \mathbb{R}^p$
- Parameters: degree $d$, offset $c \geq 0$

### RBF (Gaussian) Kernel

$$K(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)$$

- Infinite-dimensional feature space
- Universal approximator
- Parameters: bandwidth $\sigma$ (or $\gamma = 1/(2\sigma^2)$)

### Laplacian Kernel

$$K(x, x') = \exp\left(-\frac{\|x - x'\|_1}{\sigma}\right)$$

- Less smooth than RBF
- Better for certain types of data

### Matérn Kernels

$$K(x, x') = \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}\|x-x'\|}{\ell}\right)^\nu B_\nu\left(\frac{\sqrt{2\nu}\|x-x'\|}{\ell}\right)$$

- $\nu$ controls smoothness
- $\nu = 1/2$: Laplacian
- $\nu \to \infty$: RBF
- Common: $\nu \in \{1/2, 3/2, 5/2\}$

### String Kernels

For sequences $s, s'$:
$$K(s, s') = \sum_u \phi_u(s) \phi_u(s')$$

where $\phi_u$ counts subsequence $u$.

---

## 4. Kernel Operations

### Valid Kernel Constructions

If $K_1, K_2$ are valid kernels:

| Operation   | Formula          | Valid? |
| ----------- | ---------------- | ------ |
| Sum         | $K_1 + K_2$      | ✓      |
| Product     | $K_1 \cdot K_2$  | ✓      |
| Scaling     | $cK$ for $c > 0$ | ✓      |
| Polynomial  | $K^n$            | ✓      |
| Composition | $K(f(x), f(x'))$ | ✓      |
| Exponential | $\exp(K)$        | ✓      |

### Normalization

$$\tilde{K}(x, x') = \frac{K(x, x')}{\sqrt{K(x, x)K(x', x')}}$$

Ensures $\tilde{K}(x, x) = 1$ (cosine similarity interpretation).

---

## 5. Support Vector Machines

### Hard-Margin SVM

$$\min_{w, b} \frac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^T\phi(x_i) + b) \geq 1$$

### Dual Problem

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$
$$\text{s.t.} \quad \alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0$$

### Decision Function

$$f(x) = \sum_{i \in SV} \alpha_i y_i K(x_i, x) + b$$

Only support vectors ($\alpha_i > 0$) contribute.

### Soft-Margin SVM

Add slack variables:
$$\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum_i \xi_i$$
$$\text{s.t.} \quad y_i(w^T\phi(x_i) + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

Dual: $0 \leq \alpha_i \leq C$ (box constraint).

---

## 6. Kernel Ridge Regression

### Primal Problem

$$\min_f \sum_{i=1}^n (y_i - f(x_i))^2 + \lambda \|f\|_{\mathcal{H}}^2$$

### Representer Theorem

Solution has form:
$$f^*(x) = \sum_{i=1}^n \alpha_i K(x_i, x)$$

### Closed-Form Solution

$$\alpha = (K + \lambda I)^{-1} y$$

Prediction: $f(x_{new}) = k_{new}^T (K + \lambda I)^{-1} y$

where $k_{new} = [K(x_1, x_{new}), ..., K(x_n, x_{new})]^T$

---

## 7. Kernel PCA

### Standard PCA

Find principal components maximizing variance:
$$w_k = \arg\max_{\|w\|=1} \text{Var}(w^T X)$$

### Kernelized PCA

In feature space: find eigenvectors of $\Phi^T \Phi$

Using kernel: eigendecomposition of centered $K$:
$$\tilde{K} = K - \mathbf{1}_n K - K \mathbf{1}_n + \mathbf{1}_n K \mathbf{1}_n$$

where $\mathbf{1}_n$ is $n \times n$ matrix of $1/n$.

### Projection

For new point $x$:
$$\phi_k(x) = \sum_{i=1}^n \alpha_i^{(k)} K(x_i, x)$$

where $\alpha^{(k)}$ is the $k$-th eigenvector.

---

## 8. Gaussian Processes

### Definition

A GP is a distribution over functions:
$$f \sim \mathcal{GP}(m(x), K(x, x'))$$

- $m(x) = \mathbb{E}[f(x)]$ (mean function)
- $K(x, x') = \text{Cov}(f(x), f(x'))$ (covariance/kernel)

### Posterior Prediction

Given observations $(X, y)$, predict at $X_*$:

**Mean**: $\mu_* = K_*^T (K + \sigma^2 I)^{-1} y$

**Covariance**: $\Sigma_* = K_{**} - K_*^T (K + \sigma^2 I)^{-1} K_*$

where:

- $K = K(X, X)$
- $K_* = K(X, X_*)$
- $K_{**} = K(X_*, X_*)$

### Kernel = Prior Beliefs

| Kernel     | Smoothness          | Stationarity   |
| ---------- | ------------------- | -------------- |
| RBF        | Very smooth         | Stationary     |
| Matérn 3/2 | Once differentiable | Stationary     |
| Linear     | Smooth              | Non-stationary |
| Periodic   | Smooth              | Periodic       |

> 💡 **Insight:** Your choice of kernel encodes strong assumptions about the function you're learning. RBF assumes infinite smoothness—great for many applications but unrealistic for discontinuous phenomena. Matérn kernels let you dial in exactly how smooth you expect your function to be. The bandwidth parameter $\sigma$ controls "locality": small $\sigma$ means only nearby points matter, large $\sigma$ means global structure dominates.

---

## 9. Kernel Approximations

### Random Fourier Features

For shift-invariant kernels $K(x, x') = k(x - x')$:

$$K(x, x') \approx \frac{1}{D}\sum_{j=1}^D \cos(\omega_j^T x + b_j)\cos(\omega_j^T x' + b_j)$$

where $\omega_j \sim p(\omega)$ (spectral density), $b_j \sim \text{Uniform}[0, 2\pi]$

**Feature map**: $z(x) = \sqrt{\frac{2}{D}}[\cos(\omega_1^T x + b_1), ..., \cos(\omega_D^T x + b_D)]$

### Nyström Approximation

Select $m$ landmark points $\{z_1, ..., z_m\}$:

$$K \approx K_{nm} K_{mm}^{-1} K_{mn}$$

where $K_{nm} = K(X, Z)$, $K_{mm} = K(Z, Z)$

### Benefits

- Reduce complexity from $O(n^3)$ to $O(nm^2)$ or $O(nD^2)$
- Enable mini-batch training
- Scale to millions of points

---

## 10. Multiple Kernel Learning

### Problem

Learn optimal combination of kernels:
$$K(x, x') = \sum_{m=1}^M \mu_m K_m(x, x')$$

where $\mu_m \geq 0$, $\sum_m \mu_m = 1$

### SimpleMKL

Alternating optimization:

1. Fix $\mu$, solve standard SVM
2. Fix SVM, update $\mu$ via gradient descent

### Applications

- Combine different feature types
- Automatic kernel selection
- Multi-modal learning

---

## 11. Deep Kernels

### Neural Tangent Kernel (NTK)

At infinite width, neural network training = kernel regression with:
$$K_{NTK}(x, x') = \nabla_\theta f(x; \theta)^T \nabla_\theta f(x'; \theta)$$

> 💡 **Insight:** The NTK reveals something profound: neural networks at infinite width are lazy—their features don't change during training! The network function evolves linearly in the tangent space, making it exactly a kernel method. This explains the "double descent" phenomenon and why overparameterized networks can generalize. The NTK is determined entirely by the architecture, connecting network design to function space geometry.

### Deep Kernel Learning

Combine neural networks with GPs:
$$K(x, x') = K_{base}(g_\theta(x), g_\theta(x'))$$

where $g_\theta$ is a neural network.

### Convolutional Kernels

$$K(x, x') = \sum_{patches} K_{patch}(\text{patch}_i(x), \text{patch}_j(x'))$$

---

## 12. Computational Considerations

### Complexity

| Operation     | Time             | Space    |
| ------------- | ---------------- | -------- |
| Kernel matrix | $O(n^2d)$        | $O(n^2)$ |
| Cholesky      | $O(n^3)$         | $O(n^2)$ |
| Prediction    | $O(n)$ per point | -        |

### Scaling Techniques

1. **Sparse approximations**: Inducing points, subset of regressors
2. **Low-rank approximations**: Nyström, RFF
3. **Structured kernels**: Kronecker, Toeplitz
4. **Stochastic methods**: SGD for kernel methods

### Implementation Tips

- Use Cholesky instead of matrix inverse
- Cache kernel matrix when possible
- Consider kernel caching for SVM
- Use parallel computation for kernel matrix

---

## Summary

| Concept             | Key Formula                                            | Application                 |
| ------------------- | ------------------------------------------------------ | --------------------------- |
| Kernel trick        | $K(x,x') = \langle\phi(x), \phi(x')\rangle$            | All kernel methods          |
| RBF kernel          | $\exp(-\|x-x'\|^2/2\sigma^2)$                          | Universal approximator      |
| Representer theorem | $f^* = \sum_i \alpha_i K(x_i, \cdot)$                  | Solution form               |
| KRR                 | $\alpha = (K + \lambda I)^{-1}y$                       | Regression                  |
| SVM                 | $\max \sum\alpha_i - \frac{1}{2}\alpha^T Y K Y \alpha$ | Classification              |
| GP posterior        | $\mu_* = K_*^T(K + \sigma^2I)^{-1}y$                   | Prediction with uncertainty |

## Key Takeaways

- **The kernel trick trades explicit features for implicit ones**: You get the expressive power of a high-dimensional feature space while only computing $O(n^2)$ kernel evaluations.

- **Positive definiteness is the only requirement**: Any symmetric function that produces positive semi-definite Gram matrices is a valid kernel, giving immense flexibility in kernel design.

- **The representer theorem guarantees tractability**: No matter how complex your RKHS, the solution is always a linear combination of $n$ kernel evaluations—tractable computation in infinite dimensions.

- **SVMs find the maximum margin hyperplane in feature space**: The kernel transforms the problem; the optimization finds the widest possible "street" separating classes.

- **Gaussian processes are Bayesian kernel regression**: GPs give you uncertainty for free—the posterior variance tells you where the model is confident and where it's guessing.

- **Kernel approximations enable scaling**: Random Fourier features and Nyström approximation reduce the $O(n^3)$ bottleneck, making kernel methods applicable to large datasets.

- **Neural networks at infinite width are kernel machines**: The NTK unifies deep learning and kernel methods, explaining generalization in overparameterized networks.

## Key Theorems

1. **Mercer**: PD kernel = sum of eigenfunctions
2. **Moore-Aronszajn**: PD kernel ↔ unique RKHS
3. **Representer**: Optimal solution is kernel expansion
4. **Universal approximation**: RBF can approximate any continuous function

## Exercises

1. **Kernel Verification**: Prove that the polynomial kernel $K(x, x') = (x^T x' + 1)^2$ is a valid positive definite kernel by explicitly finding a feature map $\phi$ such that $K(x, x') = \langle \phi(x), \phi(x') \rangle$.

2. **Kernel Composition**: Given two valid kernels $K_1$ and $K_2$, prove that $K(x, x') = K_1(x, x') \cdot K_2(x, x')$ is also a valid positive definite kernel. What is the corresponding feature space?

3. **RBF Bandwidth Selection**: For the RBF kernel $K(x, x') = \exp(-\|x - x'\|^2 / 2\sigma^2)$, explain the effect of $\sigma$ on the decision boundary. What happens as $\sigma \to 0$ and $\sigma \to \infty$?

4. **Kernel Ridge Regression Implementation**: Given training data $(X, y)$ and test point $x_*$, derive the prediction formula $f(x_*) = k_*^T (K + \lambda I)^{-1} y$. Implement this for the RBF kernel and compare with linear regression on non-linear data.

5. **Gaussian Process Uncertainty**: For a GP with RBF kernel, explain why the posterior variance $\sigma^2_* = K(x_*, x_*) - k_*^T (K + \sigma^2 I)^{-1} k_*$ is larger far from training points. How does this relate to the kernel's "locality"?

## References

- Schölkopf & Smola, "Learning with Kernels"
- Rasmussen & Williams, "Gaussian Processes for Machine Learning"
- Shawe-Taylor & Cristianini, "Kernel Methods for Pattern Analysis"
- Hofmann, Schölkopf & Smola, "Kernel Methods in Machine Learning"
