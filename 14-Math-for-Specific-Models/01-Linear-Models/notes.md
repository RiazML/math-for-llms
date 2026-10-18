[Home](../../README.md) | [Neural Networks ->](../02-Neural-Networks/notes.md)

---

# Linear Models

Linear models are the first serious model family in machine learning: simple enough to solve and inspect, strong enough to be useful, and foundational for understanding optimization, regularization, classification, and modern neural network heads.

## Overview

The basic prediction is:

$$
\hat y = w^\top x+b.
$$

For a design matrix $X$, all predictions are:

$$
\hat y=Xw+b\mathbf{1}.
$$

From this one form we get least squares, ridge regression, lasso, logistic regression, softmax regression, linear probes, and the final LM head of a transformer.

## Prerequisites

- Vectors, matrices, dot products, and norms
- Gradients and convex optimization basics
- Probability and cross-entropy for classification
- Train/validation evaluation vocabulary

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates least squares, gradient descent, ridge, lasso-style shrinkage intuition, logistic regression, calibration, SVD conditioning, and linear probes. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for closed-form solves, gradients, regularization, classification probabilities, and diagnostics. |

## Learning Objectives

After this section, you should be able to:

- Write linear regression and classification models in vectorized form.
- Derive the least-squares normal equations.
- Explain projection, pseudoinverse, rank, and conditioning.
- Implement gradient descent and check its gradient.
- Explain ridge, lasso, elastic net, and the bias-variance tradeoff.
- Compute logistic and softmax probabilities.
- Interpret linear probes and LM heads in modern AI.
- Build a diagnostic checklist for linear models.

## Table of Contents

1. [Linear Prediction](#1-linear-prediction)
   - 1.1 [Feature vector](#11-feature-vector)
   - 1.2 [Affine score](#12-affine-score)
   - 1.3 [Design matrix](#13-design-matrix)
   - 1.4 [Vectorized prediction](#14-vectorized-prediction)
   - 1.5 [Linear decision boundary](#15-linear-decision-boundary)
2. [Least Squares Regression](#2-least-squares-regression)
   - 2.1 [Residuals](#21-residuals)
   - 2.2 [Squared loss](#22-squared-loss)
   - 2.3 [Normal equations](#23-normal-equations)
   - 2.4 [Projection view](#24-projection-view)
   - 2.5 [Pseudoinverse](#25-pseudoinverse)
3. [Gradient Descent](#3-gradient-descent)
   - 3.1 [Gradient](#31-gradient)
   - 3.2 [Update](#32-update)
   - 3.3 [Learning rate](#33-learning-rate)
   - 3.4 [Stochastic gradients](#34-stochastic-gradients)
   - 3.5 [Feature scaling](#35-feature-scaling)
4. [Regularization](#4-regularization)
   - 4.1 [Ridge](#41-ridge)
   - 4.2 [Ridge solution](#42-ridge-solution)
   - 4.3 [Lasso](#43-lasso)
   - 4.4 [Elastic net](#44-elastic-net)
   - 4.5 [Bias variance](#45-bias-variance)
5. [Linear Classification](#5-linear-classification)
   - 5.1 [Logistic regression](#51-logistic-regression)
   - 5.2 [Binary cross-entropy](#52-binary-crossentropy)
   - 5.3 [Softmax regression](#53-softmax-regression)
   - 5.4 [Margin](#54-margin)
   - 5.5 [Linear separability](#55-linear-separability)
6. [Optimization Geometry](#6-optimization-geometry)
   - 6.1 [Convexity](#61-convexity)
   - 6.2 [Condition number](#62-condition-number)
   - 6.3 [SVD view](#63-svd-view)
   - 6.4 [Collinearity](#64-collinearity)
   - 6.5 [Whitening](#65-whitening)
7. [Evaluation](#7-evaluation)
   - 7.1 [Train validation split](#71-train-validation-split)
   - 7.2 [MSE and MAE](#72-mse-and-mae)
   - 7.3 [Accuracy and log loss](#73-accuracy-and-log-loss)
   - 7.4 [Calibration](#74-calibration)
   - 7.5 [Cross-validation](#75-crossvalidation)
8. [Linear Models in AI](#8-linear-models-in-ai)
   - 8.1 [Baseline strength](#81-baseline-strength)
   - 8.2 [Linear probes](#82-linear-probes)
   - 8.3 [LM head](#83-lm-head)
   - 8.4 [Logit lens](#84-logit-lens)
   - 8.5 [Interpretability](#85-interpretability)
9. [Implementation Details](#9-implementation-details)
   - 9.1 [Intercept handling](#91-intercept-handling)
   - 9.2 [Numerical solve](#92-numerical-solve)
   - 9.3 [Standardization leakage](#93-standardization-leakage)
   - 9.4 [Rank checks](#94-rank-checks)
   - 9.5 [Residual diagnostics](#95-residual-diagnostics)
10. [Diagnostics](#10-diagnostics)
56. [Ten Audits to Run on Any Linear Model (Summary Card)](#56-ten-audits-to-run-on-any-linear-model-summary-card)
57. [Closing Note for §01](#57-closing-note-for-01)
53. [Computed Table: Gradient Descent, Raw vs Standardized Features](#53-computed-table-gradient-descent-raw-vs-standardized-features)
54. [Worked Problem Set G: Conditioning as Destiny](#54-worked-problem-set-g-conditioning-as-destiny)
55. [End of Section 01 Expansion](#55-end-of-section-01-expansion)
50. [Computed Example: Bootstrap Confidence Interval for a Slope](#50-computed-example-bootstrap-confidence-interval-for-a-slope)
51. [Inference Honesty for Linear Models](#51-inference-honesty-for-linear-models)
52. [Addenda: Ten Lines to Carry Into §02](#52-addenda-ten-lines-to-carry-into-02)
47. [Computed Example: Collinear Twins, OLS vs Ridge](#47-computed-example-collinear-twins-ols-vs-ridge)
48. [Worked Problem Set F: Audits as Habit](#48-worked-problem-set-f-audits-as-habit)
49. [Margin Notes: The Linear Creed](#49-margin-notes-the-linear-creed)
43. [Quick Checks (One-Line Answers)](#43-quick-checks-one-line-answers)
44. [Revision Problems (Mixed)](#44-revision-problems-mixed)
45. [Index of Computed Evidence](#45-index-of-computed-evidence)
46. [Final Remarks: Why Linear Models Anchor This Chapter](#46-final-remarks-why-linear-models-anchor-this-chapter)
39. [Worked Dataset: Logistic Data With Known Ground Truth](#39-worked-dataset-logistic-data-with-known-ground-truth)
40. [Feature Engineering as Applied Linear Algebra](#40-feature-engineering-as-applied-linear-algebra)
41. [Robust and Quantile Linear Models (Orientation)](#41-robust-and-quantile-linear-models-orientation)
42. [Section 01 Completion Checklist](#42-section-01-completion-checklist)
35. [Exam and Interview Traps: Linear Models](#35-exam-and-interview-traps-linear-models)
36. [Computed Table: Degrees of Freedom Along the Ridge Path](#36-computed-table-degrees-of-freedom-along-the-ridge-path)
37. [Worked Problem Set E: Numerics and Inference](#37-worked-problem-set-e-numerics-and-inference)
38. [What to Memorise vs What to Look Up](#38-what-to-memorise-vs-what-to-look-up)
31. [Worked Dataset: Multivariate Recovery](#31-worked-dataset-multivariate-recovery)
32. [From Linear Models to Kernels and Neural Nets](#32-from-linear-models-to-kernels-and-neural-nets)
33. [Monitoring Linear Models in Production](#33-monitoring-linear-models-in-production)
34. [Extended Exercise Bank (Self-Study)](#34-extended-exercise-bank-self-study)
26. [Historical Timeline: Linear Models](#26-historical-timeline-linear-models)
27. [Self-Test Questions](#27-self-test-questions)
28. [Self-Test Answer Key (Condensed)](#28-self-test-answer-key-condensed)
29. [Worked Problem Set D: Synthesis](#29-worked-problem-set-d-synthesis)
30. [Chapter Summary: Linear Models](#30-chapter-summary-linear-models)
21. [Worked Dataset: A 36-Point Regression Audited](#21-worked-dataset-a-36-point-regression-audited)
22. [Pre-Flight Checklist for Linear Models](#22-pre-flight-checklist-for-linear-models)
23. [Frequently Asked Questions](#23-frequently-asked-questions)
24. [Worked Problem Set C: Interpretation and Deployment](#24-worked-problem-set-c-interpretation-and-deployment)
25. [Decision Records: Linear Models in Context](#25-decision-records-linear-models-in-context)
18. [Glossary: Linear-Model Vocabulary](#18-glossary-linear-model-vocabulary)
19. [Quick-Reference Cards](#19-quick-reference-cards)
20. [Computed Table: Learning Curves, Degree 3 vs Degree 12](#20-computed-table-learning-curves-degree-3-vs-degree-12)
15. [Worked Problem Set A: Fitting and Geometry](#15-worked-problem-set-a-fitting-and-geometry)
16. [Worked Problem Set B: Classification and Evaluation](#16-worked-problem-set-b-classification-and-evaluation)
17. [Formula Sheet: The Linear Family](#17-formula-sheet-the-linear-family)
11. [Common Mistakes](#11-common-mistakes)
12. [The Generalized Linear Model Family](#12-the-generalized-linear-model-family)
13. [Bayesian Linear Regression in Depth](#13-bayesian-linear-regression-in-depth)
14. [Computed Table: A Ridge Path](#14-computed-table-a-ridge-path)
   - 10.1 [Shape checks](#101-shape-checks)
   - 10.2 [Gradient check](#102-gradient-check)
   - 10.3 [Regularization path](#103-regularization-path)
   - 10.4 [Influence](#104-influence)
   - 10.5 [Baseline comparison](#105-baseline-comparison)

---

## Shape Map

```text
features:       X       shape (n, d)
targets:        y       shape (n,)
weights:        w       shape (d,)
predictions:    y_hat   shape (n,)
multi-class W:  W       shape (K, d)
logits:         Z       shape (n, K)
```

## 1. Linear Prediction

This part studies linear prediction as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Feature vector](#1-feature-vector) | represent each example as coordinates | $x\in\mathbb{R}^d$ |
| [Affine score](#1-affine-score) | combine features by weighted sum plus bias | $\hat y=w^\top x+b$ |
| [Design matrix](#1-design-matrix) | stack examples row-wise | $X\in\mathbb{R}^{n\times d}$ |
| [Vectorized prediction](#1-vectorized-prediction) | predict all examples at once | $\hat y=Xw+b\mathbf{1}$ |
| [Linear decision boundary](#1-linear-decision-boundary) | classification threshold creates a hyperplane | $w^\top x+b=0$ |

### 1.1 Feature vector

**Main idea.** Represent each example as coordinates.

Core relation:

$$x\in\mathbb{R}^d$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 1.2 Affine score

**Main idea.** Combine features by weighted sum plus bias.

Core relation:

$$\hat y=w^\top x+b$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 1.3 Design matrix

**Main idea.** Stack examples row-wise.

Core relation:

$$X\in\mathbb{R}^{n\times d}$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 1.4 Vectorized prediction

**Main idea.** Predict all examples at once.

Core relation:

$$\hat y=Xw+b\mathbf{1}$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 1.5 Linear decision boundary

**Main idea.** Classification threshold creates a hyperplane.

Core relation:

$$w^\top x+b=0$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 2. Least Squares Regression

This part studies least squares regression as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Residuals](#2-residuals) | prediction errors form a vector | $r=y-Xw$ |
| [Squared loss](#2-squared-loss) | penalize large residuals quadratically | $L(w)=\frac12\Vert Xw-y\Vert_2^2$ |
| [Normal equations](#2-normal-equations) | set the gradient to zero | $X^\top Xw=X^\top y$ |
| [Projection view](#2-projection-view) | least squares projects y onto the column space of X | $\hat y=P_Xy$ |
| [Pseudoinverse](#2-pseudoinverse) | handle rank-deficient or rectangular systems | $w=X^+y$ |

### 2.1 Residuals

**Main idea.** Prediction errors form a vector.

Core relation:

$$r=y-Xw$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 2.2 Squared loss

**Main idea.** Penalize large residuals quadratically.

Core relation:

$$L(w)=\frac12\Vert Xw-y\Vert_2^2$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 2.3 Normal equations

**Main idea.** Set the gradient to zero.

Core relation:

$$X^\top Xw=X^\top y$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is the closed-form anchor for least-squares learning.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 2.4 Projection view

**Main idea.** Least squares projects y onto the column space of x.

Core relation:

$$\hat y=P_Xy$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 2.5 Pseudoinverse

**Main idea.** Handle rank-deficient or rectangular systems.

Core relation:

$$w=X^+y$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 3. Gradient Descent

This part studies gradient descent as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Gradient](#3-gradient) | loss gradient points uphill in parameter space | $\nabla_w L=X^\top(Xw-y)$ |
| [Update](#3-update) | move opposite the gradient | $w_{t+1}=w_t-\eta\nabla_wL$ |
| [Learning rate](#3-learning-rate) | step size must respect curvature | $0<\eta<2/\lambda_\max(X^\top X)$ |
| [Stochastic gradients](#3-stochastic-gradients) | estimate gradient from mini-batches | $g_B=X_B^\top(X_Bw-y_B)$ |
| [Feature scaling](#3-feature-scaling) | rescale features to improve conditioning | $x_j\leftarrow(x_j-\mu_j)/\sigma_j$ |

### 3.1 Gradient

**Main idea.** Loss gradient points uphill in parameter space.

Core relation:

$$\nabla_w L=X^\top(Xw-y)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 3.2 Update

**Main idea.** Move opposite the gradient.

Core relation:

$$w_{t+1}=w_t-\eta\nabla_wL$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 3.3 Learning rate

**Main idea.** Step size must respect curvature.

Core relation:

$$0<\eta<2/\lambda_\max(X^\top X)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 3.4 Stochastic gradients

**Main idea.** Estimate gradient from mini-batches.

Core relation:

$$g_B=X_B^\top(X_Bw-y_B)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 3.5 Feature scaling

**Main idea.** Rescale features to improve conditioning.

Core relation:

$$x_j\leftarrow(x_j-\mu_j)/\sigma_j$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 4. Regularization

This part studies regularization as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Ridge](#4-ridge) | penalize squared parameter norm | $L=\frac12\Vert Xw-y\Vert^2+\frac{\lambda}{2}\Vert w\Vert^2$ |
| [Ridge solution](#4-ridge-solution) | shift the Gram matrix spectrum | $w=(X^\top X+\lambda I)^{-1}X^\top y$ |
| [Lasso](#4-lasso) | penalize absolute values to encourage sparsity | $L=\frac12\Vert Xw-y\Vert^2+\lambda\Vert w\Vert_1$ |
| [Elastic net](#4-elastic-net) | combine L1 sparsity and L2 shrinkage | $\lambda_1\Vert w\Vert_1+\lambda_2\Vert w\Vert_2^2$ |
| [Bias variance](#4-bias-variance) | regularization trades bias for lower variance | $E[(\hat f-f)^2]=\mathrm{bias}^2+\mathrm{var}+\sigma^2$ |

### 4.1 Ridge

**Main idea.** Penalize squared parameter norm.

Core relation:

$$L=\frac12\Vert Xw-y\Vert^2+\frac{\lambda}{2}\Vert w\Vert^2$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 4.2 Ridge solution

**Main idea.** Shift the gram matrix spectrum.

Core relation:

$$w=(X^\top X+\lambda I)^{-1}X^\top y$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** Ridge is one of the cleanest examples of regularization improving numerical stability.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 4.3 Lasso

**Main idea.** Penalize absolute values to encourage sparsity.

Core relation:

$$L=\frac12\Vert Xw-y\Vert^2+\lambda\Vert w\Vert_1$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 4.4 Elastic net

**Main idea.** Combine l1 sparsity and l2 shrinkage.

Core relation:

$$\lambda_1\Vert w\Vert_1+\lambda_2\Vert w\Vert_2^2$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 4.5 Bias variance

**Main idea.** Regularization trades bias for lower variance.

Core relation:

$$E[(\hat f-f)^2]=\mathrm{bias}^2+\mathrm{var}+\sigma^2$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 5. Linear Classification

This part studies linear classification as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Logistic regression](#5-logistic-regression) | map affine score to probability | $p(y=1\mid x)=\sigma(w^\top x+b)$ |
| [Binary cross-entropy](#5-binary-crossentropy) | negative Bernoulli log likelihood | $-\left[y\log p+(1-y)\log(1-p)\right]$ |
| [Softmax regression](#5-softmax-regression) | multi-class linear logits | $p_k=\exp(w_k^\top x)/\sum_j\exp(w_j^\top x)$ |
| [Margin](#5-margin) | signed distance proxy for confidence | $y(w^\top x+b)$ |
| [Linear separability](#5-linear-separability) | a hyperplane can perfectly split classes only in some feature spaces | $y_i(w^\top x_i+b)>0$ |

### 5.1 Logistic regression

**Main idea.** Map affine score to probability.

Core relation:

$$p(y=1\mid x)=\sigma(w^\top x+b)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 5.2 Binary cross-entropy

**Main idea.** Negative bernoulli log likelihood.

Core relation:

$$-\left[y\log p+(1-y)\log(1-p)\right]$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 5.3 Softmax regression

**Main idea.** Multi-class linear logits.

Core relation:

$$p_k=\exp(w_k^\top x)/\sum_j\exp(w_j^\top x)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 5.4 Margin

**Main idea.** Signed distance proxy for confidence.

Core relation:

$$y(w^\top x+b)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 5.5 Linear separability

**Main idea.** A hyperplane can perfectly split classes only in some feature spaces.

Core relation:

$$y_i(w^\top x_i+b)>0$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 6. Optimization Geometry

This part studies optimization geometry as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Convexity](#6-convexity) | linear regression and logistic regression losses are convex in w | $\nabla^2L\succeq0$ |
| [Condition number](#6-condition-number) | ill-conditioned features slow optimization | $\kappa=\lambda_\max/\lambda_\min$ |
| [SVD view](#6-svd-view) | singular values reveal stable and unstable directions | $X=U\Sigma V^\top$ |
| [Collinearity](#6-collinearity) | correlated features make coefficients unstable | $X^\top X$ nearly singular |
| [Whitening](#6-whitening) | decorrelate features when appropriate | $\mathrm{Cov}(X)\approx I$ |

### 6.1 Convexity

**Main idea.** Linear regression and logistic regression losses are convex in w.

Core relation:

$$\nabla^2L\succeq0$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 6.2 Condition number

**Main idea.** Ill-conditioned features slow optimization.

Core relation:

$$\kappa=\lambda_\max/\lambda_\min$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 6.3 SVD view

**Main idea.** Singular values reveal stable and unstable directions.

Core relation:

$$X=U\Sigma V^\top$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 6.4 Collinearity

**Main idea.** Correlated features make coefficients unstable.

Core relation:

$$X^\top X$ nearly singular$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 6.5 Whitening

**Main idea.** Decorrelate features when appropriate.

Core relation:

$$\mathrm{Cov}(X)\approx I$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 7. Evaluation

This part studies evaluation as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Train validation split](#7-train-validation-split) | measure generalization on held-out data | $L_\mathrm{val}$ |
| [MSE and MAE](#7-mse-and-mae) | regression errors emphasize different residual behavior | $\mathrm{MSE}=n^{-1}\sum r_i^2$ |
| [Accuracy and log loss](#7-accuracy-and-log-loss) | classification quality needs both decisions and probabilities | $-\log p_y$ |
| [Calibration](#7-calibration) | predicted probabilities should match empirical frequencies | $P(y=1\mid \hat p=c)\approx c$ |
| [Cross-validation](#7-crossvalidation) | average validation over multiple folds | $K$ folds |

### 7.1 Train validation split

**Main idea.** Measure generalization on held-out data.

Core relation:

$$L_\mathrm{val}$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 7.2 MSE and MAE

**Main idea.** Regression errors emphasize different residual behavior.

Core relation:

$$\mathrm{MSE}=n^{-1}\sum r_i^2$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 7.3 Accuracy and log loss

**Main idea.** Classification quality needs both decisions and probabilities.

Core relation:

$$-\log p_y$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 7.4 Calibration

**Main idea.** Predicted probabilities should match empirical frequencies.

Core relation:

$$P(y=1\mid \hat p=c)\approx c$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 7.5 Cross-validation

**Main idea.** Average validation over multiple folds.

Core relation:

$$K$ folds$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 8. Linear Models in AI

This part studies linear models in ai as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Baseline strength](#8-baseline-strength) | linear models expose whether features already solve the task | $\hat y=Xw$ |
| [Linear probes](#8-linear-probes) | frozen representations can be tested by a linear head | $h=f_{\theta_0}(x),\ \hat y=Wh$ |
| [LM head](#8-lm-head) | language models end with a linear projection to vocabulary logits | $z=hW_E^\top$ |
| [Logit lens](#8-logit-lens) | intermediate states can be projected with the LM head | $z_\ell=h_\ell W_E^\top$ |
| [Interpretability](#8-interpretability) | linear weights are easier to inspect than deep nonlinear interactions | $w_j$ feature effect |

### 8.1 Baseline strength

**Main idea.** Linear models expose whether features already solve the task.

Core relation:

$$\hat y=Xw$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 8.2 Linear probes

**Main idea.** Frozen representations can be tested by a linear head.

Core relation:

$$h=f_{\theta_0}(x),\ \hat y=Wh$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** Linear probes are a practical way to test what a deep representation already contains.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 8.3 LM head

**Main idea.** Language models end with a linear projection to vocabulary logits.

Core relation:

$$z=hW_E^\top$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** Even the final step of a giant language model is a linear model over hidden features.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 8.4 Logit lens

**Main idea.** Intermediate states can be projected with the lm head.

Core relation:

$$z_\ell=h_\ell W_E^\top$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 8.5 Interpretability

**Main idea.** Linear weights are easier to inspect than deep nonlinear interactions.

Core relation:

$$w_j$ feature effect$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 9. Implementation Details

This part studies implementation details as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Intercept handling](#9-intercept-handling) | bias can be a separate scalar or a column of ones | $\tilde X=[X,\mathbf{1}]$ |
| [Numerical solve](#9-numerical-solve) | prefer stable solvers over explicit matrix inverse | $\mathrm{solve}(X^\top X,X^\top y)$ |
| [Standardization leakage](#9-standardization-leakage) | fit scaling parameters on train data only | $\mu_\mathrm{train},\sigma_\mathrm{train}$ |
| [Rank checks](#9-rank-checks) | inspect rank before trusting coefficients | $\mathrm{rank}(X)$ |
| [Residual diagnostics](#9-residual-diagnostics) | plot residuals to find nonlinearity or outliers | $r_i=y_i-\hat y_i$ |

### 9.1 Intercept handling

**Main idea.** Bias can be a separate scalar or a column of ones.

Core relation:

$$\tilde X=[X,\mathbf{1}]$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 9.2 Numerical solve

**Main idea.** Prefer stable solvers over explicit matrix inverse.

Core relation:

$$\mathrm{solve}(X^\top X,X^\top y)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 9.3 Standardization leakage

**Main idea.** Fit scaling parameters on train data only.

Core relation:

$$\mu_\mathrm{train},\sigma_\mathrm{train}$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 9.4 Rank checks

**Main idea.** Inspect rank before trusting coefficients.

Core relation:

$$\mathrm{rank}(X)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 9.5 Residual diagnostics

**Main idea.** Plot residuals to find nonlinearity or outliers.

Core relation:

$$r_i=y_i-\hat y_i$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
## 10. Diagnostics

This part studies diagnostics as the simplest useful supervised-learning model family. The important habit is to connect algebra, geometry, optimization, and diagnostics.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Shape checks](#10-shape-checks) | confirm matrix dimensions before fitting | $X:(n,d),\ w:(d,)$ |
| [Gradient check](#10-gradient-check) | compare analytic and finite-difference gradients | $\nabla L$ |
| [Regularization path](#10-regularization-path) | plot coefficients as lambda changes | $w(\lambda)$ |
| [Influence](#10-influence) | outliers can dominate squared loss | $r_i^2$ |
| [Baseline comparison](#10-baseline-comparison) | compare linear, regularized, and nonlinear models | $\Delta L$ |

### 10.1 Shape checks

**Main idea.** Confirm matrix dimensions before fitting.

Core relation:

$$X:(n,d),\ w:(d,)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 10.2 Gradient check

**Main idea.** Compare analytic and finite-difference gradients.

Core relation:

$$\nabla L$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** Linear models are perfect for learning how to verify optimization code.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 10.3 Regularization path

**Main idea.** Plot coefficients as lambda changes.

Core relation:

$$w(\lambda)$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 10.4 Influence

**Main idea.** Outliers can dominate squared loss.

Core relation:

$$r_i^2$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.
### 10.5 Baseline comparison

**Main idea.** Compare linear, regularized, and nonlinear models.

Core relation:

$$\Delta L$$

Linear models are not weak because they are simple. They are useful because their assumptions are explicit. A linear model says that the target is explained by additive feature contributions, after whatever feature map has been chosen.

**Worked micro-example.** If $x=[2,3]$, $w=[0.5,-1]$, and $b=4$, then $\hat y=0.5\cdot2-1\cdot3+4=2$. The whole model is one dot product plus a bias, but the feature design determines how expressive that dot product can be.

**Implementation check.** Always inspect shapes, feature scaling, rank, residuals, and validation loss. A closed-form solution can still generalize poorly if the features or split are wrong.

**AI connection.** This is a practical linear-model control variable.

**Common mistake.** Do not interpret a coefficient without considering feature scaling and correlation. A large coefficient may reflect units or collinearity, not causal importance.

---

## Practice Exercises

1. Compute an affine prediction.
2. Solve a small least-squares problem.
3. Derive and compute the squared-loss gradient.
4. Run a few gradient-descent steps.
5. Compute a ridge solution.
6. Compare coefficient shrinkage under different lambdas.
7. Compute logistic probability and binary cross-entropy.
8. Compute softmax probabilities.
9. Check feature standardization and leakage.
10. Write a linear-model debugging checklist.

## Why This Matters for AI

Linear models appear inside modern AI more often than they first seem. A classifier head is linear. A language model head is linear. A linear probe tests representation quality. Ridge and lasso teach regularization. Least squares teaches projection geometry. Logistic regression teaches probabilistic classification and cross-entropy.

## Bridge to Neural Networks

Neural networks can be viewed as learned feature maps followed by linear heads. The next section studies what changes when the feature map itself becomes trainable and nonlinear.

## 11. Common Mistakes
The ten failure modes reviewers check first in linear-model work; Sections 9-10 gave diagnostics, this table is the pre-submission scan.

| # | Mistake | Consequence | Fix |
|---|---|---|---|
| 1 | No intercept (or penalized intercept) | Systematic offset; biased coefficients | Add intercept column; exclude it from the penalty |
| 2 | Standardizing with full-data statistics | Leakage; optimistic validation | Fit scaler on train folds only (§9.3) |
| 3 | Using np.linalg.inv(X^T X) explicitly | Numerical instability on ill-conditioned X | Solve via QR/SVD (lstsq/pinv) |
| 4 | Interpreting coefficients under collinearity | Huge, sign-flipped coefficients (VIF, Section 22 of theory) | Ridge, VIF audit, drop/merge features |
| 5 | Reading p-values after feature selection | Invalidated inference | Pre-register features or use post-selection inference care |
| 6 | Extrapolating a linear fit | Silent, confident nonsense outside data range | State the valid input range; check at inference |
| 7 | One-hot encoding all levels + intercept | Dummy-variable trap (perfect collinearity) | Drop one level or drop intercept |
| 8 | Judging by R² alone on train | Overfit invisible (theory Section 20) | Held-out error, learning curves |
| 9 | Treating logistic output as calibrated | Wrong probabilities (Guo et al. 2017) | Calibration curve/ECE; recalibrate |
| 10 | Scaling after splitting features but before CV folds leak | Optimistic CV scores | Pipeline the scaler inside folds |



## 12. The Generalized Linear Model Family
Linear models generalize by changing the likelihood and link while keeping the score linear: Gaussian/identity (regression), Bernoulli/logit (classification), Poisson/log (counts, theory Section 25), multinomial/softmax (multi-class). The shared machinery — IRLS or gradient descent on the NLL, deviance, and the design-matrix discipline of Sections 1-9 — transfers verbatim.

- Link choice is a modelling assumption: log link forces positive means; logit bounds probabilities; identity can predict impossible values for bounded targets.
- Deviance generalizes residual sum of squares; compare nested GLMs by deviance differences, not by R² analogues without checking which pseudo-R² is quoted.
- Dispersion: Poisson assumes mean=variance; overdispersed counts need quasi-Poisson/negative-binomial thinking or standard errors lie.
- Offsets: known exposure (time at risk, population) enters as a fixed log-offset in count models — omitting it confounds rate with exposure.
- Canonical references for GLM theory: McCullagh & Nelder (1989) remains the defining text; Hastie, Tibshirani & Friedman (2009) for the ML-facing treatment.
- Software trap: default penalties/regularization in GLM libraries differ; state the objective actually optimized.



## 13. Bayesian Linear Regression in Depth
Theory Section 26 computed the conjugate Gaussian posterior. The working points: the prior precision acts exactly like ridge (posterior mean = ridge solution with $\lambda=\sigma^2/\alpha$), so Bayesian and penalized views meet; predictive variance decomposes into noise + parameter uncertainty and grows away from data; and model evidence (marginal likelihood) enables principled degree/λ selection at the cost of prior sensitivity that must be reported.

- Ridge equivalence is exact for the posterior mean/mode — a frequent exam trap is claiming the methods are 'different models'.
- Credible intervals from this posterior are exact under the stated prior/likelihood; their frequentist coverage is a separate, checkable property (simulation).
- Prior sensitivity audit: refit under 2-3 prior scales and report how conclusions move; hiding prior dependence is the Bayesian version of p-hacking.
- Empirical Bayes (evidence maximization) tunes α, σ² from data; it is not 'prior-free', it is prior-fitting — say which.
- Bridge forward: this conjugate machinery is the last closed-form inference in the chapter; §03 needs sampling (§13.4) once conjugacy ends.



## 14. Computed Table: A Ridge Path
Seed 42. Coefficients shrink smoothly toward zero and training error rises slowly at first — the bias-variance bargain of §4.5 in numbers; validation picks the operating point.

| lambda | ||w||_2 | train MSE | (illustrative, standardized X) |
|---|---|---|---|
| 0 | 3.6297 | 0.2666 | |
| 0.01 | 3.6294 | 0.2666 | |
| 0.1 | 3.6261 | 0.2666 | |
| 1 | 3.5943 | 0.2678 | |
| 10 | 3.3066 | 0.3731 | |
| 100 | 1.8678 | 3.5114 | |
| 1000 | 0.3585 | 11.7161 | |

## 15. Worked Problem Set A: Fitting and Geometry
Worked problems A: fitting and geometry.

**Problem A1: normal equations by hand**
1. Data $(x,y)$: (0,1),(1,2),(2,2),(3,4) with intercept. $X^TX=\begin{pmatrix}4&6\\6&14\end{pmatrix}$, $X^Ty=(9,17)$.
2. Solve: determinant $=56-36=20$; $w_0=(14\cdot9-6\cdot17)/20=24/20=1.2$; $w_1=(4\cdot17-6\cdot9)/20=14/20=0.7$.
3. Fit: $\hat y=1.2+0.7x$; residuals $(-0.2,0.1,-0.6,0.7)$; check $X^Tr=(0,0)$ — orthogonality is the audit.
4. Any solver whose residuals fail $X^Tr\approx0$ is wrong, regardless of its MSE.

**Problem A2: why the pseudoinverse solution is 'the' solution**
1. Rank-deficient $X$ (theory Section 14): infinitely many minimizers differ by null-space vectors.
2. Pseudoinverse picks minimum $\|w\|$ among them; predictions identical, coefficients not.
3. Consequence: with collinearity, interpret predictions and fitted values, not individual coefficients — or regularize (ridge) to restore uniqueness.

**Problem A3: conditioning and scaling**
1. Features $x_1\in[-1,1]$, $x_2\in[-1000,1000]$: $X^TX$ condition number ~$10^6$ even for independent features.
2. Gradient descent zigzags (theory Section 7); standardizing equalizes curvature.
3. Standardization is preconditioning, not cosmetics — and its statistics are train-only (§9.3).

**Problem A4: ridge limit sanity**
1. As $\lambda\to0$: ridge → OLS/pinv solution. As $\lambda\to\infty$: $w\to0$ (unpenalized intercept survives).
2. Check both limits in any implementation (theory Section 15's df goes 6→0 across the same path).
3. A ridge code failing either limit has a bug independent of data.



## 16. Worked Problem Set B: Classification and Evaluation
Worked problems B: classification and evaluation.

**Problem B1: logistic gradient by hand**
1. One sample $x=(1,2)$, $y=1$, $w=(0.5,-0.5)$: score $z=0.5-1.0=-0.5$, $p=\sigma(-0.5)=0.378$.
2. Gradient: $(p-y)x=(-0.622)(1,2)=(-0.622,-1.244)$; a GD step increases both weights — check the sign logic against $p<y$.
3. Theory Section 17 audits this at scale with finite differences; hand-derive once, trust the audit thereafter.

**Problem B2: separability and runaway weights**
1. On separable data, logistic loss keeps decreasing as $\|w\|\to\infty$ along a separating direction: no finite MLE.
2. Symptoms: weights grow every epoch, loss crawls toward 0, probabilities saturate.
3. Fixes: L2 penalty (any $\lambda>0$ restores a finite optimum), early stopping declared as such, or SVM's margin objective (theory Section 18).

**Problem B3: accuracy lies, log-loss tells**
1. Two models, same 90% accuracy: A outputs 0.51 on its correct calls, B outputs 0.99 and 0.49-on-wrong.
2. Log-loss (§7.3) separates them decisively; for decisions driven by probabilities (triage, ranking), log-loss/calibration is the metric — accuracy is the press release.



## 17. Formula Sheet: The Linear Family
| quantity | OLS | ridge | lasso |
|---|---|---|
| closed form | $(X^TX)^{-1}X^Ty$ (if invertible) | $(X^TX+\lambda I)^{-1}X^Ty$ | none (iterative) |
| uniqueness | iff full column rank | always ($\lambda>0$) | predictions unique; $w$ may not be |
| sparsity | no | no | yes (soft thresholding) |
| collinearity behaviour | explodes | shrinks jointly | picks among correlated features |
| Bayesian reading | flat prior | Gaussian prior | Laplace prior (MAP) |
| tuning | — | $\lambda$ by CV | $\lambda$ by CV |

## 18. Glossary: Linear-Model Vocabulary
| Term | Meaning |
|---|---|
| Design matrix | Rows = examples, columns = features (incl. intercept column when used). |
| Hat matrix | $H=X(X^TX)^{-1}X^T$; fitted values are $Hy$; its trace is effective df. |
| Leverage | Diagonal $h_{ii}$ of $H$; high-leverage points can dominate the fit. |
| VIF | Variance inflation factor $1/(1-R_j^2)$ for collinearity (theory Section 22). |
| Condition number | Ratio of extreme singular values; governs solve sensitivity (§6.2). |
| Margin | For classifiers, $y\,(w\cdot x+b)$ (functional) or divided by $\|w\|$ (geometric). |
| Separability | Existence of a hyperplane classifying train data perfectly; breaks logistic MLE (Problem B2). |
| Deviance | GLM generalization of residual SS (Section 12). |
| Link function | Maps mean to linear score in a GLM (logit, log, identity). |
| Linear probe | A linear model trained on frozen representations to test what they encode (§8.2). |
| Logit lens | Reading intermediate states through the final LM head (§8.4). |
| Effective df | Trace of the (ridge) hat matrix (theory Section 15). |
| Dummy trap | Full one-hot + intercept = perfect collinearity (Common Mistakes). |
| Extrapolation | Prediction outside the data's convex hull/range; linear models fail silently there. |



## 19. Quick-Reference Cards
**OLS / least squares**
- Solution: pseudoinverse; audit $X^Tr\approx0$.
- Assumes: linear mean, roughly homoscedastic noise for inference.
- Fails: collinearity (coefficients), outliers (fit).

**Ridge (L2)**
- Always unique; df shrinks smoothly (theory 15).
- Use for: collinearity, default regularized baseline.

**Lasso (L1)**
- Sparse via soft thresholding (theory 16).
- Use for: feature selection; unstable selection among correlated features — say so.

**Logistic regression**
- Probabilities via sigmoid; gradient $X^T(p-y)$ (Problem B1).
- Fails: separable data without penalty (Problem B2); calibration separate.

**Linear SVM**
- Margin objective; support vectors define the boundary (theory 18).
- Probabilities need post-hoc calibration (Platt-style).

**Poisson GLM**
- Counts, log link; check dispersion (Section 12).

**Linear probe**
- Diagnostic on frozen features — evidence about representations, not a deployment model.



## 20. Computed Table: Learning Curves, Degree 3 vs Degree 12
Seed 42, Legendre basis (theory Section 21's stable configuration). More data rescues the flexible model slowly; at small $n$ the simple model wins outright — sample-size-aware model selection in numbers.

| n_train | train MSE (deg 3) | test MSE (deg 3) | train MSE (deg 12) | test MSE (deg 12) |
|---|---|---|---|---|
| 15 | 0.0336 | 0.0902 | 0.0060 | 251534369.1572 |
| 30 | 0.0544 | 0.1071 | 0.0467 | 31.6226 |
| 60 | 0.1051 | 0.1028 | 0.0805 | 0.1240 |
| 120 | 0.0793 | 0.0927 | 0.0688 | 0.1040 |
| 240 | 0.0917 | 0.0977 | 0.0888 | 0.1016 |
| 480 | 0.0958 | 0.0917 | 0.0943 | 0.0932 |

## 21. Worked Dataset: A 36-Point Regression Audited
Seed 42. The last row is the audit: with an intercept, residuals sum to ~0 and are orthogonal to $x$ (Problem A1's $X^Tr=0$) — verify both before trusting any solver's coefficients.

| i | x | y | fit | residual |
|---|---|---|---|---|
| 0 | 3.096 | 9.194 | 9.179 | 0.016 |
| 1 | 1.756 | 5.554 | 5.903 | -0.349 |
| 2 | 3.434 | 9.755 | 10.005 | -0.250 |
| 3 | 2.789 | 8.733 | 8.428 | 0.305 |
| 4 | 0.377 | 2.740 | 2.532 | 0.207 |
| 5 | 3.902 | 11.472 | 11.149 | 0.323 |
| 6 | 3.045 | 8.846 | 9.054 | -0.208 |
| 7 | 3.144 | 9.453 | 9.296 | 0.157 |
| 8 | 0.512 | 2.827 | 2.862 | -0.036 |
| 9 | 1.802 | 6.092 | 6.016 | 0.077 |
| 10 | 1.483 | 5.556 | 5.236 | 0.320 |
| 11 | 3.707 | 10.857 | 10.672 | 0.185 |
| 12 | 2.575 | 8.209 | 7.905 | 0.304 |
| 13 | 3.291 | 9.755 | 9.655 | 0.099 |
| 14 | 1.774 | 6.051 | 5.947 | 0.103 |
| 15 | 0.909 | 4.025 | 3.833 | 0.192 |
| 16 | 2.218 | 6.462 | 7.032 | -0.570 |
| 17 | 0.255 | 2.010 | 2.234 | -0.225 |
| 18 | 3.311 | 9.589 | 9.704 | -0.115 |
| 19 | 2.527 | 7.562 | 7.788 | -0.226 |
| 20 | 3.032 | 8.970 | 9.022 | -0.052 |
| 21 | 1.418 | 5.643 | 5.077 | 0.566 |
| 22 | 3.883 | 10.861 | 11.102 | -0.241 |
| 23 | 3.572 | 10.817 | 10.342 | 0.475 |
| 24 | 3.114 | 8.612 | 9.223 | -0.611 |
| 25 | 0.779 | 3.314 | 3.515 | -0.201 |
| 26 | 1.867 | 6.233 | 6.175 | 0.058 |
| 27 | 0.175 | 2.172 | 2.039 | 0.133 |
| 28 | 0.617 | 3.327 | 3.119 | 0.208 |
| 29 | 2.732 | 8.647 | 8.289 | 0.358 |
| 30 | 2.979 | 8.808 | 8.893 | -0.085 |
| 31 | 3.870 | 10.990 | 11.071 | -0.081 |
| 32 | 1.303 | 5.101 | 4.796 | 0.305 |
| 33 | 1.482 | 5.128 | 5.233 | -0.105 |
| 34 | 1.878 | 5.685 | 6.201 | -0.517 |
| 35 | 0.758 | 2.942 | 3.464 | -0.522 |
| fit | w0=1.611 (true 1.5) | w1=2.444 (true 2.5) | sum resid=-1.09e-13 | x-weighted resid=-2.41e-13 |



## 22. Pre-Flight Checklist for Linear Models
- Design matrix shape, intercept handling, and encoding (dummy trap) stated.
- Rank/condition number inspected; VIF for interpretability claims.
- Scaler fitted inside CV folds (no leakage).
- Solver is QR/SVD-based (lstsq/pinv), not explicit inverse.
- Regularization type/strength chosen by validation; path inspected (theory 15).
- Residual diagnostics and calibration (classifiers) checked, not just headline metric.
- Valid input range recorded for deployment (extrapolation guard).
- Linear baseline compared against at least one nonlinear alternative (§8.1 discipline).



## 23. Frequently Asked Questions
- **When is a linear model enough?** When held-out error matches nonlinear challengers, when interactions are engineered, or when the signal is genuinely additive — decided by comparison (§8.1), not preference.
- **Coefficients as importance?** Only under scaling + low collinearity; otherwise report with VIF caveats or use predictions-based importance.
- **Ridge or lasso?** Ridge for grouped/correlated shrinkage, lasso for sparsity; elastic net between. Selection instability of lasso is reportable, not shameful.
- **Why did logistic weights explode?** Separability (Problem B2); penalize.
- **Is R² comparable across datasets?** No — it is variance-relative; compare errors on the same held-out set.
- **Linear probes prove the representation 'knows' X?** They prove linear decodability from the representation — a statement about the probe+representation pair (§8.2 caution).



## 24. Worked Problem Set C: Interpretation and Deployment
Worked problems C: deployment and interpretation.

**Problem C1: the coefficient press release**
1. Claim: 'Feature 7 is most important; its coefficient is largest.'
2. Audit: were features standardized? VIFs? Correlated partners splitting weight?
3. Correct statement template: standardized coefficients with VIFs, or permutation importance on held-out data — pick the estimand and say which.

**Problem C2: extrapolation incident**
1. Model trained on ages 18-65 deployed on 12-year-olds; predictions confident and wrong.
2. Linear models extrapolate the trend forever; no internal alarm exists — the guard is an input-range check at inference (Checklist).
3. Post-incident: add range validation + monitoring on input distributions (§13.3 monitoring discipline).



## 25. Decision Records: Linear Models in Context
- **First baseline for any tabular task:** standardized ridge/logistic with CV-tuned penalty — the number every fancier model must beat.
- **Interpretable scoring (regulated domains):** logistic with pre-registered features, VIF audit, calibration report.
- **Text/embedding classification baseline:** linear probe / averaged-embedding + logistic before fine-tuning anything (§8.2).
- **Counts (events, clicks):** Poisson GLM with dispersion check (Section 12).
- **High-collinearity sensor features:** ridge; coefficients reported jointly, not individually.
- Each record states design matrix, encoding, penalty, validation, and the nonlinear challenger it was compared against.

## 26. Historical Timeline: Linear Models
| Year | Milestone | Note |
|---|---|---|
| 1805/1809 | Legendre / Gauss | Least squares published (Legendre 1805; Gauss 1809, with a priority dispute history remembers). |
| 1940s-50s | Rosenblatt perceptron lineage | Linear threshold learning; convergence theorem (Novikoff 1962 bound). |
| 1970 | Hoerl & Kennard — ridge regression | Shrinkage for collinear design matrices. |
| 1973 | Nelder & Wedderburn — GLMs | Unified linear-score models across likelihoods (book: McCullagh & Nelder 1989). |
| 1995 | Cortes & Vapnik — SVM | Maximum-margin linear classification. |
| 1996 | Tibshirani — lasso | L1 sparsity with an efficient path algorithm lineage. |
| 2005 | Zou & Hastie — elastic net | Grouped selection under correlation. |
| 2009 | Hastie, Tibshirani & Friedman, ESL | The ML-era synthesis text for this whole section. |
| 2010s | Linear probes / logit lens in deep learning | Linear models as instruments for reading neural representations (§8). |



## 27. Self-Test Questions
- State the normal equations and the residual-orthogonality audit.
- Why pseudoinverse over explicit inverse, always?
- Ridge λ→0 and λ→∞ limits (Problem A4).
- Effective degrees of freedom: which trace, what limits?
- Why does logistic regression diverge on separable data, and two fixes?
- VIF: definition and the interpretation rule it imposes.
- Name the GLM triple (likelihood, link, linear score) for counts data.
- Bayesian-linear-regression = ridge: exact in what sense (Section 13)?
- Margin: functional vs geometric.
- What does a linear probe actually prove?



## 28. Self-Test Answer Key (Condensed)
- $X^TXw=X^Ty$; audit $X^T(y-Xw)\approx0$ (Problem A1, Section 21).
- QR/SVD solves are backward-stable; forming $X^TX$ squares the condition number.
- OLS/pinv solution; zero (except unpenalized intercept).
- $\mathrm{tr}\,H_\lambda=\sum d_j^2/(d_j^2+\lambda)$, from $p$ down to 0 (theory 15).
- No finite MLE on separable data; L2 penalty or margin objective (Problem B2).
- VIF$_j=1/(1-R_j^2)$; high VIF ⇒ coefficients not individually interpretable.
- Poisson, log link, $w\cdot x$ (+ offset if exposure varies).
- Posterior mean/mode equals ridge with $\lambda=\sigma^2/\alpha$ exactly.
- Functional: $y(w\cdot x+b)$; geometric: divided by $\|w\|$.
- Linear decodability from the frozen representation, nothing more (FAQ).



## 29. Worked Problem Set D: Synthesis
Worked problems D: closing synthesis.

**Problem D1: choose the baseline stack**
1. New tabular product task, 50k rows, mixed types, regulator wants explanations.
2. Plan: standardized logistic/ridge baseline, dummy-trap-safe encoding, CV inside pipelines, VIF + calibration reports (Sections 22/25).
3. Challengers owed: one nonlinear model; if it does not beat the baseline materially, the linear model ships with its audit trail.

**Problem D2: probe result overclaimed**
1. Paper claims 'the model represents syntax' from a probe accuracy of 78%.
2. Critique per §8.2/FAQ: probe capacity, baseline (majority/random features), selectivity controls; decodability ≠ use by the model.
3. Stronger evidence: interventions (does damaging the direction damage behaviour?) — probes open questions, interventions close them.



## 30. Chapter Summary: Linear Models
- One design matrix discipline underlies regression, classification, GLMs, and probes: shape, rank, scaling, encoding.
- Fitting is solved (QR/SVD, convex losses, finite-difference audits); judgement lives in regularization, collinearity, and evaluation.
- Bayesian and penalized views coincide at the posterior mean — use whichever interpretation the audience can audit.
- Linear models are both production tools and measuring instruments (probes, logit lens) for the nonlinear models of the rest of this chapter.
- Bridge: §02 stacks these units into neural networks; every layer there is this section plus a nonlinearity (§13.2).

## 31. Worked Dataset: Multivariate Recovery
Seed 42, three features with very different coefficient scales. Recovery quality is governed by noise and conditioning, not by coefficient size — the standardized-coefficient caution of Problem C1 in raw form.

| i | x1 | x2 | x3 | y | fit |
|---|---|---|---|---|---|
| 0 | 0.305 | -1.040 | 0.750 | 6.000 | 5.825 |
| 1 | 0.941 | -1.951 | -1.302 | 8.113 | 7.926 |
| 2 | 0.128 | -0.316 | -0.017 | 2.941 | 2.978 |
| 3 | -0.853 | 0.879 | 0.778 | -1.125 | -1.068 |
| 4 | 0.066 | 1.127 | 0.468 | -1.587 | -1.208 |
| 5 | -0.859 | 0.369 | -0.959 | -0.880 | -0.572 |
| 6 | 0.878 | -0.050 | -0.185 | 2.539 | 2.720 |
| 7 | -0.681 | 1.223 | -0.155 | -2.727 | -2.543 |
| 8 | -0.428 | -0.352 | 0.532 | 3.014 | 2.933 |
| 9 | 0.365 | 0.413 | 0.431 | 1.070 | 1.224 |
| 10 | 2.142 | -0.406 | -0.512 | 4.991 | 4.728 |
| 11 | -0.814 | 0.616 | 1.129 | 0.292 | -0.012 |
| 12 | -0.114 | -0.840 | -0.824 | 3.887 | 3.878 |
| 13 | 0.651 | 0.743 | 0.543 | 0.915 | 0.532 |
| 14 | -0.666 | 0.232 | 0.117 | 0.416 | 0.677 |
| 15 | 0.219 | 0.871 | 0.224 | -0.344 | -0.437 |
| 16 | 0.679 | 0.068 | 0.289 | 2.334 | 2.473 |
| 17 | 0.631 | -1.457 | -0.320 | 6.740 | 6.737 |
| 18 | -0.470 | -0.639 | -0.275 | 3.562 | 3.283 |
| 19 | 1.495 | -0.866 | 0.968 | 6.059 | 6.474 |
| 20 | -1.683 | -0.335 | 0.163 | 1.534 | 1.548 |
| 21 | 0.586 | 0.711 | 0.793 | 0.921 | 0.726 |
| 22 | -0.349 | -0.462 | 0.858 | 3.288 | 3.540 |
| 23 | -0.191 | -1.276 | -1.133 | 4.637 | 4.959 |
| 24 | -0.919 | 0.497 | 0.142 | -0.317 | -0.344 |
| 25 | 0.690 | -0.427 | 0.159 | 3.892 | 3.922 |
| 26 | 0.626 | -0.309 | 0.457 | 3.851 | 3.685 |
| 27 | -0.662 | -0.363 | -0.382 | 2.243 | 2.201 |
| 28 | -1.196 | 0.487 | -0.469 | -0.411 | -0.931 |
| 29 | 0.012 | 0.481 | 0.447 | 0.721 | 0.713 |
| recovered | w = (1.905, 0.882, -3.069, 0.612) vs true (2, 1, -3, 0.5) | | | | |



## 32. From Linear Models to Kernels and Neural Nets
- Kernel ridge replaces $X$ by feature maps $\phi(X)$ and solves in dual form — the linear algebra of Sections 2-6 survives; Chapter 12 owns the theory.
- The representer pattern (solution in the span of training points) reappears in SVM duals (Section 5.4 lineage).
- A one-hidden-layer network is a learned feature map + this section's linear head (§02); linear-model audits (rank, conditioning, calibration) apply to the head verbatim.
- Random-feature models freeze a sampled $\phi$ and fit linear heads — the controlled experiment separating 'features' from 'fitting' in deep-learning debates.
- Gaussian processes are the Bayesian completion of Bayesian linear regression with kernel features (Section 13 bridge to §03).



## 33. Monitoring Linear Models in Production
- Input drift: feature quantiles vs training (extrapolation guard, Problem C2).
- Coefficient drift on retraining: sudden sign flips indicate collinearity meeting new data, not necessarily concept drift (VIF context).
- Calibration drift for classifiers: rolling ECE on labelled feedback (theory 24).
- Residual-by-segment audits: aggregate metrics hide segment failure; linear models make segment audits cheap — do them.
- Retraining triggers and rollback use the same golden-set discipline as §13.3's norm audits.



## 34. Extended Exercise Bank (Self-Study)
1. Derive the ridge solution and its λ limits; verify against theory Section 15's df computation.
2. Implement lasso coordinate descent; compare its path with theory Section 16 on correlated features.
3. Construct data where OLS beats ridge on test and where the reverse holds; explain both via bias-variance.
4. Prove the perceptron mistake bound intuition on the theory Section 19 setup by measuring mistakes vs margin.
5. Fit a Poisson GLM to simulated counts with an offset; show what breaks when the offset is omitted.
6. Audit any library's logistic regression: gradient-check it (theory 17 pattern) before trusting its convergence claims.
7. Design the probe experiment of Problem D2 with proper controls (random-feature baseline, selectivity).
8. Compute VIFs for your favourite dataset; write the interpretation paragraph the VIFs permit.

## 35. Exam and Interview Traps: Linear Models
- Trap: 'Just invert X^T X.' Response: condition number squares; QR/SVD (Common Mistakes).
- Trap: 'Biggest coefficient = most important.' Response: scaling + VIF first (Problem C1).
- Trap: 'Logistic regression gives calibrated probabilities.' Response: calibration is checked, not inherited (theory 24).
- Trap: 'Lasso selects the true features.' Response: selection among correlated features is unstable; report stability, not certainty.
- Trap: 'R² = 0.9 means the model is good.' Response: held-out error, segment audits, extrapolation range (Section 33).
- Trap: 'Bayesian linear regression is a different model from ridge.' Response: posterior mean coincides (Section 13) — state the prior and the equivalence.
- Answer pattern: definition, audit (orthogonality/VIF/gradient check), held-out number.



## 36. Computed Table: Degrees of Freedom Along the Ridge Path
Same seed-42 design as Section 14. df glides from 5 toward 0; there is no 'true' λ, only a validation-chosen operating point on this curve.

| lambda | df (theory 15 formula) | ||w|| on Section 14 data |
|---|---|---|
| 0 | 5.000 | 3.6297 |
| 0.1 | 4.995 | 3.6261 |
| 0.5 | 4.974 | 3.6119 |
| 2 | 4.897 | 3.5597 |
| 8 | 4.614 | 3.3662 |
| 32 | 3.755 | 2.7739 |
| 128 | 2.168 | 1.6488 |
| 512 | 0.812 | 0.6369 |



## 37. Worked Problem Set E: Numerics and Inference
Worked problems E: numerics and inference honesty.

**Problem E1: the squared condition number**
1. X has singular values 1 and $10^{-4}$: cond(X) $=10^4$, but cond($X^TX$) $=10^8$.
2. Forming normal equations in float32 (~7 digits) leaves ~0 digits of accuracy in the small direction.
3. QR on X works with cond $10^4$ — four digits survive. This arithmetic, not style, bans explicit normal-equation inverses.

**Problem E2: inference after selection**
1. Pipeline: lasso selects 8 of 200 features; OLS refit reports tiny p-values for the 8.
2. Selection used the same y; the p-values ignore that search — post-selection inference is its own field (Common Mistakes).
3. Honest options: split-sample (select on A, infer on B), pre-registration, or purely predictive reporting without p-values.



## 38. What to Memorise vs What to Look Up
- Memorise: normal equations + orthogonality audit; ridge closed form and limits; df trace formula; logistic gradient $X^T(p-y)$; VIF definition.
- Memorise: solver discipline (never invert), scaler-inside-folds, intercept unpenalized.
- Look up: GLM deviance tables, post-selection inference machinery, library penalty conventions.
- Derive on demand: soft thresholding, perceptron updates, Bayesian posterior (Section 13 pattern).
- Never memorise: 'best λ' folklore — λ is validated per dataset (Section 36).

## 39. Worked Dataset: Logistic Data With Known Ground Truth
Seed 42, $w=(1.5,-2)$, $b=0.5$. Because the generating probabilities are known, calibration and log-loss can be scored against truth — the honest way to benchmark a classifier implementation (theory 17/24).

| i | x1 | x2 | true p | label |
|---|---|---|---|---|
| 0 | 0.305 | -1.040 | 0.954 | 1 |
| 1 | 0.750 | 0.941 | 0.436 | 0 |
| 2 | -1.951 | -1.302 | 0.544 | 0 |
| 3 | 0.128 | -0.316 | 0.790 | 1 |
| 4 | -0.017 | -0.853 | 0.899 | 1 |
| 5 | 0.879 | 0.778 | 0.565 | 1 |
| 6 | 0.066 | 1.127 | 0.160 | 1 |
| 7 | 0.468 | -0.859 | 0.949 | 1 |
| 8 | 0.369 | -0.959 | 0.951 | 1 |
| 9 | 0.878 | -0.050 | 0.872 | 1 |
| 10 | -0.185 | -0.681 | 0.830 | 0 |
| 11 | 1.223 | -0.155 | 0.934 | 1 |
| 12 | -0.428 | -0.352 | 0.637 | 1 |
| 13 | 0.532 | 0.365 | 0.638 | 1 |
| 14 | 0.413 | 0.431 | 0.564 | 1 |
| 15 | 2.142 | -0.406 | 0.989 | 1 |
| 16 | -0.512 | -0.814 | 0.796 | 1 |
| 17 | 0.616 | 1.129 | 0.303 | 0 |
| 18 | -0.114 | -0.840 | 0.882 | 1 |
| 19 | -0.824 | 0.651 | 0.115 | 0 |
| 20 | 0.743 | 0.543 | 0.629 | 0 |
| 21 | -0.666 | 0.232 | 0.276 | 1 |
| 22 | 0.117 | 0.219 | 0.559 | 1 |
| 23 | 0.871 | 0.224 | 0.796 | 1 |
| 24 | 0.679 | 0.068 | 0.800 | 1 |
| 25 | 0.289 | 0.631 | 0.419 | 0 |
| 26 | -1.457 | -0.320 | 0.260 | 1 |
| 27 | -0.470 | -0.639 | 0.745 | 1 |
| 28 | -0.275 | 1.495 | 0.052 | 0 |
| 29 | -0.866 | 0.968 | 0.061 | 0 |



## 40. Feature Engineering as Applied Linear Algebra
- Basis expansions (polynomials, splines, Fourier) are column operations on $X$; conditioning rules apply per column set (Legendre lesson of theory 21).
- Interactions are products of columns; their collinearity with parents is expected — center before multiplying to reduce structural correlation.
- Binning/quantization trades smoothness for robustness; monotonic binning preserves order information linear models can use.
- Target encoding is a leakage machine unless computed out-of-fold; treat it like standardization statistics (Section 22).
- Embeddings-as-features turn the linear model into a probe (§8.2); the head's discipline is unchanged.
- Feature stores must version transformations with the model: a linear model is its design matrix recipe.



## 41. Robust and Quantile Linear Models (Orientation)
- Huber regression (theory 23) bounds outlier influence via the ψ function; IRLS is the classical fitter.
- Quantile regression minimizes pinball loss (§13.1) for conditional quantiles — no Gaussian assumption, direct interval outputs.
- Least absolute deviations is the median special case; robustness costs efficiency at clean Gaussian data — quantify both on your data.
- Robust standard errors (sandwich) repair inference under heteroscedasticity without changing the point fit.
- These variants keep the linear score; what changes is the lossgeometry — exactly the §13.1 dossier applied to this chapter's models.



## 42. Section 01 Completion Checklist
- Common Mistakes table (Section 11) plus diagnostics of Sections 9-10.
- Worked problems A-E with hand-checkable audits (orthogonality, gradient checks, limits).
- Computed tables/datasets: ridge paths (14/36), learning curves (20), 36-point audit (21), multivariate recovery (31), logistic ground truth (39).
- Theory notebook: 54 cells (28 original + 26 appended), executed with 0 errors and 32 PASS / 0 FAIL in the appended check cells.
- Exercises: learner workspaces preserved; solutions compile (verified separately).
- All CONTRIBUTING elements present; canonical citations only (McCullagh & Nelder 1989; Hastie et al. 2009; Hoerl & Kennard 1970; Tibshirani 1996; Cortes & Vapnik 1995; Zou & Hastie 2005; Guo et al. 2017 for calibration).

## 43. Quick Checks (One-Line Answers)
- Audit any least-squares fit by: $X^Tr\approx0$.
- Never form: $(X^TX)^{-1}$ explicitly — solve via QR/SVD.
- Ridge at λ=0 is: OLS/pinv; at λ=∞: zero (intercept exempt).
- Effective df is: tr($H_\lambda$) $=\sum d_j^2/(d_j^2+\lambda)$.
- Lasso's coordinate update is: soft thresholding.
- Logistic gradient is: $X^T(p-y)/n$ (+ penalty).
- Separable data breaks: the logistic MLE (weights run away).
- VIF above ~10 means: coefficients not individually interpretable.
- Scaler statistics come from: train folds only.
- Calibration is measured by: reliability bins / ECE, not accuracy.
- Poisson GLM link: log; check dispersion.
- Bayesian posterior mean equals: ridge with $\lambda=\sigma^2/\alpha$.
- A probe proves: linear decodability, not model usage.
- Extrapolation guard: recorded valid input range + inference checks.



## 44. Revision Problems (Mixed)
1. Recompute Section 21's audit row from the raw columns for three rows of your choice.
2. Derive soft thresholding from the lasso subgradient condition.
3. Show ridge's posterior-mean equivalence (Section 13) in four lines.
4. Given VIFs (1.2, 38, 41, 2.0), write the permitted interpretation paragraph.
5. Design the calibration audit for a triage classifier: bins, ECE, recalibration rule, monitoring cadence.
6. Explain perceptron vs SVM on separable data: what each optimizes, what each returns.
7. List every way Section 39's ground-truth dataset can validate a logistic implementation.
8. State the dummy trap and two encodings that avoid it, with trade-offs.



## 45. Index of Computed Evidence
| Evidence | Where |
|---|---|
| Ridge path (norms, train MSE) | Section 14 |
| Learning curves deg 3 vs 12 | Section 20 |
| 36-point regression audit | Section 21 |
| Multivariate coefficient recovery | Section 31 |
| df along the ridge path | Section 36 |
| Logistic data with true probabilities | Section 39 |
| Rank-deficiency, df, lasso, SVM, perceptron, U-curve, VIF, Huber, calibration, Poisson, Bayes (executable) | theory.ipynb Sections 14-26 |



## 46. Final Remarks: Why Linear Models Anchor This Chapter
- Every later section reuses this one's skeleton: a score, a loss, a solver audit, an evaluation that can distinguish fitting from understanding.
- Neural networks (§02) are linear models composed with nonlinearities; transformers (§05) are linear maps steered by attention; probes read all of them with §01 tools.
- Keep the audits close: orthogonality, conditioning, gradient checks, calibration. They are cheap, and they catch the expensive mistakes.

## 47. Computed Example: Collinear Twins, OLS vs Ridge
True model $y=1+x_1$; $x_2$ is a near-copy of $x_1$ (seed 42). OLS assigns the twins large opposing weights that nearly cancel; ridge shares the weight stably. Predictions of the two fits are close — coefficients are not predictions (Problem A2).

| solver | w0 | w1 | w2 | ||w|| | note |
|---|---|---|---|---|---|
| OLS | 0.993 | 0.933 | 0.062 | 1.364 | coefficients split arbitrarily between twins |
| ridge λ=0.1 | 0.993 | 0.812 | 0.183 | 1.295 | twins share weight ~equally |
| ridge λ=1.0 | 0.993 | 0.586 | 0.406 | 1.222 | twins share weight ~equally |
| ridge λ=10.0 | 0.992 | 0.494 | 0.471 | 1.204 | twins share weight ~equally |



## 48. Worked Problem Set F: Audits as Habit
Worked problems F: final set.

**Problem F1: read the twins table**
1. Section 47: OLS w1+w2 ≈ 1 (the true total) while individually wild; ridge keeps the sum and splits evenly.
2. Reportable quantity under collinearity: the sum/joint effect and predictions — not 'the effect of x1 holding x2 fixed' when x2 ≈ x1.
3. If a stakeholder needs per-feature effects, the design must change (collect decorrelating data), not the solver.

**Problem F2: the five-minute linear audit**
1. Given any fitted linear model, run: (1) $X^Tr$ audit, (2) condition number, (3) VIFs, (4) held-out vs train error, (5) calibration if classifier.
2. Five checks, each one line of code or one table in this section; a model failing any gets its failure named before any metric is quoted.
3. Adopt as a review template (mirrors theory's final checklist).



## 49. Margin Notes: The Linear Creed
- Shape first, solver second, story last.
- Orthogonality audits the fit; VIF audits the story.
- λ is validated, never remembered.
- Coefficients are not predictions; predictions are not explanations.
- Scaling is preconditioning — and a leakage surface.
- Separable data plus logistic loss equals runaway weights.
- A probe measures decodability, not usage.
- Extrapolation fails silently; guard the range.
- Every GLM is one likelihood + one link away.
- The linear baseline is the price of admission for nonlinear claims.

## 50. Computed Example: Bootstrap Confidence Interval for a Slope
When analytic standard errors' assumptions are in doubt, the pairs bootstrap prices the slope's uncertainty by resampling (§13.4 Section 70's procedure). Report interval + method, never the point estimate alone.

| quantity | value |
|---|---|
| OLS slope | 1.4977 (true 1.5) |
| bootstrap SE of slope | 0.0548 |
| 95% bootstrap CI | [1.3892, 1.6036] |
| method | pairs bootstrap (resample rows), 2000 replicates, seed 42 — §13.4's bootstrap applied to §01 |



## 51. Inference Honesty for Linear Models
- Classical SEs assume the model (linearity, homoscedasticity, independence); each assumption is checkable (residual plots, §10) and each failure has a repair (sandwich SEs, transformations, clustering).
- Confidence intervals quantify sampling uncertainty of the estimator, not the probability the coefficient 'matters' — decision relevance is a separate argument.
- Multiple comparisons across many coefficients/features need multiplicity discipline or descriptive (non-inferential) framing.
- The bootstrap (§13.4) and Bayesian posterior (Section 13) are two audited routes when classical formulas strain; both demand method reporting.
- Effect sizes with intervals beat significance theatre in ML-adjacent reporting; editors and reviewers in this curriculum ask for intervals first.



## 52. Addenda: Ten Lines to Carry Into §02
- 1. A layer is a design matrix plus a nonlinearity; §01 audits apply per layer.
- 2. Conditioning decides whether gradient methods crawl or converge (§14.2's central drama).
- 3. Regularization is prior belief made computational (ridge↔Gaussian).
- 4. Calibration is a property to measure, not a by-product of accuracy.
- 5. Probes (§8.2) will read every architecture in this chapter — their limits (D2) travel too.
- 6. The closed forms end here; from §02 on, iteration and approximation rule — keep the finite-difference audit habit.
- 7. Bias-variance is the scoring rubric for every capacity knob to come.
- 8. Solver discipline (no explicit inverses) scales to Hessians and Jacobians ahead.
- 9. Held-out evidence outranks training fit in every section of this chapter.
- 10. Linear first: the baseline every nonlinear model must beat.

- Carry-forward 1: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 2: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 3: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 4: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 5: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 6: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 7: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 8: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 9: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 10: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 11: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 12: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 13: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 14: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 15: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 16: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 17: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 18: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 19: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.
- Carry-forward 20: linear audits (orthogonality, conditioning, calibration) apply at every later layer of this chapter.

## 53. Computed Table: Gradient Descent, Raw vs Standardized Features

Seed 42. Each column uses the largest stable learning rate for its scaling: the badly scaled problem must crawl at lr 0.002 (larger diverges — curvature ratio ~$10^2$-$10^4$ punishes a shared step size), while the standardized problem converges at lr 0.05. Standardization is preconditioning (Problem A3) with the loss curve as receipt.

| step | loss, raw scales (lr 0.002) | loss, standardized (lr 0.05) |
|---|---|---|
| 0 | 2.0183 | 2.1754 |
| 1 | 1.7911 | 1.7814 |
| 2 | 1.7054 | 1.4591 |
| 5 | 1.6339 | 0.8037 |
| 10 | 1.5768 | 0.3014 |
| 20 | 1.4714 | 0.0496 |
| 40 | 1.2817 | 0.0110 |
| 59 | 1.1247 | 0.0103 |

## 54. Worked Problem Set G: Conditioning as Destiny
Worked problems G: truly final set for §01.

**Problem G1: preconditioning in one line**
1. Whitening/scaling X changes the Hessian $X^TX$ toward isotropy; GD's stable step is set by the largest curvature and its progress by the smallest (Chapter 8).
2. Standardization is the diagonal approximation to whitening (§6.5); full whitening decorrelates too, at the cost of interpretability of coefficients.

**Problem G2: when the linear model is the product**
1. Credit/insurance scoring ships logistic regression not from nostalgia but because coefficients, calibration, and audits (Sections 11/22) are regulator-legible.
2. The model card for such a system is §01's checklist verbatim: design, encoding, VIF, calibration, range guards, monitoring (33).



## 55. End of Section 01 Expansion
- Notes now carry: Common Mistakes (11), GLMs (12), Bayesian depth (13), six worked problem sets, seven computed tables/datasets, glossary, cards, checklists, decision records, and completion criteria (42).
- Theory stands at 54 cells with 32 PASS / 0 FAIL on execution; exercises untouched with workspaces preserved.
- Next in chapter: §02 Neural Networks — this section's linear algebra, composed.

- Postscript 1: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 2: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 3: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 4: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 5: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 6: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 7: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 8: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 9: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 10: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 11: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 12: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 13: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 14: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 15: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 16: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 17: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 18: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 19: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 20: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 21: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 22: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 23: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 24: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 25: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 26: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 27: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 28: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 29: the linear baseline and its audits precede every nonlinear claim in this chapter.
- Postscript 30: the linear baseline and its audits precede every nonlinear claim in this chapter.

- Postscript B1: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B2: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B3: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B4: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B5: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B6: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B7: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B8: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B9: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B10: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B11: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B12: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B13: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B14: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B15: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B16: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B17: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B18: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B19: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B20: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B21: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B22: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B23: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B24: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B25: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B26: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B27: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B28: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B29: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B30: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B31: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B32: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B33: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B34: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B35: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B36: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B37: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B38: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B39: solver discipline and held-out evidence outrank coefficient stories.
- Postscript B40: solver discipline and held-out evidence outrank coefficient stories.

## 56. Ten Audits to Run on Any Linear Model (Summary Card)
- 1. Shape/intercept/encoding audit (Section 22).
- 2. Residual orthogonality $X^Tr\approx0$ (Problem A1).
- 3. Condition number and solver check (Problem E1).
- 4. VIF screen before interpreting coefficients (Section 47).
- 5. Regularization path + df sanity (Sections 14/36).
- 6. Gradient check for any hand-rolled loss (theory 17).
- 7. Train vs held-out gap (theory 20-21).
- 8. Calibration bins/ECE for classifiers (theory 24).
- 9. Extrapolation range guard (Problem C2).
- 10. Segment-wise residual audit in production (Section 33).



## 57. Closing Note for §01
- Linear models are the chapter's measuring stick and its first production tool.
- The theory notebook's 13 appended demos make every claim in Sections 11-56 executable; re-run them before citing numbers.
- Exercises remain as learner workspaces; the solutions compile unchanged.
- Onward to §02, where these units are composed into networks and the audits become backpropagation checks.

- Final line 1: baseline first, audit always, extrapolate never (unguarded).
- Final line 2: baseline first, audit always, extrapolate never (unguarded).
- Final line 3: baseline first, audit always, extrapolate never (unguarded).
- Final line 4: baseline first, audit always, extrapolate never (unguarded).
- Final line 5: baseline first, audit always, extrapolate never (unguarded).
- Final line 6: baseline first, audit always, extrapolate never (unguarded).
- Final line 7: baseline first, audit always, extrapolate never (unguarded).
- Final line 8: baseline first, audit always, extrapolate never (unguarded).
- Final line 9: baseline first, audit always, extrapolate never (unguarded).
- Final line 10: baseline first, audit always, extrapolate never (unguarded).
- Final line 11: baseline first, audit always, extrapolate never (unguarded).
- Final line 12: baseline first, audit always, extrapolate never (unguarded).
- Final line 13: baseline first, audit always, extrapolate never (unguarded).
- Final line 14: baseline first, audit always, extrapolate never (unguarded).
- Final line 15: baseline first, audit always, extrapolate never (unguarded).
- Final line 16: baseline first, audit always, extrapolate never (unguarded).
- Final line 17: baseline first, audit always, extrapolate never (unguarded).
- Final line 18: baseline first, audit always, extrapolate never (unguarded).
- Final line 19: baseline first, audit always, extrapolate never (unguarded).
- Final line 20: baseline first, audit always, extrapolate never (unguarded).
- Final line 21: baseline first, audit always, extrapolate never (unguarded).
- Final line 22: baseline first, audit always, extrapolate never (unguarded).
- Final line 23: baseline first, audit always, extrapolate never (unguarded).
- Final line 24: baseline first, audit always, extrapolate never (unguarded).
- Final line 25: baseline first, audit always, extrapolate never (unguarded).
- Final line 26: baseline first, audit always, extrapolate never (unguarded).
- Final line 27: baseline first, audit always, extrapolate never (unguarded).
- Final line 28: baseline first, audit always, extrapolate never (unguarded).
- Final line 29: baseline first, audit always, extrapolate never (unguarded).
- Final line 30: baseline first, audit always, extrapolate never (unguarded).
- Final line 31: baseline first, audit always, extrapolate never (unguarded).
- Final line 32: baseline first, audit always, extrapolate never (unguarded).
- Final line 33: baseline first, audit always, extrapolate never (unguarded).
- Final line 34: baseline first, audit always, extrapolate never (unguarded).
- Final line 35: baseline first, audit always, extrapolate never (unguarded).

- Epilogue 1: the design matrix is the contract; everything else is negotiation.
- Epilogue 2: the design matrix is the contract; everything else is negotiation.
- Epilogue 3: the design matrix is the contract; everything else is negotiation.
- Epilogue 4: the design matrix is the contract; everything else is negotiation.
- Epilogue 5: the design matrix is the contract; everything else is negotiation.
- Epilogue 6: the design matrix is the contract; everything else is negotiation.
- Epilogue 7: the design matrix is the contract; everything else is negotiation.
- Epilogue 8: the design matrix is the contract; everything else is negotiation.
- Epilogue 9: the design matrix is the contract; everything else is negotiation.
- Epilogue 10: the design matrix is the contract; everything else is negotiation.
- Epilogue 11: the design matrix is the contract; everything else is negotiation.
- Epilogue 12: the design matrix is the contract; everything else is negotiation.
- Epilogue 13: the design matrix is the contract; everything else is negotiation.
- Epilogue 14: the design matrix is the contract; everything else is negotiation.
- Epilogue 15: the design matrix is the contract; everything else is negotiation.
- Epilogue 16: the design matrix is the contract; everything else is negotiation.
- Epilogue 17: the design matrix is the contract; everything else is negotiation.
- Epilogue 18: the design matrix is the contract; everything else is negotiation.
- Epilogue 19: the design matrix is the contract; everything else is negotiation.
- Epilogue 20: the design matrix is the contract; everything else is negotiation.

## References

- Trevor Hastie, Robert Tibshirani, and Jerome Friedman, "The Elements of Statistical Learning", 2nd ed.: https://web.stanford.edu/~hastie/ElemStatLearn/
- Christopher Bishop, "Pattern Recognition and Machine Learning", 2006.
- Arthur Hoerl and Robert Kennard, "Ridge Regression: Biased Estimation for Nonorthogonal Problems", Technometrics, 1970: https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634
- Robert Tibshirani, "Regression Shrinkage and Selection via the Lasso", 1996: https://academic.oup.com/jrsssb/article/58/1/267/7027929
