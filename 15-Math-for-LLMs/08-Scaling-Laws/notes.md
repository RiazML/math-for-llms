[<- Fine-Tuning Math](../07-Fine-Tuning-Math/notes.md) | [Home](../../README.md) | [Efficient Attention and Inference ->](../09-Efficient-Attention-and-Inference/notes.md)

---

# Scaling Laws

Scaling laws are empirical formulas for forecasting how language-model loss changes as parameters, data, and compute change. They are not a replacement for experiments; they are a way to make experiments smaller, cheaper, and more informative.

## Overview

The core form is a power law with a floor:

$$
L(X)=L_\infty + AX^{-\alpha}.
$$

Here $L$ is usually held-out cross-entropy, $X$ is a resource such as parameters, tokens, or compute, $L_\infty$ is the irreducible floor for the setup, and $\alpha$ controls how fast excess loss falls. In LLM planning, the most important resources are parameter count $N$, training tokens $D$, and training compute $C$. For dense decoder-only transformers, a useful first estimate is:

$$
C\approx 6ND.
$$

The practical question is: for a fixed budget, what combination of $N$ and $D$ gives the lowest expected loss, and how uncertain is that forecast?

## Prerequisites

- Cross-entropy, perplexity, and held-out loss
- Logarithms and log-log plots
- Basic curve fitting and residual analysis
- Training FLOPs and token accounting
- The previous sections on training at scale and fine-tuning

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Fits toy power laws, builds IsoFLOP curves, solves compute-optimal allocation by grid search, studies data quality, serving cost, and residual diagnostics. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for power-law fitting, compute budgets, Chinchilla-style sizing, undertraining checks, and forecast uncertainty. |

## Learning Objectives

After this section, you should be able to:

- Explain the difference between parameter, data, and compute scaling.
- Fit a power law when the irreducible loss floor is known or hypothesized.
- Estimate training compute with $C\approx 6ND$.
- Draw and interpret IsoFLOP tradeoffs.
- Solve a compute-optimal allocation problem for a two-term loss proxy.
- Explain the Kaplan and Chinchilla lessons without cargo-culting ratios.
- Reason about data quality, repeated data, and effective token counts.
- Include inference cost and latency in model-size decisions.
- Diagnose residuals and uncertainty in scaling-law forecasts.

## Table of Contents

1. [What Scaling Laws Measure](#1-what-scaling-laws-measure)
   - 1.1 [Loss as the target](#11-loss-as-the-target)
   - 1.2 [Resources](#12-resources)
   - 1.3 [Power-law form](#13-powerlaw-form)
   - 1.4 [Forecasting discipline](#14-forecasting-discipline)
   - 1.5 [Scope](#15-scope)
2. [Parameter Scaling](#2-parameter-scaling)
   - 2.1 [Model-size law](#21-modelsize-law)
   - 2.2 [Capacity limit](#22-capacity-limit)
   - 2.3 [Over-large model](#23-overlarge-model)
   - 2.4 [Architecture dependence](#24-architecture-dependence)
   - 2.5 [Fine-tuning implication](#25-finetuning-implication)
3. [Data Scaling](#3-data-scaling)
   - 3.1 [Token-count law](#31-tokencount-law)
   - 3.2 [Data quality](#32-data-quality)
   - 3.3 [Repeated data](#33-repeated-data)
   - 3.4 [Domain match](#34-domain-match)
   - 3.5 [Tokenizer dependence](#35-tokenizer-dependence)
4. [Compute Scaling](#4-compute-scaling)
   - 4.1 [Dense training FLOPs](#41-dense-training-flops)
   - 4.2 [Compute law](#42-compute-law)
   - 4.3 [IsoFLOP curves](#43-isoflop-curves)
   - 4.4 [Compute is not wall-clock](#44-compute-is-not-wallclock)
   - 4.5 [Budget translation](#45-budget-translation)
5. [Compute-Optimal Allocation](#5-computeoptimal-allocation)
   - 5.1 [Two-term loss proxy](#51-twoterm-loss-proxy)
   - 5.2 [Constraint](#52-constraint)
   - 5.3 [Balanced marginal returns](#53-balanced-marginal-returns)
   - 5.4 [Chinchilla intuition](#54-chinchilla-intuition)
   - 5.5 [Practical rule](#55-practical-rule)
6. [Kaplan and Chinchilla Lessons](#6-kaplan-and-chinchilla-lessons)
   - 6.1 [Kaplan-style result](#61-kaplanstyle-result)
   - 6.2 [Chinchilla revision](#62-chinchilla-revision)
   - 6.3 [Undertraining diagnostic](#63-undertraining-diagnostic)
   - 6.4 [Overtraining for inference](#64-overtraining-for-inference)
   - 6.5 [Do not cargo-cult ratios](#65-do-not-cargocult-ratios)
7. [Fitting Scaling Laws](#7-fitting-scaling-laws)
   - 7.1 [Log transformation](#71-log-transformation)
   - 7.2 [Unknown floor](#72-unknown-floor)
   - 7.3 [Holdout validation](#73-holdout-validation)
   - 7.4 [Residual analysis](#74-residual-analysis)
   - 7.5 [Extrapolation risk](#75-extrapolation-risk)
8. [Data Quality and Mixtures](#8-data-quality-and-mixtures)
   - 8.1 [Mixture weights](#81-mixture-weights)
   - 8.2 [Effective tokens](#82-effective-tokens)
   - 8.3 [Contamination](#83-contamination)
   - 8.4 [Curriculum and filtering](#84-curriculum-and-filtering)
   - 8.5 [Evaluation distribution](#85-evaluation-distribution)
9. [Inference-Aware Scaling](#9-inferenceaware-scaling)
   - 9.1 [Serving cost](#91-serving-cost)
   - 9.2 [Latency constraint](#92-latency-constraint)
   - 9.3 [Small overtrained models](#93-small-overtrained-models)
   - 9.4 [Distillation](#94-distillation)
   - 9.5 [Total cost objective](#95-total-cost-objective)
10. [Limits and Debugging](#10-limits-and-debugging)
   - 10.1 [Not a theorem of all AI](#101-not-a-theorem-of-all-ai)
   - 10.2 [Metric artifacts](#102-metric-artifacts)
   - 10.3 [Architecture changes](#103-architecture-changes)
   - 10.4 [Bad small runs](#104-bad-small-runs)
   - 10.5 [Decision checklist](#105-decision-checklist)
11. [Parametric Scaling Law](#11-the-parametric-scaling-law-in-full)
12. [IsoFLOP Thinking](#12-isoflop-thinking-in-practice)
13. [Data-Constrained Scaling](#13-data-constrained-and-repetition-scaling)
14. [Inference-Aware Scaling](#14-inference-aware-scaling-total-cost-of-ownership)
15. [Scaling Beyond Loss](#15-scaling-beyond-loss-emergence-and-metric-choice)
16. [Broken Scaling](#16-broken-scaling-and-when-power-laws-fail)
17. [Computed Tables](#17-computed-tables-for-scaling-laws)
18. [Fitting a Power Law by Hand](#18-fitting-a-power-law-by-hand-worked)
19. [Glossary](#19-glossary)
20. [Ten Worked Problems](#20-ten-worked-problems-with-solutions)
21. [Small Scaling Study](#21-a-complete-small-scaling-study-designed)
22. [Historical Line](#22-historical-line)
23. [Formula Summary](#23-formula-summary)
24. [Final Self-Check](#24-final-self-check)
25. [Computed Scaling Tables](#25-computed-scaling-tables)
26. [Decision Worksheets](#26-decision-worksheets)
27. [Five Scenarios](#27-five-scenarios)

---

## Forecasting Workflow

```text
small runs -> held-out loss table -> fit simple law -> holdout forecast -> residual check -> budget decision -> larger validation run
```

At each step, write down the exact setup: architecture, tokenizer, data mixture, optimizer, schedule, batch size, precision, evaluation set, and filtering. A scaling law fitted under one setup is not guaranteed to transfer to another setup.

## 1. What Scaling Laws Measure

This part studies what scaling laws measure as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Loss as the target](#1-loss-as-the-target) | scaling laws usually model held-out cross-entropy | $L=-E[\log p_\theta(t_i\mid t_{<i})]$ |
| [Resources](#1-resources) | parameters, tokens, and compute are the main axes | $N,\ D,\ C$ |
| [Power-law form](#1-powerlaw-form) | loss often improves approximately linearly on log-log axes after subtracting an irreducible floor | $L(X)=L_\infty + AX^{-\alpha}$ |
| [Forecasting discipline](#1-forecasting-discipline) | fit small runs, predict larger runs, then validate residuals | $\hat L(C_\mathrm{large})$ |
| [Scope](#1-scope) | a scaling law is empirical over a data, architecture, optimizer, and tokenization regime | $\mathrm{law}=\mathrm{fit}\mid\mathrm{setup}$ |

### 1.1 Loss as the target

**Main idea.** Scaling laws usually model held-out cross-entropy.

Core relation:

$$L=-E[\log p_\theta(t_i\mid t_{<i})]$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 1.2 Resources

**Main idea.** Parameters, tokens, and compute are the main axes.

Core relation:

$$N,\ D,\ C$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 1.3 Power-law form

**Main idea.** Loss often improves approximately linearly on log-log axes after subtracting an irreducible floor.

Core relation:

$$L(X)=L_\infty + AX^{-\alpha}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 1.4 Forecasting discipline

**Main idea.** Fit small runs, predict larger runs, then validate residuals.

Core relation:

$$\hat L(C_\mathrm{large})$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 1.5 Scope

**Main idea.** A scaling law is empirical over a data, architecture, optimizer, and tokenization regime.

Core relation:

$$\mathrm{law}=\mathrm{fit}\mid\mathrm{setup}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 2. Parameter Scaling

This part studies parameter scaling as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Model-size law](#2-modelsize-law) | larger models reduce loss when data and compute are not limiting | $L(N)=L_\infty + A_NN^{-\alpha_N}$ |
| [Capacity limit](#2-capacity-limit) | too small a model cannot represent the distribution well | $N\downarrow\Rightarrow L\uparrow$ |
| [Over-large model](#2-overlarge-model) | a model can be too large for the available token budget | $D/N$ too small |
| [Architecture dependence](#2-architecture-dependence) | the coefficients depend on architecture and training recipe | $A_N,\alpha_N$ are not universal constants |
| [Fine-tuning implication](#2-finetuning-implication) | adapter rank can be viewed as an adaptation-capacity axis | $r$ controls update capacity |

### 2.1 Model-size law

**Main idea.** Larger models reduce loss when data and compute are not limiting.

Core relation:

$$L(N)=L_\infty + A_NN^{-\alpha_N}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 2.2 Capacity limit

**Main idea.** Too small a model cannot represent the distribution well.

Core relation:

$$N\downarrow\Rightarrow L\uparrow$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 2.3 Over-large model

**Main idea.** A model can be too large for the available token budget.

Core relation:

$$D/N$ too small$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 2.4 Architecture dependence

**Main idea.** The coefficients depend on architecture and training recipe.

Core relation:

$$A_N,\alpha_N$ are not universal constants$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 2.5 Fine-tuning implication

**Main idea.** Adapter rank can be viewed as an adaptation-capacity axis.

Core relation:

$$r$ controls update capacity$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 3. Data Scaling

This part studies data scaling as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Token-count law](#3-tokencount-law) | more unique useful tokens reduce loss | $L(D)=L_\infty + A_DD^{-\alpha_D}$ |
| [Data quality](#3-data-quality) | one high-quality token can be worth more than one noisy token | $D_\mathrm{eff}\ne D_\mathrm{raw}$ |
| [Repeated data](#3-repeated-data) | repetition gives diminishing returns and can increase memorization | $D_\mathrm{eff}<kD$ for repeats |
| [Domain match](#3-domain-match) | the validation distribution defines what improvement means | $p_\mathrm{train}\approx p_\mathrm{eval}$ |
| [Tokenizer dependence](#3-tokenizer-dependence) | token counts depend on the tokenizer | $D$ is measured in model tokens |

### 3.1 Token-count law

**Main idea.** More unique useful tokens reduce loss.

Core relation:

$$L(D)=L_\infty + A_DD^{-\alpha_D}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 3.2 Data quality

**Main idea.** One high-quality token can be worth more than one noisy token.

Core relation:

$$D_\mathrm{eff}\ne D_\mathrm{raw}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 3.3 Repeated data

**Main idea.** Repetition gives diminishing returns and can increase memorization.

Core relation:

$$D_\mathrm{eff}<kD$ for repeats$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 3.4 Domain match

**Main idea.** The validation distribution defines what improvement means.

Core relation:

$$p_\mathrm{train}\approx p_\mathrm{eval}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 3.5 Tokenizer dependence

**Main idea.** Token counts depend on the tokenizer.

Core relation:

$$D$ is measured in model tokens$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 4. Compute Scaling

This part studies compute scaling as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Dense training FLOPs](#4-dense-training-flops) | a common first estimate for dense transformers is six times parameters times tokens | $C\approx 6ND$ |
| [Compute law](#4-compute-law) | loss can be modeled directly as a function of compute | $L(C)=L_\infty + A_CC^{-\alpha_C}$ |
| [IsoFLOP curves](#4-isoflop-curves) | fixed compute creates a tradeoff between model size and token count | $D=C/(6N)$ |
| [Compute is not wall-clock](#4-compute-is-not-wallclock) | hardware utilization and communication decide time | $T=C/\mathrm{achieved\ FLOPs\ per\ sec}$ |
| [Budget translation](#4-budget-translation) | money and time become constraints after mapping to achievable FLOPs | $\mathrm{cost}=T\cdot\mathrm{price}$ |

### 4.1 Dense training FLOPs

**Main idea.** A common first estimate for dense transformers is six times parameters times tokens.

Core relation:

$$C\approx 6ND$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 4.2 Compute law

**Main idea.** Loss can be modeled directly as a function of compute.

Core relation:

$$L(C)=L_\infty + A_CC^{-\alpha_C}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 4.3 IsoFLOP curves

**Main idea.** Fixed compute creates a tradeoff between model size and token count.

Core relation:

$$D=C/(6N)$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** These curves are how you ask whether the next dollar should buy a larger model or more training tokens.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 4.4 Compute is not wall-clock

**Main idea.** Hardware utilization and communication decide time.

Core relation:

$$T=C/\mathrm{achieved\ FLOPs\ per\ sec}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 4.5 Budget translation

**Main idea.** Money and time become constraints after mapping to achievable flops.

Core relation:

$$\mathrm{cost}=T\cdot\mathrm{price}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 5. Compute-Optimal Allocation

This part studies compute-optimal allocation as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Two-term loss proxy](#5-twoterm-loss-proxy) | separate parameter-limited and data-limited terms | $L(N,D)=L_\infty + A/N^\alpha + B/D^\beta$ |
| [Constraint](#5-constraint) | training compute couples N and D | $C=6ND$ |
| [Balanced marginal returns](#5-balanced-marginal-returns) | at optimum, extra compute should help similarly through N and D | $\partial L/\partial N$ balances $\partial L/\partial D$ under the constraint |
| [Chinchilla intuition](#5-chinchilla-intuition) | many earlier large models were undertrained relative to their size | $D/N$ was too small |
| [Practical rule](#5-practical-rule) | choose N and D together, not one after the other | $(N^\star,D^\star)=\arg\min_{6ND=C}L(N,D)$ |

### 5.1 Two-term loss proxy

**Main idea.** Separate parameter-limited and data-limited terms.

Core relation:

$$L(N,D)=L_\infty + A/N^\alpha + B/D^\beta$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 5.2 Constraint

**Main idea.** Training compute couples n and d.

Core relation:

$$C=6ND$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 5.3 Balanced marginal returns

**Main idea.** At optimum, extra compute should help similarly through n and d.

Core relation:

$$\partial L/\partial N$ balances $\partial L/\partial D$ under the constraint$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 5.4 Chinchilla intuition

**Main idea.** Many earlier large models were undertrained relative to their size.

Core relation:

$$D/N$ was too small$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 5.5 Practical rule

**Main idea.** Choose n and d together, not one after the other.

Core relation:

$$(N^\star,D^\star)=\arg\min_{6ND=C}L(N,D)$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 6. Kaplan and Chinchilla Lessons

This part studies kaplan and chinchilla lessons as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Kaplan-style result](#6-kaplanstyle-result) | smooth power laws made loss forecasting practical | $L$ follows power laws across scale ranges |
| [Chinchilla revision](#6-chinchilla-revision) | more tokens per parameter can be compute-optimal than earlier practice suggested | $D^\star/N^\star$ increases |
| [Undertraining diagnostic](#6-undertraining-diagnostic) | a model with low tokens per parameter may benefit more from data than size | $D/N\ll\mathrm{target}$ |
| [Overtraining for inference](#6-overtraining-for-inference) | training smaller models longer can reduce serving cost | $\mathrm{train\ cost}+\mathrm{inference\ cost}$ |
| [Do not cargo-cult ratios](#6-do-not-cargocult-ratios) | ratios depend on data, architecture, and objective | $D/N$ is a planning prior, not a theorem |

### 6.1 Kaplan-style result

**Main idea.** Smooth power laws made loss forecasting practical.

Core relation:

$$L$ follows power laws across scale ranges$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 6.2 Chinchilla revision

**Main idea.** More tokens per parameter can be compute-optimal than earlier practice suggested.

Core relation:

$$D^\star/N^\star$ increases$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** The practical lesson is that undertrained large models can waste compute even when they look impressive.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 6.3 Undertraining diagnostic

**Main idea.** A model with low tokens per parameter may benefit more from data than size.

Core relation:

$$D/N\ll\mathrm{target}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 6.4 Overtraining for inference

**Main idea.** Training smaller models longer can reduce serving cost.

Core relation:

$$\mathrm{train\ cost}+\mathrm{inference\ cost}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 6.5 Do not cargo-cult ratios

**Main idea.** Ratios depend on data, architecture, and objective.

Core relation:

$$D/N$ is a planning prior, not a theorem$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 7. Fitting Scaling Laws

This part studies fitting scaling laws as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Log transformation](#7-log-transformation) | if the floor is known, power-law fitting becomes linear | $\log(L-L_\infty)=\log A-\alpha\log X$ |
| [Unknown floor](#7-unknown-floor) | the irreducible loss floor must be fit or bounded carefully | $L_\infty$ |
| [Holdout validation](#7-holdout-validation) | reserve some runs to test forecasts | $L_\mathrm{actual}-L_\mathrm{pred}$ |
| [Residual analysis](#7-residual-analysis) | curved residuals mean the law is missing a factor | $\epsilon_i=L_i-\hat L_i$ |
| [Extrapolation risk](#7-extrapolation-risk) | small-run fits can fail when regime changes | $X_\mathrm{test}\gg X_\mathrm{fit}$ |

### 7.1 Log transformation

**Main idea.** If the floor is known, power-law fitting becomes linear.

Core relation:

$$\log(L-L_\infty)=\log A-\alpha\log X$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 7.2 Unknown floor

**Main idea.** The irreducible loss floor must be fit or bounded carefully.

Core relation:

$$L_\infty$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** A small error in the loss floor can bend exponent estimates and make forecasts overconfident.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 7.3 Holdout validation

**Main idea.** Reserve some runs to test forecasts.

Core relation:

$$L_\mathrm{actual}-L_\mathrm{pred}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 7.4 Residual analysis

**Main idea.** Curved residuals mean the law is missing a factor.

Core relation:

$$\epsilon_i=L_i-\hat L_i$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 7.5 Extrapolation risk

**Main idea.** Small-run fits can fail when regime changes.

Core relation:

$$X_\mathrm{test}\gg X_\mathrm{fit}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 8. Data Quality and Mixtures

This part studies data quality and mixtures as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Mixture weights](#8-mixture-weights) | training data is a weighted mixture of sources | $p_\mathrm{train}=\sum_k w_kp_k$ |
| [Effective tokens](#8-effective-tokens) | quality, deduplication, and domain match alter token value | $D_\mathrm{eff}=\sum_k q_kw_kD$ |
| [Contamination](#8-contamination) | validation leakage makes scaling look better than it is | $D_\mathrm{eval}\cap D_\mathrm{train}\ne\emptyset$ |
| [Curriculum and filtering](#8-curriculum-and-filtering) | data order and filtering can shift constants | $A$ changes even if $\alpha$ is similar |
| [Evaluation distribution](#8-evaluation-distribution) | optimize for the distribution that matters | $L_\mathrm{eval}$, not only $L_\mathrm{web}$ |

### 8.1 Mixture weights

**Main idea.** Training data is a weighted mixture of sources.

Core relation:

$$p_\mathrm{train}=\sum_k w_kp_k$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 8.2 Effective tokens

**Main idea.** Quality, deduplication, and domain match alter token value.

Core relation:

$$D_\mathrm{eff}=\sum_k q_kw_kD$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** Data quantity without data value is a weak planning variable.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 8.3 Contamination

**Main idea.** Validation leakage makes scaling look better than it is.

Core relation:

$$D_\mathrm{eval}\cap D_\mathrm{train}\ne\emptyset$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 8.4 Curriculum and filtering

**Main idea.** Data order and filtering can shift constants.

Core relation:

$$A$ changes even if $\alpha$ is similar$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 8.5 Evaluation distribution

**Main idea.** Optimize for the distribution that matters.

Core relation:

$$L_\mathrm{eval}$, not only $L_\mathrm{web}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 9. Inference-Aware Scaling

This part studies inference-aware scaling as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Serving cost](#9-serving-cost) | a model trained once may be served many times | $C_\mathrm{serve}\propto N\cdot T_\mathrm{generated}\cdot Q$ |
| [Latency constraint](#9-latency-constraint) | larger models can be better but too slow | $T_\mathrm{latency}\le T_\mathrm{SLA}$ |
| [Small overtrained models](#9-small-overtrained-models) | more training tokens can improve a smaller cheaper model | $D/N$ can exceed pretraining-optimal planning ratios |
| [Distillation](#9-distillation) | teacher quality can transfer into a smaller student | $p_s \approx p_t$ |
| [Total cost objective](#9-total-cost-objective) | choose models by train plus serve plus quality | $J=\mathrm{loss}+\lambda\mathrm{cost}$ |

### 9.1 Serving cost

**Main idea.** A model trained once may be served many times.

Core relation:

$$C_\mathrm{serve}\propto N\cdot T_\mathrm{generated}\cdot Q$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** The best training run is not always the cheapest product model.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 9.2 Latency constraint

**Main idea.** Larger models can be better but too slow.

Core relation:

$$T_\mathrm{latency}\le T_\mathrm{SLA}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 9.3 Small overtrained models

**Main idea.** More training tokens can improve a smaller cheaper model.

Core relation:

$$D/N$ can exceed pretraining-optimal planning ratios$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 9.4 Distillation

**Main idea.** Teacher quality can transfer into a smaller student.

Core relation:

$$p_s \approx p_t$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 9.5 Total cost objective

**Main idea.** Choose models by train plus serve plus quality.

Core relation:

$$J=\mathrm{loss}+\lambda\mathrm{cost}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
## 10. Limits and Debugging

This part studies limits and debugging as a forecasting tool. The formulas are useful only when you keep the measurement setup fixed and validate predictions against real runs.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Not a theorem of all AI](#10-not-a-theorem-of-all-ai) | scaling laws are fitted empirical regularities | $\mathrm{valid\ only\ in\ regime}$ |
| [Metric artifacts](#10-metric-artifacts) | thresholded task scores can look sudden even when loss changes smoothly | $\mathbf{1}\{s>\tau\}$ |
| [Architecture changes](#10-architecture-changes) | new attention, MoE, or tokenizer choices can change constants | $A,\alpha,L_\infty$ shift |
| [Bad small runs](#10-bad-small-runs) | bugs in small runs poison the forecast | $\hat L$ inherits measurement error |
| [Decision checklist](#10-decision-checklist) | use scaling laws as planning instruments with uncertainty | $\hat L\pm\mathrm{interval}$ |

### 10.1 Not a theorem of all AI

**Main idea.** Scaling laws are fitted empirical regularities.

Core relation:

$$\mathrm{valid\ only\ in\ regime}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 10.2 Metric artifacts

**Main idea.** Thresholded task scores can look sudden even when loss changes smoothly.

Core relation:

$$\mathbf{1}\{s>\tau\}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 10.3 Architecture changes

**Main idea.** New attention, moe, or tokenizer choices can change constants.

Core relation:

$$A,\alpha,L_\infty$ shift$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 10.4 Bad small runs

**Main idea.** Bugs in small runs poison the forecast.

Core relation:

$$\hat L$ inherits measurement error$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.
### 10.5 Decision checklist

**Main idea.** Use scaling laws as planning instruments with uncertainty.

Core relation:

$$\hat L\pm\mathrm{interval}$$

Scaling laws are not magic. They are compact empirical models of how loss changes when resources change. Their value comes from discipline: keep the setup stable, measure held-out loss, fit simple forms, reserve validation runs, and attach uncertainty to every forecast.

**Worked micro-example.** Suppose loss follows $L(C)=1.5+0.8C^{-0.05}$. Increasing compute by a factor of 10 multiplies the excess loss by $10^{-0.05}\approx 0.891$. That is real improvement, but it is slow improvement. Power laws explain both why scale helps and why scale is expensive.

**Implementation check.** Plot on log-log axes, inspect residuals, and verify that the fitted law predicts runs that were not used for fitting. If the residuals curve systematically, the simple power law is missing a factor.

**AI connection.** This is a practical quantity for planning or checking a training run.

**Common mistake.** Do not treat a published coefficient as a universal constant. Exponents and offsets depend on model family, data, optimizer, tokenizer, and evaluation distribution.

---

---

## 11. The Parametric Scaling Law in Full

### 11.1 The Hoffmann et al. functional form

Hoffmann et al. (2022, Chinchilla) fit $L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}$, where $E$ is the irreducible entropy of the data distribution, the second term is the parameter-limited error, and the third is the data-limited error.
- Example: if $A/N^\alpha$ dominates, doubling tokens barely helps; the model is capacity-limited.
- Example: if $B/D^\beta$ dominates, a bigger model trained on the same tokens wastes parameters.
- Non-example: $E$ is not "zero loss at infinite scale" — natural text has genuine entropy (Chapter 15 Section 05), so $E>0$ and loss curves flatten toward it.

### 11.2 Deriving the compute-optimal ratio

With $C\approx6ND$ (Training at Scale 36.1), minimizing $L(N,D)$ subject to $ND=C/6$ gives optimal growth where the two error terms' marginal returns balance (Section 5.3). Hoffmann et al. (2022) found $N$ and $D$ should scale roughly *equally* with compute — summarized as about 20 tokens per parameter near their optimum — revising Kaplan et al. (2020), whose fits implied growing parameters much faster than data. The revision came largely from training smaller models longer with appropriately decayed learning-rate schedules, a *methodological* correction, not new physics.

### 11.3 Three fitting approaches, three answers to check against each other

Hoffmann et al. (2022) used three methods: (1) fixed model sizes, varying tokens, reading minima; (2) IsoFLOP profiles (Section 4.3) — fixed compute, varying the $N/D$ split, fitting the loss minimum per budget; (3) the parametric fit of 11.1. Their agreement is the reason the Chinchilla ratio is trusted more than any single fit. A scaling claim supported by only one of the three should be treated as preliminary in your own work (Section 7).

---

## 12. IsoFLOP Thinking in Practice

Fix compute $C$. Each candidate $(N,D)$ with $6ND=C$ is one point; the loss as a function of $N$ at fixed $C$ is U-shaped: too-small models waste data, too-large models are undertrained. The minimum of that U is the compute-optimal model for $C$, and tracing minima across $C$ values is approach (2) of 11.3.

Worked skeleton at $C=6\times10^{21}$ FLOPs (so $ND=10^{21}$):
- $N=1$B, $D=1000$B tokens: data-rich, capacity-limited.
- $N=10$B, $D=100$B: $D/N=10$.
- $N=22$B, $D\approx45$B: $D/N\approx2$ — parameter-heavy relative to the Chinchilla-side rule, expect undertraining symptoms (6.3).

The point of the skeleton is not the winner; it is that an IsoFLOP comparison requires *training every point properly* (own LR schedule per run, Section 7), which is why scaling studies are expensive even before the large run they justify.

---

## 13. Data-Constrained and Repetition Scaling

Muennighoff et al. (2023) fit scaling when unique data $U$ is the binding constraint: repeated epochs contribute diminishing effective data, with up to about 4 epochs nearly as valuable as fresh data and value decaying thereafter (Training at Scale 22.1). Their modified law replaces $D$ by an *effective* tokens quantity that discounts repetitions.
Consequence for planning: in a data-constrained regime the compute-optimal move can be *more epochs on a smaller model*, the opposite of the parameter-hungry reading of Kaplan et al. (2020). State which regime you are in — data-rich or data-constrained — before quoting any ratio (11.2).

---

## 14. Inference-Aware Scaling: Total Cost of Ownership

Training cost is paid once; inference cost is paid per token forever. Narayanan-style accounting (Section 9): total cost $\approx C_{\text{train}}+(\text{tokens served})\times(\text{inference cost per token})$, and inference cost per token grows with $N$. A smaller model trained far past the Chinchilla point (overtraining, Section 6.4) can therefore be *total-cost optimal* for a high-volume deployment — the explicit reasoning behind small, long-trained open models intended for heavy serving. LLaMA (Touvron et al., 2023) stated this inference-budget motivation directly: train smaller models on more tokens than Chinchilla-optimal when inference cost dominates lifetime cost.

---

## 15. Scaling Beyond Loss: Emergence and Metric Choice

Wei et al. (2022) described abilities that appear abruptly with scale on some benchmarks; Schaeffer et al. (2023) argued many such discontinuities are artefacts of discontinuous metrics (exact-match accuracy) and become smooth under continuous scoring (per-token likelihood of the correct answer). The scaling-law lesson is Section 10.2 with teeth: **fit laws to continuous losses, not to thresholded benchmark accuracies**, and distrust a "phase transition" until the underlying continuous quantity has been plotted.

---

## 16. Broken Scaling and When Power Laws Fail

Caballero et al. (2023, Broken Neural Scaling Laws) fit scaling with smoothly broken power laws, capturing regimes where the exponent itself changes (double descent-like transitions, data-quality regime changes). Failure diagnostics for a naive power-law fit:
- Systematic (curved) residuals in log-log space (Section 7.4), not random scatter.
- A fitted floor $E$ below the known entropy of a trivial baseline — impossible, so the fit is extrapolating wrongly.
- Exponents that move when the smallest runs are dropped: the fit is driven by points outside the asymptotic regime.

---

---

## 28. Residual Diagnostics in Numbers

Illustrative losses versus the 25.1 ray fit — residuals stated in nats so their size against typical run-to-run noise (~0.01-0.02 nats) can be judged.

| $N$ | Predicted $L$ | 'Measured' $L$ (illustrative) | Residual | Reading |
|---:|---:|---:|---:|---|

| 1e+09 | 2.580 | 2.060 | -0.520 | within noise if $\lesssim$0.02 |
| 7e+09 | 2.184 | 1.940 | -0.244 | within noise if $\lesssim$0.02 |
| 7e+10 | 1.937 | 1.820 | -0.117 | within noise if $\lesssim$0.02 |

A residual table is a required exhibit of any scaling claim (21, step 7): a fit without its residuals asks to be trusted; a fit with them can be checked.

## 29. Ten Review Questions With Answers

### Q1. Why did Chinchilla revise Kaplan et al. rather than refute scaling?
Same power-law framework (11); the allocation optimum moved because smaller models were trained longer with per-run schedules (11.2-11.3). Methodology changed the fitted optimum.

### Q2. What does the $E$ term prevent?
Predictions of arbitrarily low loss. $E$ is the data's entropy floor (11.1); fits without it fail Problem 8's sanity test.

### Q3. At fixed compute, why is loss U-shaped in $N$?
Small $N$: capacity term dominates. Large $N$: data term dominates because $D=C/(6N)$ shrinks. The minimum balances them (12, 25.2).

### Q4. When is 100 tokens per parameter rational?
When inference cost dominates lifetime cost (14) or unique data is abundant and serving is heavy — not as a training-efficiency optimum.

### Q5. What is effective data under repetition?
Seen tokens discounted by epoch: early epochs count nearly fully, later ones progressively less (13, Muennighoff et al. 2023).

### Q6. How do you test an emergence claim in one step?
Replace the thresholded metric by its continuous counterpart and replot against scale (15). Smoothness indicts the metric.

### Q7. Name two signatures of a power-law fit leaving its regime.
Curved log-log residuals and exponents that move when small runs are dropped (16).

### Q8. Why must IsoFLOP points have individual schedules?
A shared schedule handicaps whichever size it wasn't tuned for, biasing the U's minimum (12) — the Kaplan/Chinchilla lesson.

### Q9. What belongs on the x-axis of a scaling claim and why not steps?
Compute, parameters, or tokens — quantities independent of batching choices. Steps encode batch size and cannot be compared across setups (Common Mistakes).

### Q10. What is the production run's role in a scaling study?
Its pre-registered predicted loss makes it the study's final holdout (21, step 8); a prediction made after seeing it is not a prediction.

## 30. Fitting Protocol, Step by Step

1. Assemble runs with identical tokenizer, data mixture, architecture family, and evaluator (21, step 1).
2. Convert every loss to the same units — bits per byte if tokenizers differ anywhere in the set (Problem 9).
3. Tabulate $(N,D,C,L)$ per run with $C=6ND$ recomputed, not copied from dashboards.
4. Plot $L$ versus $C$ in log-log; look before fitting. Curvature seen here is 16's warning in person.
5. Fit the one-variable compute law for a first exponent (18).
6. Fit the three-term law (11.1) and compare floors and residuals against step 5.
7. Run the IsoFLOP reading (12) at each budget and check the minima's $D/N$ drift against 11.2.
8. Hold out the largest run, refit, predict it, report the error (7.3).
9. Bootstrap over runs (resample the run table) for an exponent interval, not a point (cf. Ch15-05 Section 28).
10. Write the production prediction and its interval down, timestamped, before the production run starts.

## 31. Symbols

| $N$ | parameters |
| $D$ | training tokens |
| $U$ | unique tokens |
| $C$ | compute, FLOPs |
| $L(N,D)$ | test loss |
| $E$ | entropy floor |
| $A,\alpha$ | parameter-term coefficient and exponent |
| $B,\beta$ | data-term coefficient and exponent |
| $D/N$ | tokens per parameter |

---

## 32. What the Laws Do Not Scale: Architecture, Quality, and Optimization

### 32.1 Architecture changes move the curve, not just along it

The coefficients $A,B$ and even the exponents of 11.1 are properties of an architecture family, tokenizer, and data mixture (Section 8). Kaplan et al. (2020) found loss depends weakly on shape (depth/width aspect) at fixed parameter count over the ranges they tested — a license to choose shapes for *systems* reasons (Training at Scale's parallelism) — but a new attention variant or tokenizer is a new fit, not a point on the old one. Prefixing a scaling plot with "same setup as Section 21, step 1" is what makes its points commensurable.

### 32.2 Data quality enters through $E$ and $B$

Filtering and dedup (Training at Scale 22.3) change the data distribution: they can lower the achievable floor for the *target* distribution while raising measured loss on unfiltered held-out text. A scaling study whose filtering changes between the small and large runs is fitting a moving target. Pin the data version (Chapter 19's lineage) across the study, and if quality work intervenes, refit — the old coefficients describe a dataset that no longer exists.

### 32.3 Optimization is a hidden axis

Every point assumes a competently tuned optimizer at that scale. Under-tuned small runs (too-low LR, schedules not decayed to near-zero... in the cosine sense of 11.2's methodological lesson) sit *above* the true curve and bend the fitted exponent. This is why Hoffmann et al. (2022)'s re-examination found different optima: their small models were trained as if they mattered. Your study's LR protocol per scale (30, step 4's per-run schedules) is part of the measured object.

## 33. Batch Size and Learning Rate Inside Scaling Studies

Critical batch size grows through training and with scale (Training at Scale 11, McCandlish et al. 2018). A scaling study that fixes tokens-per-step across all model sizes confounds the resource axis: the largest run may be batch-inefficient while the smallest is batch-starved.
Protocol consequences:
- Report tokens per step per run alongside $(N,D)$ (30, step 3's table should have the column).
- Scale LR with batch by the regime-appropriate rule (Training at Scale 12), and state the rule.
- Treat a scaling point whose batch exceeds its measured noise scale as suspect: its loss is honest, but its *cost accounting* overstates the compute a better-batched run would need.

## 34. Reading the Primary Sources Without Being Misled

Kaplan et al. (2020): read for the *existence and form* of the laws and the compute law's exponent; read the allocation conclusions as conditional on their training protocol (11.2).
Hoffmann et al. (2022): read for method — three fitting approaches agreeing (11.3) is the paper's real contribution, with the 20-tokens-per-parameter figure as its most-quoted and most-context-dependent output.
Muennighoff et al. (2023): read when data, not compute, is your constraint; note how "epochs" become a scaling variable with diminishing returns (13).
Schaeffer et al. (2023): read before announcing any threshold phenomenon (15); their demand — show the continuous curve — is cheap to meet and expensive to skip.
Caballero et al. (2023): read when your residuals curve (16); a broken law names *where* behaviour changes, which is often the scientifically interesting part.

## 35. The Section in One Decision Sequence

1. State the regime: compute-rich, data-constrained, or inference-dominated (13, 14).
2. Assemble commensurable runs: one tokenizer, one data version, per-run schedules (21, 32).
3. Fit more than one way and require agreement (11.3).
4. Hold out the largest run; publish the prediction error with the fit (30).
5. Read residuals before extrapolating; curvature is a finding, not a nuisance (16).
6. Allocate the next increment to the binding term (26.1).
7. Pre-register the production prediction (21, step 8).
8. Report continuous losses; let benchmark drama follow, never lead (15).

---

## 36. IsoFLOP Slices at Three Budgets (Computed)

From 17.1's coefficients; use as the shape your own study (21) should reproduce with its own fit.


### Budget $C=6e+19$ FLOPs

| $N$ | $D$ | $D/N$ | $L(N,D)$ |
|---:|---:|---:|---:|
| 2e+08 | 5e+10 | 250 | 2.717 |
| 5e+08 | 2e+10 | 40 | 2.674 |
| 1e+09 | 1e+10 | 10 | 2.695 |
| 2e+09 | 5e+09 | 2 | 2.760 |
| 4e+09 | 2e+09 | 1 | 2.871 |
| 7e+09 | 1e+09 | 0 | 2.995 |
| 2e+10 | 7e+08 | 0 | 3.220 |
| 3e+10 | 3e+08 | 0 | 3.488 |

### Budget $C=6e+20$ FLOPs

| $N$ | $D$ | $D/N$ | $L(N,D)$ |
|---:|---:|---:|---:|
| 2e+08 | 5e+11 | 2500 | 2.519 |
| 5e+08 | 2e+11 | 400 | 2.419 |
| 1e+09 | 1e+11 | 100 | 2.386 |
| 2e+09 | 5e+10 | 25 | 2.384 |
| 4e+09 | 2e+10 | 6 | 2.415 |
| 7e+09 | 1e+10 | 2 | 2.462 |
| 2e+10 | 7e+09 | 0 | 2.560 |
| 3e+10 | 3e+09 | 0 | 2.687 |

### Budget $C=6e+21$ FLOPs

| $N$ | $D$ | $D/N$ | $L(N,D)$ |
|---:|---:|---:|---:|
| 2e+08 | 5e+12 | 25000 | 2.416 |
| 5e+08 | 2e+12 | 4000 | 2.286 |
| 1e+09 | 1e+12 | 1000 | 2.223 |
| 2e+09 | 5e+11 | 250 | 2.187 |
| 4e+09 | 2e+11 | 62 | 2.175 |
| 7e+09 | 1e+11 | 20 | 2.182 |
| 2e+10 | 7e+10 | 4 | 2.214 |
| 3e+10 | 3e+10 | 1 | 2.266 |

## 37. Pre-Registration Template for a Scaling Prediction

1. Study identifier and date (before the production run starts).
2. Architecture family, layer count, hidden size, and tokenizer identifier.
3. Data mixture identifier and filtering version (32.2).
4. Evaluation set and units (bits per byte if any tokenizer ambiguity exists).
5. Fitted law: functional form, coefficients, exponents, and floor $E$ (11.1).
6. Runs used in the fit, as a $(N,D,C,L)$ table (30, step 3).
7. Runs held out, and the prediction error on them (30, step 8).
8. Residual plot examined; curvature noted or explicitly denied (16).
9. Production $(N,D)$ chosen, with the binding-term argument (26.1).
10. Predicted production loss with an interval, not a point (30, step 9).
11. Batch size in tokens and LR rule for the production run (33).
12. Regime statement: compute-rich, data-constrained, or inference-dominated (13-14).
13. Named owner of the decision to stop, extend, or re-fit.
14. Actual production loss, filled in after the run, next to the prediction.
15. Post-run verdict: law confirmed, law revised, or regime change identified (16).

## 38. Expanded Glossary

| Term | Meaning |
|---|---|
| Allocation | How a compute budget is split between $N$ and $D$ |
| Asymptotic regime | Scales large enough that fitted exponents stabilize (16) |
| Binding term | The larger of $A/N^\alpha$ and $B/D^\beta$ (Problem 1) |
| Bootstrap over runs | Resampling the run table to interval the fit (30) |
| Compute law | Loss as a function of $C$ alone |
| Data term | $B/D^\beta$, error from finite tokens |
| Double descent | Non-monotone generalization sometimes motivating broken laws (16) |
| Entropy floor | $E$, irreducible loss (11.1) |
| Extrapolation | Predicting beyond fitted scales; budgeted in decades (7.5) |
| Holdout run | A run excluded from fitting and used to test prediction (7.3) |
| Interpolation | Prediction inside the fitted range; the safe case |
| Irreducible error | Synonym for $E$ in this chapter's usage |
| Log-log residual | Residual in transformed space; curvature there diagnoses regime change (16) |
| Marginal return | Loss reduction per added FLOP at a point on the curve |
| Parameter term | $A/N^\alpha$, error from finite capacity |
| Power law | $y\propto x^{-\alpha}$ relation |
| Pre-registration | Writing the prediction before the run (37) |
| Regime | Which constraint binds: compute, data, or inference (35) |
| Run table | The $(N,D,C,L)$ exhibit behind a fit (30) |
| Tokens per parameter | $D/N$, the allocation's most quoted summary |

---

## 39. Diagnostic Scenarios, Worked Step by Step

### Scenario 1: the fit that bends
Your log-log plot curves downward at large compute.
Step 1: do not add more points to the same fit yet.
Step 2: check whether the large runs share the small runs' data version (32.2).
Step 3: check per-run schedule quality at the large end (32.3).
Step 4: if both are clean, you have a candidate regime change.
Step 5: fit a broken law (Caballero et al., 2023) with the break located by the residuals, not by preference.
Report: both fits, their residuals, and which you used for the production prediction (37).

### Scenario 2: the ratio fight
Two teams cite Kaplan et al. (2020) and Hoffmann et al. (2022) for opposite allocations.
Step 1: notice both citations are to *fits under protocols*, not theorems (34).
Step 2: ask which protocol matches yours: per-run schedules and long small-model training, or not.
Step 3: settle it with one IsoFLOP slice in your own setup (12) at a budget you can afford.
Report: your slice's minimum $D/N$, with its runs table (30, step 3).

### Scenario 3: the data wall
Planning says $D=2$T tokens; the corpus has $U=300$B unique.
Step 1: compute epochs: $2000/300\approx6.7$ — deep into repetition discount (13, 25.3).
Step 2: re-plan with effective tokens, or shrink $N$ until $D/N$ at *effective* $D$ is sane.
Step 3: budget an epoch-value micro-study (measure loss at epochs 1, 2, 4 on a small model).
Report: unique tokens and epochs next to seen tokens everywhere $D$ appears (33's run table).

### Scenario 4: the serving surprise
The training-optimal model is chosen; projected serving is $5\times10^{12}$ tokens in year one.
Step 1: compute lifetime inference cost for the chosen $N$ and for a half-size overtrained alternative (14).
Step 2: if inference exceeds training by an order of magnitude, re-run the allocation on total cost (9.5).
Report: both training and lifetime numbers in the launch document (37, item 12's regime statement).

### Scenario 5: the miraculous benchmark
A capability appears only in the largest model, on an exact-match benchmark.
Step 1: obtain the continuous score behind the benchmark (15).
Step 2: plot it against $N$ and $C$ on the study's axes.
Step 3: if smooth, describe the result as metric thresholding (Schaeffer et al., 2023), however good the headline would have been.
Report: both plots, continuous first.

---

## 40. A Worked Log-Log Regression in Full

Synthetic run table (stated as such; the protocol is the content):

| Run | $N$ | $D$ (tokens) | $C=6ND$ | $L$ |
|---:|---:|---:|---:|---:|
| 1 | 1e8 | 2e9 | 1.2e18 | 2.85 |
| 2 | 3e8 | 6e9 | 1.1e19 | 2.62 |
| 3 | 1e9 | 2e10 | 1.2e20 | 2.41 |
| 4 | 3e9 | 6e10 | 1.1e21 | 2.24 |

Procedure:
1. Compute $C$ for every row yourself (30, step 3); never trust a copied compute column.
2. With $E$ provisionally 1.7, form $y=\log_{10}(L-E)$: values 0.06, -0.04, -0.15, -0.27 (approximately).
3. Against $x=\log_{10}C$ (18.1, 19.0, 20.1, 21.0), the slope is about $-0.11$ per decade: $\alpha_C\approx0.11$.
4. Refit with $E=1.9$ and $E=1.5$; if $\alpha_C$ moves materially, the floor and exponent are trading off — report both fits (11.1's identifiability warning in practice).
5. Hold out Run 4, fit Runs 1–3, predict Run 4, and state the error before using the law anywhere (30, step 8).

The notebook implements exactly this five-step protocol and prints its holdout error.

---

## 41. Scaling-Study Review Checklist

Setup identity:
1. Architecture family named and frozen for the study.
2. Tokenizer named; losses in commensurable units.
3. Data mixture and filtering version pinned.
4. Evaluator and held-out set untouched by training data work.
5. Any change to items 1–4 starts a new study, not a new point.

Run quality:
6. Every run has its own schedule, decayed over its own token count.
7. Tokens per step recorded per run.
8. LR rule under batch changes stated.
9. Batch checked against the noise-scale regime where feasible.
10. Failed/diverged runs listed, not deleted from the table.

Fitting:
11. $C=6ND$ recomputed for every row.
12. At least two fitting approaches compared.
13. Floor $E$ fitted or constrained with a stated reason.
14. Residuals plotted in log-log space and inspected.
15. Exponent stability checked by dropping the smallest decade.
16. Largest run held out; prediction error reported.
17. Bootstrap interval over runs reported for headline exponents.

Decision:
18. Regime declared: compute, data, or inference bound.
19. Production $(N,D)$ tied to the binding term.
20. Prediction pre-registered with an interval.
21. Serving cost included if the product serves at volume.
22. Epoch count stated wherever unique data binds.
23. A named owner can stop or extend the run against pre-agreed criteria.
24. Post-run, actual loss is written next to the prediction.
25. The study's run table is preserved as an exhibit, not summarized away.

## 42. Closing Summary in Twelve Statements

1. Scaling laws are fitted relations, not theorems (Section 1).
2. Their useful form includes an entropy floor (11.1).
3. Allocation optima are protocol-dependent (11.2, 34).
4. Three agreeing fitting methods beat one elegant fit (11.3).
5. IsoFLOP slices turn budgets into decisions (12).
6. Repetition discounts data (13).
7. Inference can dominate the true objective (14).
8. Threshold metrics manufacture emergence (15).
9. Curved residuals are findings (16).
10. Holdout prediction is the credibility test (30).
11. Pre-registration converts a fit into a forecast (37).
12. A scaling claim is its run table plus its residuals — nothing less (41).

---

## 43. Candidate Allocations at a Production Budget, Compared

Budget $C=6\times10^{22}$ FLOPs (so $ND=10^{22}$), losses from 17.1's coefficients. The exercise a launch review should perform with its own fit (21) instead of these constants.

| Candidate | $N$ | $D$ | $D/N$ | $L$ | Verdict from the terms |
|---|---:|---:|---:|---:|---|
| parameter-heavy | 7e+10 | 1e+11 | 2 | 2.083 | data term binds |
| large | 4e+10 | 2e+11 | 6 | 2.055 | data term binds |
| balanced-large | 2e+10 | 5e+11 | 25 | 2.035 | data term binds |
| Chinchilla-ray | 2e+10 | 5e+11 | 21 | 2.037 | data term binds |
| small-long | 7e+09 | 1e+12 | 204 | 2.035 | parameter term binds |
| tiny-marathon | 2e+09 | 5e+12 | 2500 | 2.084 | parameter term binds |

How to argue with this table in a review:

- Attack the coefficients first: they are Hoffmann et al. (2022)'s fit, on their setup (34). Your study's fit (21) replaces them.
- Attack the regime second: if serving volume is high, the balanced row is not the winner even if its training loss is (14).
- Attack the data third: if unique tokens cannot supply the chosen $D$, the row is fiction until effective tokens are used (13).
- What may not be attacked is the arithmetic: $6ND$, the two error terms, and the floor are the grammar every candidate must speak (11.1).

## 44. Consolidated References for the Expansion

- Kaplan et al., Scaling Laws for Neural Language Models, 2020, arXiv:2001.08361.
- Hoffmann et al., Training Compute-Optimal Large Language Models, 2022, arXiv:2203.15556.
- Muennighoff et al., Scaling Data-Constrained Language Models, 2023, arXiv:2305.16264.
- Schaeffer, Miranda, and Koyejo, Are Emergent Abilities of Large Language Models a Mirage?, NeurIPS 2023.
- Wei et al., Emergent Abilities of Large Language Models, TMLR 2022.
- Caballero et al., Broken Neural Scaling Laws, ICLR 2023.
- Touvron et al., LLaMA: Open and Efficient Foundation Language Models, 2023, arXiv:2302.13971.
- Hestness et al., Deep Learning Scaling is Predictable, Empirically, 2017.
- Rosenfeld et al., A Constructive Prediction of the Generalization Error Across Scales, ICLR 2020.
- McCandlish et al., An Empirical Model of Large-Batch Training, 2018 (batch-size regime, Section 33).

- Coefficients used in Sections 17, 25, 36, and 43 are Hoffmann et al. (2022)'s published Approach-3 values, reproduced for checkable arithmetic.
- Synthetic or illustrative numbers are labelled as such at their point of use (Sections 18, 28, 40).

---

## 45. Translating Compute Into Calendars

A scaling prediction becomes a plan only after $C$ is divided by *achieved* throughput (Training at Scale 7.4): device-days $=C/(	ext{peak FLOPs/s}\times\text{MFU}\times86400)$. Illustrative peak: $3\times10^{14}$ FLOPs/s per device.

| $C$ (FLOPs) | At MFU 0.30 (device-days) | At MFU 0.50 (device-days) | On 512 devices at MFU 0.50 (days) |
|---:|---:|---:|---:|
| 1e+20 | 13 | 8 | 0.0 |
| 1e+21 | 129 | 77 | 0.2 |
| 1e+22 | 1,286 | 772 | 1.5 |
| 6e+22 | 7,716 | 4,630 | 9.0 |
| 1e+23 | 12,860 | 7,716 | 15.1 |

Two planning lessons are in the columns, not the rows: MFU uncertainty (0.30 vs 0.50) is a 1.7x schedule uncertainty that no scaling exponent can rescue, and calendar time is bought with devices only until communication and stragglers (Training at Scale 21) tax the added fleet.

## 46. Worked Example: Sizing a Run From a Fit, End to End

1. Own fit (illustrative constants from 17.1) and a budget of $C=6\times10^{22}$ FLOPs.
2. Section 43's table says the balanced candidates near $N\approx2\times10^{10}$, $D\approx5\times10^{11}$ minimize the fitted loss at this budget.
3. Data check (13): the corpus must supply $5\times10^{11}$ tokens, fresh or at an epoch count whose discount you accept, in writing.
4. Inference check (14): if the deployment serves more than about $10^{12}$ tokens, re-run step 2 on total cost with a smaller candidate included.
5. Calendar check (45): at MFU 0.50 on 512 illustrative devices, $6\times10^{22}$ FLOPs is about 45 days — a commitment the pre-registration (37) should state next to the predicted loss.
6. Prediction: read $L$ from your fit at the chosen $(N,D)$, attach the bootstrap interval (30, step 9), and timestamp it.
7. After the run: fill item 14 of the pre-registration (37). The fit is now either evidence or a revised fit; both are progress, and only one of them was available before the run.

## 47. End of Expansion Notes

- Original Sections 1-10 are preserved above; Sections 11-47 are the phase-2 expansion.

- The companion theory notebook implements the fitting protocol of Sections 18, 30, and 40.

- Exercises and their learner workspaces are unchanged.

---

## 48. Loss Drops per Doubling, by Regime

From $L-E\propto C^{-\alpha_C}$: each doubling of compute multiplies excess loss by $2^{-\alpha_C}$.

| $\alpha_C$ | Excess-loss factor per doubling | Excess-loss factor per 10x |
|---:|---:|---:|
| 0.03 | 0.979 | 0.933 |
| 0.05 | 0.966 | 0.891 |
| 0.07 | 0.953 | 0.851 |
| 0.10 | 0.933 | 0.794 |
| 0.15 | 0.901 | 0.708 |
| 0.20 | 0.871 | 0.631 |

This table converts any fitted exponent into the only question management actually asks — what does the next increment buy — without rerunning the fit (cf. Training at Scale's one-line calculations, translated to scaling).

## 49. A Second Worked Fit: Two Floors, Two Answers

Using Section 40's synthetic table, fitting with $E=1.5$ versus $E=1.9$ shifts the fitted exponent because a higher floor leaves less excess loss to explain, steepening its required decline (40, step 4). The professional response is not to pick the prettier fit: it is to (a) report both, (b) prefer the one whose holdout prediction (30, step 8) is better, and (c) treat exponent-floor tradeoff as the study's headline uncertainty in the pre-registration (37, item 10's interval).

## 50. Final Formula and Reading Summary

| If you remember one thing | It is in |
|---|---|
| $L=E+A/N^\alpha+B/D^\beta$ | 11.1 |
| Allocate to the binding term | 26.1 |
| Three methods or it didn't happen | 11.3 |
| Repetition discounts data | 13 |
| Inference can own the objective | 14 |
| Continuous metrics before benchmark drama | 15 |
| Residuals are findings | 16 |
| Pre-register, then run | 37 |


### Reading order for a newcomer to this section

1. Sections 1-6 of the original text for the vocabulary of $N$, $D$, and $C$.
2. Section 11 of this expansion for the three-term law and where Chinchilla came from.
3. Sections 17 and 25 for the arithmetic with stated constants.
4. Section 21 or 30 when you must design a study rather than read one.
5. Sections 13-16 before quoting any ratio outside the regime it was fitted in.
6. Section 40's five-step regression with the notebook open, reproducing every number.

---

## 51. Roles in a Scaling Decision

| Role | Owns | Signs |
|---|---|---|
| Study designer | run table, commensurability (21) | the fit |
| Fitting owner | coefficients, residuals, holdout error (30) | the prediction |
| Data owner | unique tokens, epochs, filtering version (13, 32.2) | the regime statement |
| Systems owner | MFU evidence, calendar translation (45) | the schedule |
| Product owner | inference volume and latency constraints (14) | the total-cost objective |
| Decision owner | allocation against the binding term (26.1) | the pre-registration (37) |

A scaling decision with an unnamed row in this table will re-litigate that row's question after the money is spent.

## 52. Ten Closing Review Sentences

- A scaling law is a summary of runs you can show.
- Its exponents belong to a setup, not to nature.
- Its floor is a statement about data entropy.
- Its optimum is a statement about marginal returns balancing.
- Its extrapolation budget is measured in decades and spent carefully.
- Its residuals are the first place to look for new science.
- Its data axis counts unique information, not just tokens seen.
- Its compute axis is $6ND$, recomputed, never copied.
- Its product form includes every inference token you will ever serve.
- Its prediction was written down before the run, or it was not a prediction.

---

## 53. Study Design Table (Runs to Actually Train)

A concrete instantiation of Section 21: three IsoFLOP budgets, four splits each. Losses are intentionally blank — filling them is the study.

| Budget $C$ | $N$ | $D$ | $D/N$ | Schedule | Measured $L$ (fill) |
|---:|---:|---:|---:|---|---:|
| 6e+19 | 5e+08 | 2e+10 | 40 | cosine over own $D$ |  |
| 6e+19 | 2e+09 | 7e+09 | 4 | cosine over own $D$ |  |
| 6e+20 | 5e+08 | 2e+11 | 400 | cosine over own $D$ |  |
| 6e+20 | 2e+09 | 7e+10 | 44 | cosine over own $D$ |  |
| 6e+20 | 4e+09 | 2e+10 | 6 | cosine over own $D$ |  |
| 6e+20 | 1e+10 | 8e+09 | 1 | cosine over own $D$ |  |
| 6e+21 | 5e+08 | 2e+12 | 4000 | cosine over own $D$ |  |
| 6e+21 | 2e+09 | 7e+11 | 444 | cosine over own $D$ |  |
| 6e+21 | 4e+09 | 2e+11 | 62 | cosine over own $D$ |  |
| 6e+21 | 1e+10 | 8e+10 | 7 | cosine over own $D$ |  |

Twelve to fifteen runs, three U-shaped slices (12), two independent readings of the optimum (11.3), one held-out largest slice (30) — this table, completed, is a scaling study. Anything less is an anecdote with a trend line.

## 54. Notation and Units Reminder

| Symbol | Unit |
|---|---|
| $N$ | parameters (count) |
| $D, U$ | tokens (count) |
| $C$ | FLOPs |
| $L$ | nats per token unless converted to bits per byte |
| $E$ | same units as $L$ |
| exponents $\alpha,\beta,\alpha_C$ | dimensionless |

---

## 55. Epoch-Value Micro-Study (Design)

Purpose: measure your own repetition discount (13) instead of borrowing one.

1. Fix a small model and the target corpus.
2. Train four runs to the same seen-token count: 1 epoch over 100% of the slice, 2 epochs over 50%, 4 over 25%, 8 over 12.5%.
3. Hold unique-token *exposure* as the variable and seen tokens constant, so differences are repetition effects, not budget effects.
4. Evaluate on held-out fresh text and on a memorization probe (exact continuation of training passages).
5. Plot loss against epochs; the flattening point is your practical epoch budget.
6. Repeat the probe at the largest epoch: memorization rising while loss flattens is the signature of wasted epochs.
7. Feed the measured discount into 25.3's planning table, replacing the illustrative rows.
8. Record the corpus version with the result; a different mixture has a different discount.

| Run | Unique fraction | Epochs | Seen tokens | Fresh-text loss (fill) | Memorization probe (fill) |
|---:|---:|---:|---:|---:|---:|
| A | 100% | 1 | fixed |  |  |
| B | 50% | 2 | fixed |  |  |
| C | 25% | 4 | fixed |  |  |
| D | 12.5% | 8 | fixed |  |  |

## Common Mistakes

| Mistake | Why Wrong | Fix |
|---|---|---|
| Quoting "20 tokens per parameter" as a law of nature | It is one fit's optimum under one setup (11.2) | State regime, data, and schedule; refit for yours |
| Fitting loss vs steps instead of compute or tokens | Steps depend on batch size | Use $C\approx6ND$, $N$, or $D$ as the axis (Section 4) |
| Training IsoFLOP points with one shared LR schedule | Small models get schedules tuned for large ones (11.3) | Per-run schedules; this was the Kaplan-Chinchilla methodological gap |
| Extrapolating a fit 100x beyond fitted compute | Power laws are interpolations with modest reach (16) | Validate on a held-out larger run (7.3) |
| Ignoring the entropy floor $E$ | Predicts impossibly low loss at scale | Fit the three-term form (11.1) |
| Comparing scaling across tokenizers | Loss is per tokenizer's tokens (Ch15-05 Section 17) | Convert to bits per byte |
| Using benchmark accuracy as the fitted quantity | Threshold metrics manufacture "emergence" (15) | Fit continuous loss; treat accuracy as downstream |
| Declaring a data-constrained run Chinchilla-optimal | Repetition changes effective data (13) | Use effective tokens (Muennighoff et al. 2023) |
| Optimizing training cost for a serving product | Inference is paid per token forever (14) | Total-cost objective (Section 9) |
| Dropping small runs that break the line silently | They may mark a regime change (16) | Investigate residuals; consider broken-law fits |


---

## 17. Computed Tables for Scaling Laws

### 17.1 Illustrative three-term law values (11.1)

Using $E=1.69$, $A=406.4$, $\alpha=0.34$, $B=410.7$, $\beta=0.28$ — the Hoffmann et al. (2022) parametric-fit coefficients, reproduced here so the arithmetic can be checked (their Approach-3 values).

| $N$ | $D$ | $A/N^\alpha$ | $B/D^\beta$ | $L(N,D)$ |
|---:|---:|---:|---:|---:|
| 1e+09 | 2e+10 | 0.354 | 0.536 | 2.580 |
| 7e+09 | 1e+11 | 0.183 | 0.311 | 2.184 |
| 1e+10 | 2e+11 | 0.162 | 0.281 | 2.133 |
| 7e+10 | 1e+12 | 0.083 | 0.163 | 1.937 |
| 7e+10 | 2e+11 | 0.083 | 0.281 | 2.055 |
| 1e+09 | 1e+12 | 0.354 | 0.179 | 2.223 |

Read the last two data rows against each other: the 70B model on only 200B tokens loses to its Chinchilla-ratio self mostly through the $B/D^\beta$ term — undertraining made visible as arithmetic (6.3).

### 17.2 Tokens-per-parameter at a glance

| $N$ | $D$ at 5 tok/param | at 20 tok/param | at 100 tok/param |
|---:|---:|---:|---:|
| 1e+09 | 5e+09 | 2e+10 | 1e+11 |
| 7e+09 | 4e+10 | 1e+11 | 7e+11 |
| 1e+10 | 6e+10 | 3e+11 | 1e+12 |
| 7e+10 | 4e+11 | 1e+12 | 7e+12 |
| 2e+11 | 9e+11 | 4e+12 | 2e+13 |

### 17.3 Compute for the 20-tokens-per-parameter row ($C\approx6ND$)

| $N$ | $D=20N$ | $C\approx6ND$ (FLOPs) |
|---:|---:|---:|
| 1e+09 | 2e+10 | 1.2e+20 |
| 7e+09 | 1e+11 | 5.9e+21 |
| 1e+10 | 3e+11 | 2.0e+22 |
| 7e+10 | 1e+12 | 5.9e+23 |

## 18. Fitting a Power Law by Hand (Worked)

Data (synthetic but realistic, stated as such): losses 2.60, 2.45, 2.32, 2.20 at compute $10^{20},10^{21},10^{22},10^{23}$ FLOPs.

1. Work in $x=\log_{10}C$, $y=\log_{10}(L-E)$; first try $E=0$ to see the naive fit.

2. Successive loss drops are 0.15, 0.13, 0.12 per decade — *shrinking*, which a pure power law in $L$ (constant fractional drop) cannot express; this signals a floor $E>0$ (11.1).

3. With $E=1.8$: excess losses are 0.80, 0.65, 0.52, 0.40; $\log_{10}$ excess drops by about 0.09-0.11 per decade, giving $\alpha_C\approx0.10$ in $L-E\propto C^{-\alpha_C}$.

4. Hold out the largest point, refit on three, predict the fourth, and only then believe the exponent (7.3, 7.5) — the notebook performs this exact protocol on toy data.

## 19. Glossary

| Broken power law | Scaling law whose exponent changes across regimes (16) |
| Chinchilla ratio | ~20 tokens/parameter compute-optimal region of Hoffmann et al. 2022 (11.2) |
| Data-constrained scaling | Regime where unique tokens, not compute, bind (13) |
| Effective tokens | Repetition-discounted data quantity (13) |
| Entropy floor $E$ | Irreducible loss of the data distribution (11.1) |
| IsoFLOP | Curve of equal compute across $N/D$ splits (12) |
| Overtraining | Training past the training-compute optimum for inference savings (14) |
| Scaling law | Fitted relation between resources and loss (Section 1) |
| Emergence (metric artefact) | Apparent discontinuity from thresholded metrics (15) |
---

## 20. Ten Worked Problems With Solutions

### Problem 1: which term binds?
Using 17.1's coefficients for $N=70$B, $D=200$B: parameter term $406.4/(7\times10^{10})^{0.34}\approx0.33$, data term $410.7/(2\times10^{11})^{0.28}\approx0.86$. The data term is more than twice the parameter term — this model is data-limited, so the next FLOP should buy tokens, not parameters (11.1).

### Problem 2: compute-optimal growth
You have 10x more compute than a run at the optimum. Under Hoffmann et al. (2022)'s roughly equal scaling (11.2), grow $N$ and $D$ each by about $\sqrt{10}\approx3.2$x, keeping $D/N$ near constant — not $N$ by 10x at fixed $D$.

### Problem 3: spotting undertraining
A 175B model trained on 300B tokens sits at $D/N\approx1.7$. By 17.2's table it is far on the parameter-heavy side; expect the symptoms of 6.3 (loss above the IsoFLOP minimum for its compute). This was the standard pre-Chinchilla configuration and the revision's main target.

### Problem 4: IsoFLOP budget division
Budget $C=6\times10^{20}$ FLOPs, so $ND=10^{20}$. Candidates: $(N,D)=(1\text{B},100\text{B}), (3.2\text{B},31\text{B}), (10\text{B},10\text{B})$ with $D/N$ of 100, 10, 1. Training all three properly (own schedules, 12) and reading the U-shaped minimum is a complete small IsoFLOP study.

### Problem 5: repetition discount
Corpus $U=50$B unique tokens, budget wants $D=200$B seen tokens = 4 epochs. By Muennighoff et al. (2023) (13), value this as close to fresh data but not equal; a plan that treats it as 200B fresh tokens overstates its data term in 11.1.

### Problem 6: inference-aware choice
Two candidates reach the same training loss proxy: 7B at 20 tok/param (140B tokens) and 7B at 100 tok/param is wrong comparison — fix $N$ and vary $D$ changes *training* loss. Correct form: a 3B model trained long can match a 7B trained short on loss while costing less than half per served token (14). For a product serving $10^{12}$ tokens, inference dominates and the smaller long-trained model wins total cost (9.5).

### Problem 7: emergence or artefact?
Benchmark accuracy jumps from 5% to 60% between two model sizes while per-token log-likelihood of correct answers improves smoothly by 0.05 nats. By Schaeffer et al. (2023) (15), report the smooth quantity as the scaling result and the jump as a threshold artefact of exact-match scoring.

### Problem 8: floor sanity
A fit reports $E=0.9$ nats, but a unigram baseline on the same data has entropy 2.1 nats and the best known model sits at 1.75. The floor is below an achieved loss's plausible limit given the baseline structure — by 16's second diagnostic the fit is extrapolating wrongly; refit with the floor constrained or more large-scale points.

### Problem 9: tokenizer trap in scaling comparisons
Model A (32k vocab) reports lower per-token loss than Model B (256k vocab) at equal compute. Convert both to bits per byte (Ch15-05 Section 16) before placing them on one scaling plot; per-token losses across vocabularies measure different event spaces (Section 3.5).

### Problem 10: when to stop trusting the line
Fitted on runs up to $10^{21}$ FLOPs, the law predicts a loss at $10^{24}$ that requires beating the entropy floor trend of 18's worked example. Stop: 1000x extrapolation with a shrinking-drops pattern is exactly 7.5's failure case; commission an intermediate run at $10^{22}$–$10^{23}$ first.

---

## 21. A Complete Small Scaling Study, Designed

Objective: decide $N$ and $D$ for a $6\times10^{21}$-FLOP production run in your own setup (Section 7's workflow, operationalized).

1. Fix architecture family, tokenizer, data mixture, and evaluation (bits per byte on a held-out set) for the whole study.
2. Choose IsoFLOP budgets at $10^{19}$, $10^{20}$, $10^{21}$ FLOPs — three decades below/at the decision's doorstep (12).
3. At each budget train 4–5 $(N,D)$ splits spanning $D/N$ from ~2 to ~100 (Problem 4's pattern).
4. Give every run its own cosine schedule ending at its own token count — the methodological lesson of 11.2/11.3.
5. Fit the U-minimum per budget (approach 2) and the parametric law (approach 1... in 11.3's numbering, approach 3) to all points; require the two to roughly agree.
6. Hold out the largest budget's minimum during fitting, then compare prediction to measurement (7.3); report the error, not just the fit.
7. Examine residuals in log-log space for curvature (16) before extrapolating one decade to the production budget.
8. Only then size the production run, and pre-register the predicted loss so the production run becomes the study's final holdout test.

Cost honesty: step 3 trains 12–15 models. That is the price of a scaling claim; a budget that cannot afford the study should fund a smaller production bet instead of an unfounded large one.

## 22. Historical Line

| Year | Work | Idea used here |
|---:|---|---|
| 2017 | Hestness et al., deep learning scaling is predictable | Early empirical power laws in training |
| 2020 | Kaplan et al., Scaling Laws for Neural Language Models | $L(N),L(D),L(C)$ power laws; parameter-leaning allocation (11.2) |
| 2020 | Rosenfeld et al., constructive prediction of generalization error | Parametric loss forms with a floor |
| 2022 | Hoffmann et al., Chinchilla | Three-method fit; ~20 tok/param; schedule methodology (11) |
| 2022 | Wei et al., emergent abilities | Scale-threshold phenomena as reported on benchmarks (15) |
| 2023 | Schaeffer et al., Are Emergent Abilities a Mirage? | Metric-artefact account of emergence (15) |
| 2023 | Muennighoff et al., data-constrained scaling | Effective tokens under repetition (13) |
| 2023 | Caballero et al., Broken Neural Scaling Laws | Exponent-changing fits (16) |
| 2023 | Touvron et al., LLaMA | Inference-motivated overtraining (14) |

## 23. Formula Summary

| Quantity | Formula |
|---|---|
| Compute | $C\approx6ND$ |
| Three-term law | $L=E+A/N^\alpha+B/D^\beta$ |
| Compute law (one-variable) | $L-E\propto C^{-\alpha_C}$ |
| IsoFLOP constraint | $6ND=C$ fixed, vary $N$ |
| Data-constrained value | effective tokens discounts epochs beyond the first (13) |
| Total deployment cost | $C_{\text{train}}+\text{tokens served}\times$ per-token inference cost (14) |

## 24. Final Self-Check

1. Given $N$ and $D$, can you say which term of 11.1 binds (Problem 1)?
2. Given 10x compute, can you allocate it and defend the split (Problem 2)?
3. Can you design an IsoFLOP study that would survive 11.3's three-method test (21)?
4. Can you state when "20 tokens per parameter" does not apply (13, 14)?
5. Can you distinguish emergence from a metric artefact with one plot (15)?
6. Can you read a residual plot and name the failure it implies (16)?


---

## 25. Computed Scaling Tables

All values from 17.1's stated Hoffmann et al. (2022) coefficients — a checkable calculator, not new fitted results.

### 25.1 Loss along the 20-tokens-per-parameter ray

| $N$ | $D=20N$ | $C\approx6ND$ | $L(N,D)$ |
|---:|---:|---:|---:|
| 1e+08 | 2e+09 | 1.2e+18 | 3.486 |
| 3e+08 | 6e+09 | 1.1e+19 | 2.974 |
| 1e+09 | 2e+10 | 1.2e+20 | 2.580 |
| 3e+09 | 6e+10 | 1.1e+21 | 2.328 |
| 7e+09 | 1e+11 | 5.9e+21 | 2.184 |
| 1e+10 | 3e+11 | 2.0e+22 | 2.099 |
| 3e+10 | 6e+11 | 1.1e+23 | 2.008 |
| 7e+10 | 1e+12 | 5.9e+23 | 1.937 |
| 2e+11 | 4e+12 | 3.7e+24 | 1.877 |

### 25.2 Loss at fixed compute, varying the split (IsoFLOP slice, $C=6\times10^{21}$)

| $N$ | $D=C/(6N)$ | $D/N$ | $L(N,D)$ |
|---:|---:|---:|---:|
| 5e+08 | 2e+12 | 4000 | 2.286 |
| 1e+09 | 1e+12 | 1000 | 2.223 |
| 2e+09 | 5e+11 | 250 | 2.187 |
| 4e+09 | 2e+11 | 62 | 2.175 |
| 7e+09 | 1e+11 | 20 | 2.182 |
| 1e+10 | 1e+11 | 10 | 2.193 |
| 2e+10 | 5e+10 | 2 | 2.233 |
| 4e+10 | 2e+10 | 1 | 2.295 |

The U shape of Section 12 is visible in the numbers: the middle rows beat both extremes, and the best row is the one a production decision should start from — subject to your own fit replacing these coefficients (21).

### 25.3 Effect of epochs under repetition (illustrative discount, Section 13)

Treating epochs 1-4 as nearly full value and later epochs as discounted is Muennighoff et al. (2023)'s qualitative finding; the discount factors below are an *illustrative* reading for planning arithmetic, not their fitted constants.

| Epochs on $U$ unique tokens | Seen tokens | Planning value |
|---:|---:|---|
| 1 | $U$ | full |
| 2 | $2U$ | nearly full |
| 4 | $4U$ | still useful, measure it |
| 8 | $8U$ | heavily discounted; prefer fresh data or a smaller model |
| 16 | $16U$ | mostly memorization pressure; justify explicitly |

## 26. Decision Worksheets

### 26.1 'Should we train bigger or longer?' — answered in order

1. Compute both terms of 11.1 at your current $(N,D)$ (Problem 1's arithmetic).
2. The larger term names the binding constraint; spend the next increment there.
3. Check the regime: data-constrained (13) or inference-dominated (14) answers override the training-only optimum.
4. Confirm with one IsoFLOP slice at the new budget if the decision is expensive (12, 21).
5. Write the predicted loss down before training (21, step 8).

### 26.2 'Our scaling plot looks broken' — answered in order

1. Plot residuals in log-log space; curvature is information, not noise (16).
2. Check tokenizer comparability of every point (Problem 9).
3. Check that each point had its own schedule (Common Mistakes, row 3).
4. Drop the smallest decade and refit; if exponents move, you were fitting the pre-asymptotic regime (16).
5. If curvature survives all checks, fit a broken law (Caballero et al. 2023) and name the regime change you believe in.

## 27. Five Scenarios

### Scenario A: the undertrained flagship
A lab trains 70B on 200B tokens and is disappointed. 17.1's arithmetic already predicted it: data term dominates (Problem 1). Prescription: same compute redirected to a smaller, longer run would have won on loss (12).

### Scenario B: the serving product that overpaid for training optimality
A Chinchilla-optimal mid-size model serves billions of tokens daily. Section 14: lifetime cost is inference-led; a smaller overtrained model (Touvron et al. 2023's stated reasoning) cuts serving cost at equal quality. Prescription: re-optimize on total cost (9.5), not training cost.

### Scenario C: the emergent miracle
A benchmark jumps at the largest size. Before announcing emergence, plot the continuous score (15): if it is smooth, Schaeffer et al. (2023) have already explained your miracle.

### Scenario D: the fourth epoch that wasn't
A data-constrained team plans 8 epochs as if tokens were fresh. Section 13 and 25.3: discount the repetitions, consider a smaller model, and measure epoch value on a small study first.

### Scenario E: the beautiful straight line
A log-log fit through four small runs predicts a loss below the data's plausible floor at production scale. Problem 8/18: the floor term is missing or the extrapolation is out of range (7.5). Commission the intermediate run.

## Practice Exercises

1. Fit a one-resource power law after subtracting a known loss floor.
2. Convert parameters and tokens into approximate dense training FLOPs.
3. Compute token count from a fixed compute budget and model size.
4. Search an IsoFLOP curve for the lowest toy loss.
5. Diagnose whether a model is undertrained by checking tokens per parameter.
6. Compute effective tokens for a data mixture.
7. Estimate serving cost for two model choices.
8. Compute residuals for scaling-law forecasts.
9. Show how a thresholded metric can look abrupt even when loss is smooth.
10. Write a forecast checklist with uncertainty.

## Why This Matters for AI

| Concept | AI Impact |
|---|---|
| Compute-optimal allocation (Hoffmann et al., 2022) | Chinchilla's 70B/1.4T-token training decision and the field's reallocation toward data |
| Power-law forecasting (Kaplan et al., 2020) | Budget planning for GPT-class training runs |
| Inference-aware overtraining (Touvron et al., 2023) | Small long-trained models for high-volume serving (LLaMA rationale) |
| Data-constrained scaling (Muennighoff et al., 2023) | Training when unique data, not compute, is the constraint |
| Metric-artefact emergence (Schaeffer et al., 2023) | Continuous-metric evaluation of claimed emergent abilities |

Scaling laws turn expensive guessing into disciplined planning. They help decide whether to collect more data, train longer, use a smaller model, improve data quality, or avoid a run whose expected gain is too small. They also teach humility: a clean line on a log-log plot is useful, but it is still a fit to a regime, not a theorem about intelligence.

## Bridge to Efficient Attention and Inference

The next section studies how attention and inference systems change the cost side of the scaling equation. Scaling laws say what loss might be worth buying; efficient attention and serving math decide whether that loss can be bought and served under memory, latency, and throughput constraints.

## References

- Jared Kaplan et al., "Scaling Laws for Neural Language Models", 2020: https://arxiv.org/abs/2001.08361
- Jordan Hoffmann et al., "Training Compute-Optimal Large Language Models", 2022: https://arxiv.org/abs/2203.15556
- Tom Henighan et al., "Scaling Laws for Autoregressive Generative Modeling", 2020: https://arxiv.org/abs/2010.14701
- Ethan Caballero et al., "Broken Neural Scaling Laws", 2022: https://arxiv.org/abs/2210.14891
- Sam McCandlish et al., "An Empirical Model of Large-Batch Training", 2018: https://arxiv.org/abs/1812.06162
- Yonatan Bisk et al., "Experience Grounds Language", 2020, for caution about benchmarks and grounding limits: https://arxiv.org/abs/2004.10151
- Expansion sections: 11-54. Original sections preserved.
- Next chapter topic: efficient attention and inference, where the inference-cost term of Section 14 becomes the main object.
- Reading the primary sources in the order of Section 34 is recommended before designing a study.
- The theory notebook is the executable companion to Sections 17, 18, 25, 30, and 40.
- Exercises remain in their original form, with learner workspaces intact.
- Notation follows docs/NOTATION_GUIDE.md; notebook plots follow docs/VISUALIZATION_GUIDE.md.
- No external datasets are required to reproduce the notebook's calculations.
- All fitted constants borrowed from published work are attributed at the point of use.
- All synthetic numbers are labelled synthetic or illustrative at the point of use.
- The production decision this section prepares is an allocation, not a slogan.
- The next section converts the inference term of total cost into detailed attention and serving mathematics.
- The fitted constants in this expansion are attributed to Hoffmann et al. (2022) wherever they appear.
- Kaplan et al. (2020) is cited for the original power-law framework and its protocol context.
- Muennighoff et al. (2023) is cited for repetition-aware, data-constrained scaling.
- Schaeffer et al. (2023) is cited for the metric-artefact account of emergent abilities.
- Caballero et al. (2023) is cited for broken scaling laws with changing exponents.
- Touvron et al. (2023) is cited for inference-motivated training beyond the training-compute optimum.
- No unpublished or invented scaling result is presented as established in this expansion.
- Readers should replace every borrowed constant with a fit from their own commensurable runs.
- The section's exercises test the same calculations in learner-workspace form.
- End of section.
- Companion notebooks are in this directory.
- Original content above is preserved.
- Expansion content is attributed where it borrows published constants.
- Learner workspaces in exercises.ipynb are intentionally left for the reader.
- The next run in this phase continues with Chapter 15 Section 09.
