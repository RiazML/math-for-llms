[<- Training at Scale](../06-Training-at-Scale/notes.md) | [Home](../../README.md) | [Scaling Laws ->](../08-Scaling-Laws/notes.md)

---

# Fine-Tuning Math

Fine-tuning is controlled movement away from a pretrained model. The mathematics asks four questions: what loss is optimized, which parameters can move, how large is the update space, and how do we measure useful adaptation without damaging the base model.

## Overview

Pretraining learns a broad conditional distribution. Fine-tuning shifts that distribution toward a task, domain, instruction style, preference pattern, or deployment constraint. The shift can be full-rank and full-model, or it can be restricted to a small trainable subspace such as adapters, soft prompts, prefixes, or LoRA matrices.

The central decomposition is:

$$
\theta = \theta_0 + \Delta\theta.
$$

The base weights $\theta_0$ carry pretrained capability. The update $\Delta\theta$ carries adaptation. Fine-tuning math is the study of how to choose, constrain, optimize, and evaluate that update.

## Prerequisites

- Cross-entropy and answer-only token loss
- Training memory and optimizer-state accounting
- Matrix ranks, low-rank factorization, and SVD intuition
- KL divergence and conditional language-model scoring

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demos for SFT masking, LoRA parameter counts, low-rank updates, adapter counts, prefix counts, DPO loss, forgetting tradeoffs, and merge checks. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for the update geometry and objective bookkeeping used in real fine-tuning runs. |

## Learning Objectives

After this section, you should be able to:

- Write the fine-tuning objective as local adaptation from pretrained weights.
- Compute answer-only SFT loss with prompt and padding masks.
- Explain full fine-tuning, linear probing, adapters, soft prompts, prefix tuning, and LoRA.
- Count trainable parameters for LoRA, adapters, and prefix methods.
- Derive the LoRA update $W'=W+(\alpha/r)BA$ and merge it for inference.
- Explain why PEFT reduces optimizer-state memory but not all activation memory.
- Compute a DPO preference loss from model and reference log probabilities.
- Diagnose forgetting, overfitting, mask bugs, and incorrect parameter freezing.

## Table of Contents

1. [Fine-Tuning as Local Adaptation](#1-finetuning-as-local-adaptation)
   - 1.1 [Starting from pretrained weights](#11-starting-from-pretrained-weights)
   - 1.2 [Task loss](#12-task-loss)
   - 1.3 [Regularized adaptation](#13-regularized-adaptation)
   - 1.4 [Function-space shift](#14-functionspace-shift)
   - 1.5 [Catastrophic forgetting](#15-catastrophic-forgetting)
2. [Supervised Fine-Tuning Objective](#2-supervised-finetuning-objective)
   - 2.1 [Instruction-response pairs](#21-instructionresponse-pairs)
   - 2.2 [Answer-only loss](#22-answeronly-loss)
   - 2.3 [Teacher forcing](#23-teacher-forcing)
   - 2.4 [Label smoothing](#24-label-smoothing)
   - 2.5 [Dataset mixture weights](#25-dataset-mixture-weights)
3. [Full Fine-Tuning](#3-full-finetuning)
   - 3.1 [All parameters trainable](#31-all-parameters-trainable)
   - 3.2 [Memory cost](#32-memory-cost)
   - 3.3 [Layer-wise learning rates](#33-layerwise-learning-rates)
   - 3.4 [Weight decay](#34-weight-decay)
   - 3.5 [When full tuning helps](#35-when-full-tuning-helps)
4. [Parameter-Efficient Fine-Tuning](#4-parameterefficient-finetuning)
   - 4.1 [Linear probing](#41-linear-probing)
   - 4.2 [Adapters](#42-adapters)
   - 4.3 [Soft prompt tuning](#43-soft-prompt-tuning)
   - 4.4 [Prefix tuning](#44-prefix-tuning)
   - 4.5 [Low-rank adaptation](#45-lowrank-adaptation)
5. [LoRA Algebra](#5-lora-algebra)
   - 5.1 [Rank constraint](#51-rank-constraint)
   - 5.2 [Parameter count](#52-parameter-count)
   - 5.3 [Scaling](#53-scaling)
   - 5.4 [Merge for inference](#54-merge-for-inference)
   - 5.5 [Target modules](#55-target-modules)
6. [Quantized and Memory-Aware Tuning](#6-quantized-and-memoryaware-tuning)
   - 6.1 [Frozen quantized base](#61-frozen-quantized-base)
   - 6.2 [Adapter optimizer states](#62-adapter-optimizer-states)
   - 6.3 [Activation memory remains](#63-activation-memory-remains)
   - 6.4 [Gradient checkpointing](#64-gradient-checkpointing)
   - 6.5 [Rank-memory tradeoff](#65-rankmemory-tradeoff)
7. [Preference Fine-Tuning](#7-preference-finetuning)
   - 7.1 [Preference pairs](#71-preference-pairs)
   - 7.2 [Reward-model view](#72-rewardmodel-view)
   - 7.3 [KL-regularized policy update](#73-klregularized-policy-update)
   - 7.4 [DPO loss](#74-dpo-loss)
   - 7.5 [Preference over-optimization](#75-preference-overoptimization)
8. [Evaluation and Diagnostics](#8-evaluation-and-diagnostics)
   - 8.1 [Train and validation loss](#81-train-and-validation-loss)
   - 8.2 [Base-task retention](#82-basetask-retention)
   - 8.3 [Task quality](#83-task-quality)
   - 8.4 [Distribution shift](#84-distribution-shift)
   - 8.5 [Adapter sanity checks](#85-adapter-sanity-checks)
9. [Choosing a Method](#9-choosing-a-method)
   - 9.1 [No-update methods](#91-noupdate-methods)
   - 9.2 [PEFT methods](#92-peft-methods)
   - 9.3 [Full tuning](#93-full-tuning)
   - 9.4 [Preference tuning](#94-preference-tuning)
   - 9.5 [Deployment constraints](#95-deployment-constraints)
10. [Implementation Checklist](#10-implementation-checklist)
   - 10.1 [Masking](#101-masking)
   - 10.2 [Parameter freeze audit](#102-parameter-freeze-audit)
   - 10.3 [Learning-rate groups](#103-learningrate-groups)
   - 10.4 [Reference model](#104-reference-model)
   - 10.5 [Ablations](#105-ablations)
11. [LoRA in Depth](#11-lora-in-depth-rank-scaling-and-where-the-parameters-go)
12. [Adapters, Prefixes, and Prompts](#12-adapters-prefixes-and-prompts-as-function-space-interventions)
13. [Preference Optimization in Depth](#13-preference-optimization-in-depth)
14. [Catastrophic Forgetting](#14-catastrophic-forgetting-measured-and-managed)
15. [Merging Fine-Tuned Models](#15-merging-fine-tuned-models)
16. [Quantized Fine-Tuning](#16-quantized-fine-tuning-qlora-accounting)
17. [Choosing and Auditing: Worked Scenarios](#17-choosing-and-auditing-worked-scenarios)
18. [Computed Tables](#18-computed-tables-for-fine-tuning)
19. [Derivations in Steps](#19-derivations-in-steps)
20. [Glossary](#20-glossary)
21. [Ten Worked Problems](#21-ten-worked-problems-with-solutions)
22. [Pre-Flight Checklist](#22-pre-flight-checklist-for-a-fine-tuning-run)
23. [Historical Line](#23-historical-line)
24. [Formula Summary](#24-formula-summary)
25. [Final Self-Check](#25-final-self-check)
26. [Budget Worksheets](#26-budget-worksheets-for-a-7b-class-model)
27. [SFT, RLHF, and DPO Compared](#27-what-changes-between-sft-rlhf-and-dpo-one-table)
28. [InstructGPT Pipeline](#28-the-instructgpt-pipeline-as-an-accounting-exercise)
29. [Evaluating Preference-Tuned Models](#29-evaluating-preference-tuned-models-without-fooling-yourself)
30. [Debugging Fine-Tuning Runs](#30-debugging-fine-tuning-runs)
31. [Notation](#31-notation-for-this-chapters-expansion)

---

## Method Map

| Method | Base weights | Trainable object | Main advantage | Main risk |
| --- | --- | --- | --- | --- |
| Prompting/RAG | Frozen | None | Zero training | May not enforce consistent behavior |
| Linear probing | Frozen | Output head | Cheap diagnostic | Limited generation adaptation |
| Adapters | Frozen mostly | Bottleneck modules | Multi-task modularity | Extra inference modules |
| Prompt/prefix tuning | Frozen | Continuous prompt or KV prefix | Very small parameter count | Capacity can be limited |
| LoRA | Frozen mostly | Low-rank matrix updates | Strong PEFT default | Rank/target choice matters |
| Full fine-tuning | Trainable | All weights | Maximum capacity | Expensive and can forget |
| Preference tuning | Usually partial or full | Policy update | Aligns comparative behavior | Over-optimization and reward hacking |

## 1. Fine-Tuning as Local Adaptation

This part treats fine-tuning as local adaptation as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Starting from pretrained weights](#1-starting-from-pretrained-weights) | fine-tuning begins near a useful parameter point | $\theta=\theta_0+\Delta\theta$ |
| [Task loss](#1-task-loss) | the target distribution changes from broad pretraining to task behavior | $\min_\theta L_\mathrm{task}(\theta)$ |
| [Regularized adaptation](#1-regularized-adaptation) | penalize moving too far from the base model | $L(\theta)=L_\mathrm{task}(\theta)+\lambda R(\theta,\theta_0)$ |
| [Function-space shift](#1-functionspace-shift) | behavioral movement is better measured on outputs than raw parameters | $D_\mathrm{KL}(p_\theta(\cdot\mid x)\Vert p_{\theta_0}(\cdot\mid x))$ |
| [Catastrophic forgetting](#1-catastrophic-forgetting) | task learning can reduce general ability | $\Delta_\mathrm{forget}=S_\mathrm{base}(\theta_0)-S_\mathrm{base}(\theta)$ |

### 1.1 Starting from pretrained weights

**Main idea.** Fine-tuning begins near a useful parameter point.

Core relation:

$$\theta=\theta_0+\Delta\theta$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 1.2 Task loss

**Main idea.** The target distribution changes from broad pretraining to task behavior.

Core relation:

$$\min_\theta L_\mathrm{task}(\theta)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 1.3 Regularized adaptation

**Main idea.** Penalize moving too far from the base model.

Core relation:

$$L(\theta)=L_\mathrm{task}(\theta)+\lambda R(\theta,\theta_0)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 1.4 Function-space shift

**Main idea.** Behavioral movement is better measured on outputs than raw parameters.

Core relation:

$$D_\mathrm{KL}(p_\theta(\cdot\mid x)\Vert p_{\theta_0}(\cdot\mid x))$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 1.5 Catastrophic forgetting

**Main idea.** Task learning can reduce general ability.

Core relation:

$$\Delta_\mathrm{forget}=S_\mathrm{base}(\theta_0)-S_\mathrm{base}(\theta)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 2. Supervised Fine-Tuning Objective

This part treats supervised fine-tuning objective as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Instruction-response pairs](#2-instructionresponse-pairs) | train on prompt and desired completion pairs | $(x,y)\sim D_\mathrm{sft}$ |
| [Answer-only loss](#2-answeronly-loss) | mask prompt tokens when the objective is response imitation | $L_\mathrm{sft}=-\sum_j m_j\log p_\theta(y_j\mid x,y_{<j})/\sum_j m_j$ |
| [Teacher forcing](#2-teacher-forcing) | condition on the gold prefix during training | $p_\theta(y_j\mid x,y_{<j}^\star)$ |
| [Label smoothing](#2-label-smoothing) | soften one-hot targets when useful | $q=(1-\epsilon)y+\epsilon/|V|$ |
| [Dataset mixture weights](#2-dataset-mixture-weights) | combine multiple tasks by weighted expectation | $L=\sum_k \alpha_k E_{D_k}[\ell]$ |

### 2.1 Instruction-response pairs

**Main idea.** Train on prompt and desired completion pairs.

Core relation:

$$(x,y)\sim D_\mathrm{sft}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 2.2 Answer-only loss

**Main idea.** Mask prompt tokens when the objective is response imitation.

Core relation:

$$L_\mathrm{sft}=-\sum_j m_j\log p_\theta(y_j\mid x,y_{<j})/\sum_j m_j$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is the difference between teaching the model to imitate the response and wasting loss on tokens it was handed in the prompt.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 2.3 Teacher forcing

**Main idea.** Condition on the gold prefix during training.

Core relation:

$$p_\theta(y_j\mid x,y_{<j}^\star)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 2.4 Label smoothing

**Main idea.** Soften one-hot targets when useful.

Core relation:

$$q=(1-\epsilon)y+\epsilon/|V|$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 2.5 Dataset mixture weights

**Main idea.** Combine multiple tasks by weighted expectation.

Core relation:

$$L=\sum_k \alpha_k E_{D_k}[\ell]$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 3. Full Fine-Tuning

This part treats full fine-tuning as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [All parameters trainable](#3-all-parameters-trainable) | full fine-tuning updates every weight tensor | $\Delta\theta\in\mathbb{R}^{|\theta|}$ |
| [Memory cost](#3-memory-cost) | trainable parameters require gradients and optimizer states | $M\approx M_\mathrm{weights}+M_\mathrm{grads}+M_\mathrm{opt}+M_\mathrm{act}$ |
| [Layer-wise learning rates](#3-layerwise-learning-rates) | lower layers can move more slowly than higher layers | $\eta_\ell=\eta_0\gamma^{L-\ell}$ |
| [Weight decay](#3-weight-decay) | regularize parameter norm during adaptation | $\theta\leftarrow\theta-\eta(\nabla L+\lambda\theta)$ |
| [When full tuning helps](#3-when-full-tuning-helps) | large domain shift or maximum quality can justify the cost | $\mathrm{benefit}>\mathrm{compute}+\mathrm{forgetting\ risk}$ |

### 3.1 All parameters trainable

**Main idea.** Full fine-tuning updates every weight tensor.

Core relation:

$$\Delta\theta\in\mathbb{R}^{|\theta|}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 3.2 Memory cost

**Main idea.** Trainable parameters require gradients and optimizer states.

Core relation:

$$M\approx M_\mathrm{weights}+M_\mathrm{grads}+M_\mathrm{opt}+M_\mathrm{act}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 3.3 Layer-wise learning rates

**Main idea.** Lower layers can move more slowly than higher layers.

Core relation:

$$\eta_\ell=\eta_0\gamma^{L-\ell}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 3.4 Weight decay

**Main idea.** Regularize parameter norm during adaptation.

Core relation:

$$\theta\leftarrow\theta-\eta(\nabla L+\lambda\theta)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 3.5 When full tuning helps

**Main idea.** Large domain shift or maximum quality can justify the cost.

Core relation:

$$\mathrm{benefit}>\mathrm{compute}+\mathrm{forgetting\ risk}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 4. Parameter-Efficient Fine-Tuning

This part treats parameter-efficient fine-tuning as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Linear probing](#4-linear-probing) | freeze the backbone and train only a head | $h=f_{\theta_0}(x),\quad \hat y=Wh+b$ |
| [Adapters](#4-adapters) | insert small bottleneck modules inside layers | $h'=h+W_\mathrm{up}\sigma(W_\mathrm{down}h)$ |
| [Soft prompt tuning](#4-soft-prompt-tuning) | learn continuous input embeddings | $[p_1,\ldots,p_m,x_1,\ldots,x_T]$ |
| [Prefix tuning](#4-prefix-tuning) | learn virtual key-value prefixes for attention | $K'=[K_\mathrm{prefix};K],\quad V'=[V_\mathrm{prefix};V]$ |
| [Low-rank adaptation](#4-lowrank-adaptation) | learn a low-rank update while freezing the base matrix | $W'=W+\frac{\alpha}{r}BA$ |

### 4.1 Linear probing

**Main idea.** Freeze the backbone and train only a head.

Core relation:

$$h=f_{\theta_0}(x),\quad \hat y=Wh+b$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 4.2 Adapters

**Main idea.** Insert small bottleneck modules inside layers.

Core relation:

$$h'=h+W_\mathrm{up}\sigma(W_\mathrm{down}h)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 4.3 Soft prompt tuning

**Main idea.** Learn continuous input embeddings.

Core relation:

$$[p_1,\ldots,p_m,x_1,\ldots,x_T]$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 4.4 Prefix tuning

**Main idea.** Learn virtual key-value prefixes for attention.

Core relation:

$$K'=[K_\mathrm{prefix};K],\quad V'=[V_\mathrm{prefix};V]$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 4.5 Low-rank adaptation

**Main idea.** Learn a low-rank update while freezing the base matrix.

Core relation:

$$W'=W+\frac{\alpha}{r}BA$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** LoRA works because many useful task updates can be represented well inside a small trainable subspace.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 5. LoRA Algebra

This part treats lora algebra as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Rank constraint](#5-rank-constraint) | the update matrix has rank at most r | $\mathrm{rank}(BA)\le r$ |
| [Parameter count](#5-parameter-count) | a d_out by d_in matrix gets r(d_in+d_out) trainable parameters | $P_\mathrm{LoRA}=r(d_\mathrm{in}+d_\mathrm{out})$ |
| [Scaling](#5-scaling) | the alpha over r factor controls update magnitude | $\Delta W=(\alpha/r)BA$ |
| [Merge for inference](#5-merge-for-inference) | after training, add the low-rank update into the base matrix | $W_\mathrm{merged}=W+\Delta W$ |
| [Target modules](#5-target-modules) | attention and MLP projections can receive separate adapters | $W_q,W_k,W_v,W_o,W_\mathrm{up},W_\mathrm{down}$ |

### 5.1 Rank constraint

**Main idea.** The update matrix has rank at most r.

Core relation:

$$\mathrm{rank}(BA)\le r$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 5.2 Parameter count

**Main idea.** A d_out by d_in matrix gets r(d_in+d_out) trainable parameters.

Core relation:

$$P_\mathrm{LoRA}=r(d_\mathrm{in}+d_\mathrm{out})$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 5.3 Scaling

**Main idea.** The alpha over r factor controls update magnitude.

Core relation:

$$\Delta W=(\alpha/r)BA$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 5.4 Merge for inference

**Main idea.** After training, add the low-rank update into the base matrix.

Core relation:

$$W_\mathrm{merged}=W+\Delta W$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is why LoRA can add no extra matrix multiplication at serving time after merging.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 5.5 Target modules

**Main idea.** Attention and mlp projections can receive separate adapters.

Core relation:

$$W_q,W_k,W_v,W_o,W_\mathrm{up},W_\mathrm{down}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 6. Quantized and Memory-Aware Tuning

This part treats quantized and memory-aware tuning as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Frozen quantized base](#6-frozen-quantized-base) | store base weights in low precision and train small adapters | $W_0\approx Q(W_0)$ |
| [Adapter optimizer states](#6-adapter-optimizer-states) | optimizer states are needed only for trainable adapter weights | $M_\mathrm{opt}\propto P_\mathrm{trainable}$ |
| [Activation memory remains](#6-activation-memory-remains) | PEFT reduces parameter-state memory but still backpropagates through the model | $M_\mathrm{act}$ can dominate |
| [Gradient checkpointing](#6-gradient-checkpointing) | recompute activations to fit longer sequences or larger batches | $\mathrm{memory}\downarrow,\quad\mathrm{compute}\uparrow$ |
| [Rank-memory tradeoff](#6-rankmemory-tradeoff) | higher rank improves capacity but increases trainable parameters | $P_\mathrm{LoRA}\propto r$ |

### 6.1 Frozen quantized base

**Main idea.** Store base weights in low precision and train small adapters.

Core relation:

$$W_0\approx Q(W_0)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 6.2 Adapter optimizer states

**Main idea.** Optimizer states are needed only for trainable adapter weights.

Core relation:

$$M_\mathrm{opt}\propto P_\mathrm{trainable}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 6.3 Activation memory remains

**Main idea.** Peft reduces parameter-state memory but still backpropagates through the model.

Core relation:

$$M_\mathrm{act}$ can dominate$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 6.4 Gradient checkpointing

**Main idea.** Recompute activations to fit longer sequences or larger batches.

Core relation:

$$\mathrm{memory}\downarrow,\quad\mathrm{compute}\uparrow$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 6.5 Rank-memory tradeoff

**Main idea.** Higher rank improves capacity but increases trainable parameters.

Core relation:

$$P_\mathrm{LoRA}\propto r$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 7. Preference Fine-Tuning

This part treats preference fine-tuning as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Preference pairs](#7-preference-pairs) | learn from chosen and rejected responses | $(x,y^+,y^-)$ |
| [Reward-model view](#7-rewardmodel-view) | RLHF trains or uses a reward signal for completions | $r_\phi(x,y)$ |
| [KL-regularized policy update](#7-klregularized-policy-update) | keep the tuned model near a reference model | $E[r]-\beta D_\mathrm{KL}(\pi_\theta\Vert\pi_\mathrm{ref})$ |
| [DPO loss](#7-dpo-loss) | optimize preference likelihood directly with a reference model | $-\log\sigma(\beta[(\log\pi_\theta^+-\log\pi_\theta^-)-(\log\pi_\mathrm{ref}^+-\log\pi_\mathrm{ref}^-)])$ |
| [Preference over-optimization](#7-preference-overoptimization) | too much preference pressure can reduce diversity or factuality | $r\uparrow$ does not guarantee all qualities improve |

### 7.1 Preference pairs

**Main idea.** Learn from chosen and rejected responses.

Core relation:

$$(x,y^+,y^-)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 7.2 Reward-model view

**Main idea.** Rlhf trains or uses a reward signal for completions.

Core relation:

$$r_\phi(x,y)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 7.3 KL-regularized policy update

**Main idea.** Keep the tuned model near a reference model.

Core relation:

$$E[r]-\beta D_\mathrm{KL}(\pi_\theta\Vert\pi_\mathrm{ref})$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 7.4 DPO loss

**Main idea.** Optimize preference likelihood directly with a reference model.

Core relation:

$$-\log\sigma(\beta[(\log\pi_\theta^+-\log\pi_\theta^-)-(\log\pi_\mathrm{ref}^+-\log\pi_\mathrm{ref}^-)])$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** DPO turns a preference pair into a logistic loss over relative log probabilities.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 7.5 Preference over-optimization

**Main idea.** Too much preference pressure can reduce diversity or factuality.

Core relation:

$$r\uparrow$ does not guarantee all qualities improve$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 8. Evaluation and Diagnostics

This part treats evaluation and diagnostics as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Train and validation loss](#8-train-and-validation-loss) | memorization shows up as train loss improving while validation stalls | $L_\mathrm{train}\downarrow,\quad L_\mathrm{val}\not\downarrow$ |
| [Base-task retention](#8-basetask-retention) | evaluate old skills after adaptation | $S_\mathrm{retain}(\theta)$ |
| [Task quality](#8-task-quality) | use task-specific automatic and human checks | $S_\mathrm{task}(\theta)$ |
| [Distribution shift](#8-distribution-shift) | fine-tune data should match deployment use | $p_\mathrm{train}(x,y)\approx p_\mathrm{deploy}(x,y)$ |
| [Adapter sanity checks](#8-adapter-sanity-checks) | disable the adapter to confirm measured change comes from the adapter | $f_{W+\Delta W}(x)-f_W(x)$ |

### 8.1 Train and validation loss

**Main idea.** Memorization shows up as train loss improving while validation stalls.

Core relation:

$$L_\mathrm{train}\downarrow,\quad L_\mathrm{val}\not\downarrow$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 8.2 Base-task retention

**Main idea.** Evaluate old skills after adaptation.

Core relation:

$$S_\mathrm{retain}(\theta)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 8.3 Task quality

**Main idea.** Use task-specific automatic and human checks.

Core relation:

$$S_\mathrm{task}(\theta)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 8.4 Distribution shift

**Main idea.** Fine-tune data should match deployment use.

Core relation:

$$p_\mathrm{train}(x,y)\approx p_\mathrm{deploy}(x,y)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 8.5 Adapter sanity checks

**Main idea.** Disable the adapter to confirm measured change comes from the adapter.

Core relation:

$$f_{W+\Delta W}(x)-f_W(x)$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 9. Choosing a Method

This part treats choosing a method as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [No-update methods](#9-noupdate-methods) | prompting and retrieval are cheapest when they work | $\Delta\theta=0$ |
| [PEFT methods](#9-peft-methods) | adapters and LoRA are the default when cost and deployment flexibility matter | $|\Delta\theta|\ll|\theta|$ |
| [Full tuning](#9-full-tuning) | use when the task requires broad representation movement | $|\Delta\theta|$ can be large |
| [Preference tuning](#9-preference-tuning) | use when the target is comparative behavior rather than exact demonstrations | $y^+\succ y^-$ |
| [Deployment constraints](#9-deployment-constraints) | latency, adapter routing, merging, and safety evaluation are part of the method choice | $\mathrm{quality}/\mathrm{cost}$ is the real metric |

### 9.1 No-update methods

**Main idea.** Prompting and retrieval are cheapest when they work.

Core relation:

$$\Delta\theta=0$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 9.2 PEFT methods

**Main idea.** Adapters and lora are the default when cost and deployment flexibility matter.

Core relation:

$$|\Delta\theta|\ll|\theta|$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 9.3 Full tuning

**Main idea.** Use when the task requires broad representation movement.

Core relation:

$$|\Delta\theta|$ can be large$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 9.4 Preference tuning

**Main idea.** Use when the target is comparative behavior rather than exact demonstrations.

Core relation:

$$y^+\succ y^-$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 9.5 Deployment constraints

**Main idea.** Latency, adapter routing, merging, and safety evaluation are part of the method choice.

Core relation:

$$\mathrm{quality}/\mathrm{cost}$ is the real metric$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
## 10. Implementation Checklist

This part treats implementation checklist as a measurable adaptation decision. The goal is to know what is updated, what objective is optimized, how much capacity the update has, and how to detect damage to the base model.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Masking](#10-masking) | prompt, padding, and answer masks must match the intended objective | $m_j\in\{0,1\}$ |
| [Parameter freeze audit](#10-parameter-freeze-audit) | verify only intended tensors require gradients | $\mathrm{requires\_grad}$ |
| [Learning-rate groups](#10-learningrate-groups) | base weights and adapters need different optimizer groups if both train | $\eta_\mathrm{base}\ne\eta_\mathrm{adapter}$ |
| [Reference model](#10-reference-model) | preference losses require a stable reference distribution | $\pi_\mathrm{ref}$ |
| [Ablations](#10-ablations) | compare base, prompt-only, PEFT, and full-tune when feasible | $\Delta S=S_\mathrm{method}-S_\mathrm{base}$ |

### 10.1 Masking

**Main idea.** Prompt, padding, and answer masks must match the intended objective.

Core relation:

$$m_j\in\{0,1\}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 10.2 Parameter freeze audit

**Main idea.** Verify only intended tensors require gradients.

Core relation:

$$\mathrm{requires\_grad}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** A PEFT run with the wrong tensors trainable is just an expensive surprise.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 10.3 Learning-rate groups

**Main idea.** Base weights and adapters need different optimizer groups if both train.

Core relation:

$$\eta_\mathrm{base}\ne\eta_\mathrm{adapter}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 10.4 Reference model

**Main idea.** Preference losses require a stable reference distribution.

Core relation:

$$\pi_\mathrm{ref}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.
### 10.5 Ablations

**Main idea.** Compare base, prompt-only, peft, and full-tune when feasible.

Core relation:

$$\Delta S=S_\mathrm{method}-S_\mathrm{base}$$

Fine-tuning is local learning around a pretrained solution. The base model already has useful representations, so the adaptation method decides how much freedom the update receives. Full fine-tuning gives maximum freedom. PEFT constrains the update to a small module, prompt vector, prefix, or low-rank subspace. Preference tuning changes the objective from imitation to comparative behavior.

**Worked micro-example.** A projection matrix with $d_\mathrm{in}=4096$ and $d_\mathrm{out}=4096$ has about 16.8 million weights. A rank-8 LoRA update for the same matrix has $8(4096+4096)=65,536$ trainable weights, before optimizer states. The base matrix can stay frozen while the low-rank update learns the task movement.

**Implementation check.** Print the number of trainable parameters, inspect masks, run one batch, and verify that loss changes. Then disable the adapter or reload the base model and confirm the difference is caused by the fine-tuning method.

**AI connection.** This is a concrete control knob in fine-tuning.

**Common mistake.** Do not treat fine-tuning quality as one number. Track task quality, base-skill retention, calibration, refusal or safety behavior if relevant, and deployment cost.

---

---

## 11. LoRA in Depth: Rank, Scaling, and Where the Parameters Go

### 11.1 The update is a constrained matrix, not a small model

LoRA (Hu et al., 2022) writes the adapted weight as $W=W_0+\frac{\alpha}{r}BA$, with $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times k}$, $r\ll\min(d,k)$, $W_0$ frozen.
- Example 1: $d=k=4096$, $r=16$: $BA$ has $16\times(4096+4096)=131{,}072$ trainable numbers versus $16{,}777{,}216$ in $W$ — 0.78% of the matrix.
- Example 2: the same rank on a $1024\times1024$ projection costs $32{,}768$ numbers, 3.1% — rank is a *relative* budget, so a fixed $r$ is a bigger intervention in small layers.
- Example 3: adapting query and value projections in 32 layers at $r=16$, $d=4096$ trains $32\times2\times131{,}072\approx8.4$M parameters, the classic GPT-3-scale LoRA configuration reported by Hu et al. (2022) at roughly 0.01% of 175B... scaled here to a 7B-class model.
- Non-example 1: training $A$ and $B$ but also unfreezing $W_0$ is full fine-tuning with extra steps, not LoRA.
- Non-example 2: a rank-1 update is not "1% of a LoRA"; rank is a dimension, and rank-1 can already move one direction in weight space decisively.

### 11.2 Initialization makes LoRA start at the pretrained model

Hu et al. (2022) initialize $A$ with random Gaussian entries and $B=0$, so $BA=0$ and training starts exactly at the base model's function. This matters mathematically: early gradients flow into $B$ first ($\partial L/\partial B\propto (\partial L/\partial W)A^\top$ is nonzero while $\partial L/\partial A\propto B^\top(\partial L/\partial W)$ starts at zero), so adaptation *grows* a direction instead of perturbing a random one.

### 11.3 The $\alpha/r$ scaling is a learning-rate-like knob

With update $\frac{\alpha}{r}BA$, doubling $r$ at fixed $\alpha$ halves the scale of a *randomly initialized magnitude* update; Hu et al. (2022) motivate $\alpha$ as the constant you tune first so that changing $r$ does not force a full learning-rate re-search. In practice: fix $\alpha$ (commonly $\alpha=2r$ in many recipes makes the factor 2), tune rank for capacity and learning rate for stability — conflating the two is a common silent bug (see Common Mistakes).

### 11.4 Merging and its exactness

At inference $W_{\text{merged}}=W_0+\frac{\alpha}{r}BA$ can replace $W_0$ with *zero* added latency, because the adapter is a plain matrix addition in the same space. Merging is exact in exact arithmetic; in bf16 the merged weight rounds once, which is why merging in fp32 and then casting is the safer order for quantized deployment (Section 6).

### 11.5 Which modules to adapt

Hu et al. (2022) found adapting query and value projections most effective per parameter for GPT-3; later practice often adapts all linear layers. The mathematical statement behind the empirical rule: attention projections steer *where information moves*, feed-forward layers steer *what is computed*; a task that changes style/content routing needs the former, a task that adds factual computation needs the latter. State your target-module choice as a hypothesis about the task, not as a default.

---

## 12. Adapters, Prefixes, and Prompts as Function-Space Interventions

### 12.1 Bottleneck adapters

Houlsby et al. (2019) insert small bottleneck modules (down-project, nonlinearity, up-project) after transformer sublayers in BERT and match full fine-tuning on GLUE with about 3.6% additional parameters per task. The adapter is a *residual* intervention: with the up-projection initialized near zero it starts near the identity function, the adapter analogue of LoRA's $B=0$ trick (11.2).

### 12.2 Prefix tuning changes keys and values, not weights

Li and Liang (2021) prepend trainable continuous vectors to the keys and values of every attention layer for generation tasks (GPT-2/GPT-3 class), training only about 0.1% of parameters. Formally the prefix extends the attention context with "virtual tokens" whose content is learned rather than embedded text.

### 12.3 Prompt tuning is the input-only limit

Lester et al. (2021) train only a soft prompt at the input embedding layer and showed the capacity gap to full tuning closes with model scale (their T5 experiments). The lesson generalizes: the larger and more capable the frozen base, the more of the task is *steering* (reachable by small interventions) rather than *new computation*.

### 12.4 (IA)^3 and learned rescaling

Liu et al. (2022, (IA)^3) learn vectors that rescale keys, values, and feed-forward activations elementwise — fewer parameters than LoRA with competitive few-shot results. Rescaling is a diagonal intervention: it can reweight features the base model already computes, and provably cannot create a feature direction absent from the base activations. That limitation is the cleanest illustration of this section's theme: every PEFT method is a hypothesis about which function-space directions the task needs.

---

## 13. Preference Optimization in Depth

### 13.1 From reward model to policy objective

The RLHF pipeline (Ouyang et al., 2022, InstructGPT): fit a reward model $r(x,y)$ to human preferences with the Bradley-Terry loss $-\log\sigma(r(x,y_w)-r(x,y_l))$, then optimize the policy against $r$ with a KL penalty to the reference (SFT) policy, classically with PPO (Schulman et al., 2017). The KL term is not decoration: it prices distribution shift, because the reward model is only trustworthy near the data it was fitted on.

### 13.2 DPO removes the explicit reward model

Rafailov et al. (2023, Direct Preference Optimization) use the closed-form solution of the KL-regularized objective, $\pi^*(y\mid x)\propto\pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}$, to rewrite the preference loss directly on the policy:
$L_{\text{DPO}}=-\log\sigma\big(\beta[\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}-\log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}]\big)$.
Worked number: if the policy has moved so the log-ratio difference is $0.5$ and $\beta=0.1$, the logit inside the sigmoid is $0.05$, the loss is $-\log\sigma(0.05)\approx0.668$, barely below the chance value $\ln2\approx0.693$ — DPO's implicit reward is measured in *nats of policy movement*, which is why small $\beta$ demands large movements for the same loss reduction.

### 13.3 Over-optimization is a scaling law too

Gao et al. (2023) showed reward-model over-optimization follows its own scaling regularities: as the policy moves away in KL, true (gold) reward rises, peaks, and falls while proxy reward keeps rising — Goodhart's law with a measurable KL axis. Practical consequence: report gold-eval performance against KL-from-reference, not against training steps; steps are not the conserved quantity.

### 13.4 IPO, KTO, and why variants exist

Azar et al. (2024, IPO) argue DPO can overfit deterministic preferences and replace the log-sigmoid with a squared loss toward a finite margin. Gethin/ETH Zürich's KTO (Ethayarajh et al., 2024) learns from binary desirable/undesirable labels instead of pairs. These are not fads: each changes the *preference model* assumed by the loss (13.1–13.2), and the right choice depends on what preference data you actually have.

---

## 14. Catastrophic Forgetting, Measured and Managed

### 14.1 Forgetting is distribution shift you cause yourself

Fine-tuning on task $A$ moves parameters toward $A$'s optimum; performance on earlier task $B$ degrades in proportion to how much the *shared* function directions move. Kirkpatrick et al. (2017, Elastic Weight Consolidation) penalize movement in directions with high Fisher information for $B$, making the penalty quadratic: $L=L_A+\frac{\lambda}{2}\sum_i F_i(\theta_i-\theta^*_{B,i})^2$.

### 14.2 Replay and mixing are the empirical workhorses

Mixing a fraction of general/pretraining-style data into SFT (Section 2.5's mixture weights) is replay in data form. The mixture weight is then a forgetting knob: raise it and task gain falls while base capability retention rises. Report both axes (task score, base benchmark score) for every mixture you try; a single task number hides the trade you actually made.

### 14.3 LoRA forgets less, for a structural reason

Because LoRA's update is confined to a rank-$r$ subspace per matrix and starts at zero (11.2), its function-space movement is smaller and more easily removed (unmerge) than full fine-tuning's. This is a *prior* toward retention, not immunity: a high-rank, high-LR LoRA on narrow data can still overwrite base behaviour through the layers it touches.

---

## 15. Merging Fine-Tuned Models

### 15.1 Weight averaging and model soups

Wortsman et al. (2022, model soups) average the weights of several fine-tunes of one base (different hyper-parameters/data order) and obtain a model often better than the best ingredient — possible because fine-tunes from a shared base stay in one loss basin (cf. Training-at-Scale Q4). Averaging models fine-tuned from *different* bases generally fails for the permutation reasons given there.

### 15.2 Task arithmetic

Ilharco et al. (2023) define a task vector $\tau=\theta_{\text{ft}}-\theta_{\text{base}}$ and show adding, negating, and combining task vectors edits model behaviour algebraically: subtracting a task vector can *remove* a behaviour. LoRA adapters make this concrete and cheap: an unmerged adapter *is* a stored task vector in low-rank form.

### 15.3 Interference when merging adapters

Merging adapters trained for different tasks into one weight set superposes their updates; interference grows with adapter norm and rank overlap. Mitigations used in practice: merge with coefficients < 1, orthogonalize ranks, or keep adapters separate and route (Chapter 15 Section 10's routing applied to adapters). Test the merged model on *every* ingredient task (14.2's two-axis reporting).

---

## 16. Quantized Fine-Tuning (QLoRA) Accounting

Dettmers et al. (2023, QLoRA) fine-tune a 4-bit NormalFloat-quantized frozen base with LoRA adapters and paged optimizers, matching 16-bit fine-tuning quality on their evaluations while fine-tuning a 65B model on a single 48 GB GPU. The accounting that makes it possible:
- Base weights: $65\times10^9\times0.5$ bytes $\approx32.5$ GB at 4 bits, versus 130 GB in bf16.
- Trainable state: only adapters + their Adam moments (Section 6.2), a few hundred MB at typical ranks.
- The catch from Section 6.3 remains: activations and gradients *through* the frozen base still cost memory and compute; quantization shrinks storage, not the backward pass.
Double quantization (quantizing the quantization constants) and NF4's information-theoretic block design are the two ideas that keep the 4-bit base faithful; both are "spend cleverness on the constants" moves, not new learning rules.

---

## 17. Choosing and Auditing: Worked Scenarios

### Scenario A: style adaptation for a support assistant
Data: 20k instruction pairs, base already capable. Prescription: LoRA $r=16$ on attention + MLP projections (11.5), answer-only loss (2.2), 2–3 epochs, base-capability eval in the report (14.2). Rationale: the task is steering, so a small function-space intervention (12.3) should suffice; full tuning risks forgetting for little gain.

### Scenario B: teaching a new formal notation
Data: large, systematic, unlike pretraining text. Prescription: full fine-tuning or high-rank LoRA, because new *computation* (12.4's limitation) may be required; monitor forgetting explicitly (14).

### Scenario C: aligning refusals without harming capability
Data: preference pairs. Prescription: SFT first, then DPO at modest $\beta$ with KL-to-reference tracked as the x-axis (13.2–13.3), gold eval at several KL values, stop before the over-optimization peak.

### Scenario D: ten customer-specific variants on one GPU
Prescription: one frozen (possibly 4-bit, Section 16) base, ten LoRA adapters, merged or routed per request (15.3). Storage is base + $10\times$ adapter (MBs), not $10\times$ model.

---

---

## 32. More Computed Tables

### 32.1 Win-rate confidence intervals (29.1)

| Prompts $n$ | Observed win rate | Standard error | Approx. 95% interval |
|---:|---:|---:|---:|
| 100 | 0.52 | 0.050 | (0.422, 0.618) |
| 400 | 0.52 | 0.025 | (0.471, 0.569) |
| 2500 | 0.52 | 0.010 | (0.500, 0.540) |
| 100 | 0.60 | 0.049 | (0.504, 0.696) |
| 2500 | 0.60 | 0.010 | (0.581, 0.619) |

### 32.2 Merge coefficients and expected interference (15.3, qualitative guide)

| Merge coefficient per adapter | Expected behaviour | When to try it |
|---:|---|---|
| 1.0 each | Full superposition; highest interference risk | Adapters trained on near-identical distributions |
| 0.5 each | Halved task vectors; milder interference, milder gains | Default first merge for distinct tasks |
| Task-weighted (e.g. 0.8/0.2) | Prioritizes the primary task | One production task plus an auxiliary behaviour |


Every row still requires the per-task merged evals of the Common Mistakes table; coefficients are a search space, not a guarantee.

### 32.3 Preference-movement reading for DPO runs (13.2)

Track three numbers per checkpoint: DPO loss, mean KL from reference (in nats/token, estimated on held-out prompts), and gold/human win rate (29.1). The healthy pattern is loss and KL moving together while win rate rises; the over-optimization pattern (13.3) is KL rising, proxy metrics rising, and win rate flattening then falling. A run that logs only the loss cannot distinguish these.

## 33. Ten Review Questions With Answers

### Q1. Why does $B=0$ initialization matter beyond elegance?
Because training then starts exactly at the base function (11.2), gradients select directions from a known-good point, and step-0 evaluation must reproduce the base model — a free correctness test (checklist item 8).

### Q2. A colleague tunes LoRA rank and LR together in a grid and reports the best cell. What is confounded?
Capacity ($r$) with update scale ($lpha/r$ and LR, 11.3). Fix $lpha$, tune LR at one rank, then vary rank at that LR, or the grid measures an interaction it cannot attribute.

### Q3. When would you choose prefix tuning over LoRA?
When the task is plausibly steering of attention routing and per-task storage must be minimal (12.2), accepting that input-space interventions cannot add new feed-forward computation (12.4).

### Q4. State the DPO implicit reward.
$r(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$ up to a $y$-independent constant (19.1). It is why DPO movement is measured in nats and why $eta$ rescales the loss (Problem 3).

### Q5. Why is EWC not just weight decay toward the base?
Weight decay pulls all coordinates toward zero (or the base) uniformly; EWC pulls toward the *earlier task's* weights with per-coordinate Fisher prices (14.1, Problem 8).

### Q6. What does a model soup require that arbitrary checkpoint averaging does not have?
A shared base and fine-tuning trajectories that remain in one basin (15.1); different bases break the permutation alignment the average assumes.

### Q7. Give the three-term QLoRA memory statement.
Quantized frozen weights (small), adapter + adapter-optimizer state (small), activations through the frozen base (unchanged, often dominant) (16, Problem 10).

### Q8. How do you detect reward over-optimization without a gold reward?
You cannot fully; use held-out human/model-judge win rates against KL as a proxy gold axis (13.3, 29.1) and stop when it flattens.

### Q9. Why report base benchmarks after a style SFT that 'only changes tone'?
Because tone lives in shared function directions; 'only' is a hypothesis the two-axis report tests (14.2, 29.3).

### Q10. An adapter helps task A and, merged, hurts task B. Options?
Reduce merge coefficients, orthogonalize/retrain with B's data in the mix, or keep adapters unmerged and route per request (15.3, 32.2).

## 34. The Section in One Decision Sequence

1. Decide whether the task is steering or new computation (12.4).
2. Pick the smallest intervention family that can express it (12, 11).
3. Set rank/target modules as a stated hypothesis (11.1, 11.5).
4. Fix $lpha$, tune LR, verify step-0 equals base (11.2-11.3).
5. Mask the loss to the tokens you mean to teach (2.2).
6. Choose epochs with a forgetting report at each count (26.4, 14.2).
7. If aligning preferences, log KL-from-reference as a first-class axis (32.3).
8. Evaluate on two axes and with judge-protocol hygiene (29).
9. Merge only with per-task verification, or do not merge (15.3).
10. Store base, adapters, data version, and reference checkpoint together (22).

---

## 35. A Defensible Ablation Plan for One Fine-Tuning Question

Question: "Does rank help our support-assistant LoRA?" A defensible plan, in execution order:

1. Freeze the data version, base checkpoint, target modules (q,k,v,o + MLP, 26.1), $\alpha=32$, and the two-axis evaluation (14.2) before any training.
2. Tune LR at $r=16$ over $\{5\times10^{-5},10^{-4},2\times10^{-4}\}$, 2 epochs each, because LR is the stability knob and rank is the capacity knob (11.3).
3. At the best LR, sweep $r\in\{4,16,64\}$. Each run trains for identical scored tokens (26.3), not identical wall time, so comparisons are data-matched.
4. Record for every cell: task score, base benchmark, trainable count from 18.1's arithmetic, and adapter storage from 26.2.
5. Expect, and pre-register, the interpretation: if $r=64$ beats $r=16$ on task and ties on base retention, capacity was binding; if all ranks tie, the task is steering (12.3) and the smallest adapter wins on storage.
6. Only then consider target-module ablations, one factor at a time, because module choice and rank interact (11.5) and a two-factor grid without the one-factor baselines cannot attribute its best cell.
7. Report the full grid, including losing cells. A single best-cell number is a selection, and selections without their grids are how fine-tuning folklore is manufactured.

## 36. Data Quality for SFT, Stated Mathematically

SFT minimizes cross-entropy to the empirical distribution of your demonstrations (Section 2). Three consequences follow directly:

- **The ceiling is the data's conditional distribution.** If demonstrations disagree stylistically, the optimum is a mixture of those styles; no optimizer setting recovers a consistency the data does not contain.
- **Duplicates are mixture weights.** A response pattern repeated 500 times counts 500 times in Section 2.5's mixture — deduplication is a *reweighting decision*, not hygiene theatre (cf. Training-at-Scale 22.3).
- **Length distribution is taught.** If demonstrations are systematically verbose, answer-only loss (2.2) teaches verbosity; the preference-stage length confound (29.2) often begins as an SFT data property.

## 37. When Fine-Tuning Is the Wrong Tool

| Situation | Better first tool | Why |
|---|---|---|
| Missing facts that change daily | Retrieval (Section 12 of this chapter) | Facts in weights are stale by construction; retrieval updates without training |
| Behaviour fixable by instructions | Prompting / system prompt | Zero-training intervention; test before spending gradient (12.3's steering question) |
| One-off formatting requirement | Constrained decoding | A mask enforces format exactly; fine-tuning enforces it statistically |
| Base model lacks the capability entirely | Pretraining / larger base, then fine-tune | PEFT cannot create absent features (12.4); full FT on a weak base teaches a narrow slice at best |
| Preference signal is noisy or contested | Better preference data before any optimizer | DPO/RLHF amplify the preference distribution given (28); optimizer choice cannot fix label disagreement |


---

## 18. Computed Tables for Fine-Tuning

### 18.1 LoRA trainable parameters per square matrix

| $d=k$ | Full matrix | $r=4$ | $r=16$ | $r=64$ | $r=16$ as % of full |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1,048,576 | 8,192 | 32,768 | 131,072 | 3.12% |
| 2048 | 4,194,304 | 16,384 | 65,536 | 262,144 | 1.56% |
| 4096 | 16,777,216 | 32,768 | 131,072 | 524,288 | 0.78% |
| 8192 | 67,108,864 | 65,536 | 262,144 | 1,048,576 | 0.39% |

### 18.2 DPO loss at increasing policy movement ($eta=0.1$)

Log-ratio difference $\Delta=\log\frac{\pi(y_w)}{\pi_{ref}(y_w)}-\log\frac{\pi(y_l)}{\pi_{ref}(y_l)}$ (Section 13.2).

| $\Delta$ (nats) | Sigmoid logit $eta\Delta$ | DPO loss |
|---:|---:|---:|
| 0 | 0.00 | 0.693 |
| 0.5 | 0.05 | 0.668 |
| 1 | 0.10 | 0.644 |
| 2 | 0.20 | 0.598 |
| 5 | 0.50 | 0.474 |
| 10 | 1.00 | 0.313 |
| 20 | 2.00 | 0.127 |

Chance loss is $\ln2=0.693$; even a 10-nat preference movement only reaches 0.313 at this $eta$ — the quantitative reason $eta$ is a real hyper-parameter (13.2).

### 18.3 Trainable-parameter budget by method (7B-class base, illustrative)

| Method | Trainable | Stored per task | Inference overhead |
|---|---|---|---|
| Full fine-tuning | 7B (100%) | full model copy | none after save |
| LoRA $r=16$ all-linear (11.1 style) | ~40M (~0.6%) | ~80 MB in bf16 | zero if merged (11.4) |
| Adapters (12.1) | ~3-4% typical (Houlsby et al. 2019) | small module set | extra module compute |
| Prefix tuning (12.2) | ~0.1% (Li and Liang 2021) | prefix vectors | longer effective context |
| Prompt tuning (12.3) | embeddings only | smallest | input positions consumed |


### 18.4 A forgetting-report template (14.2)

| Run | Target-task score | Base benchmark A | Base benchmark B | Mixture weight of general data |
|---|---:|---:|---:|---:|
| Base model | — | (fill) | (fill) | — |
| LoRA $r=16$, task-only data | (fill) | (fill) | (fill) | 0 |
| LoRA $r=16$, +20% general replay | (fill) | (fill) | (fill) | 0.2 |
| Full FT, task-only data | (fill) | (fill) | (fill) | 0 |


The table is deliberately left as a template: a fine-tuning claim that cannot fill its row has not measured the trade it made.

## 19. Derivations in Steps

### 19.1 DPO from the KL-regularized objective

1. Start from Section 13.1's objective: maximize $\mathbb{E}[r(x,y)]-eta D_{KL}(\pi\|\pi_{ref})$.

2. Its closed-form optimum is $\pi^*(y|x)\propto\pi_{ref}(y|x)\exp(r(x,y)/\beta)$.

3. Invert it: $r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)}+\beta\log Z(x)$, with $Z$ independent of $y$.

4. Insert this reward into the Bradley-Terry preference probability $\sigma(r(x,y_w)-r(x,y_l))$; the $Z(x)$ terms cancel.

5. Replacing $\pi^*$ by the trainable $\pi_\theta$ and taking negative log gives the DPO loss of 13.2 (Rafailov et al., 2023). No reward model was fitted at any step.

### 19.2 Why merged LoRA is exact

1. The adapted forward pass computes $Wx=W_0x+\frac{\alpha}{r}BAx$.

2. Matrix multiplication distributes over addition, so this equals $(W_0+\frac{\alpha}{r}BA)x$ for every input $x$.

3. Therefore storing $W_{merged}$ reproduces the adapted function exactly (in exact arithmetic); only floating-point rounding distinguishes merged from unmerged deployment (11.4).

## 20. Glossary

| Term | Meaning |
|---|---|
| Adapter | Bottleneck module inserted into a frozen network (12.1) |
| Bradley-Terry model | Preference probability via a reward difference sigmoid (13.1) |
| Catastrophic forgetting | Loss of earlier-task performance after later training (14) |
| DPO | Direct Preference Optimization, reward-model-free preference loss (13.2) |
| EWC | Fisher-weighted penalty anchoring earlier-task weights (14.1) |
| LoRA | Low-rank additive weight adaptation (11) |
| Model soup | Weight average of fine-tunes from one base (15.1) |
| PEFT | Parameter-efficient fine-tuning, the family of Section 4/12 |
| Prefix tuning | Trainable key/value prefixes per layer (12.2) |
| QLoRA | LoRA on a 4-bit quantized frozen base (16) |
| Task vector | Fine-tuned minus base weights (15.2) |

---

## 21. Ten Worked Problems With Solutions

### Problem 1: LoRA parameter count
A 4096x4096 projection, $r=8$: trainable $=8\times(4096+4096)=65{,}536$, versus $16{,}777{,}216$ full — 0.39%. If this feels "too small to matter", recall Problem 1's rank is a *direction budget*: 8 directions in a 4096-dimensional space, chosen by gradient descent, starting from an already-good $W_0$ (11.2).

### Problem 2: merging by hand
$W_0=\begin{pmatrix}1&0\\0&1\end{pmatrix}$, $B=(1,2)^\top$, $A=(0.5,-1)$, $\alpha/r=1$. Then $BA=\begin{pmatrix}0.5&-1\\1&-2\end{pmatrix}$ and $W_{\text{merged}}=\begin{pmatrix}1.5&-1\\1&-1\end{pmatrix}$. Check with $x=(1,1)$: adapted path gives $W_0x+BAx=(1,1)+(-0.5,-1)=(0.5,0)$; merged gives $(1.5-1,1-1)=(0.5,0)$. Equal, as 19.2 requires.

### Problem 3: DPO loss at a glance
Policy log-ratios: chosen $+0.3$, rejected $-0.2$, so $\Delta=0.5$ nats. At $\beta=0.1$: loss $-\log\sigma(0.05)=0.668$. At $\beta=1$: $-\log\sigma(0.5)=0.474$. Same policy movement, very different loss — never compare DPO losses across $\beta$ values.

### Problem 4: answer-only loss
Token NLLs: prompt $(0.1,0.2,0.1)$, answer $(1.0,1.5)$. Answer-only loss $=1.25$. Full-sequence mean $=0.58$ — a lower, *misleading* number that credits the model for predicting the user's prompt (2.2).

### Problem 5: forgetting arithmetic
Base benchmark falls from 70 to 62 after task-only SFT, task score rises 40 to 85. Adding 20% replay: task 82, base 68. The replay row trades 3 task points for 6 base points (14.2's template). Which row is "better" is a deployment question; reporting only the first row's task number conceals that a choice existed.

### Problem 6: adapter storage for 50 tasks
LoRA $r=16$ adapters at ~80 MB each (18.3): 50 tasks = 4 GB of adapters + one base. Full fine-tunes: 50 full model copies. At 7B in bf16 that is 700 GB versus 4 GB + 14 GB shared — the storage argument for PEFT in multi-task serving (Scenario D).

### Problem 7: rank as a relative budget
$r=16$ on a $1024\times1024$ layer (18.1): $32{,}768/1{,}048{,}576=3.1\%$. The same rank on $8192\times8192$: 0.39%. A single global rank is therefore a *different* intervention in different layers; per-layer relative budgets are the fairer ablation.

### Problem 8: EWC penalty size
Fisher $F_i=100$ for a weight important to task B, $\lambda=1$: moving it by 0.05 costs $\frac12\times100\times0.0025=0.125$ added loss (14.1). A weight with $F_i=0.01$ pays $1.25\times10^{-5}$ for the same move — EWC prices movement by measured importance, not uniformly like weight decay.

### Problem 9: task-vector negation
If $\tau=\theta_{\text{ft}}-\theta_{\text{base}}$ encodes a behaviour, $\theta_{\text{base}}-\tau$ should suppress it (Ilharco et al., 2023). Verify on a held-out probe of that behaviour before believing the algebra; task vectors compose approximately, not exactly (15).

### Problem 10: QLoRA memory sanity
65B base at 4 bits: ~32.5 GB weights (16). Adapters + optimizer: order 1 GB. Activations for your batch/sequence: the remaining variable, and the one that decides checkpointing (6.3). A "fits on 48 GB" claim is a claim about all three terms, not the first alone.

---

## 22. Pre-Flight Checklist for a Fine-Tuning Run

Data and objective:
1. Answer-only masking verified on one printed example (2.2, 10.1).
2. Mixture weights written down, including any replay fraction (2.5, 14.2).
3. Preference data provenance stated: pairs, binary labels, or reward scores (13).
4. A held-out set the training pipeline cannot touch, including via packing.

Method:
5. Method chosen as a function-space hypothesis (12.4), not a habit.
6. Target modules listed with a reason per group (11.5).
7. Rank, $\alpha$, and LR each assigned one job; $\alpha$ fixed while LR is tuned (11.3).
8. Initialization check: adapted model at step 0 reproduces base outputs (11.2).

Optimization:
9. LR groups separated for adapters vs any unfrozen base weights (10.3).
10. Epoch count justified against repetition effects; early-stop metric named.
11. Reference model for KL/DPO is the intended SFT checkpoint, versioned (13).

Evaluation:
12. Target-task metric and at least one base-capability metric both scheduled (14.2).
13. For preference runs: gold eval plotted against KL-from-reference (13.3).
14. Merge plan tested on every ingredient task if adapters will be merged (15.3).
15. A rollback: the un-adapted base and every adapter stored separately (15.2).

## 23. Historical Line

| Year | Work | Idea used here |
|---:|---|---|
| 2017 | Schulman et al., PPO | Policy optimization in the RLHF stage (13.1) |
| 2017 | Kirkpatrick et al., EWC | Fisher-priced forgetting penalty (14.1) |
| 2019 | Houlsby et al., adapter modules | Bottleneck PEFT for BERT (12.1) |
| 2021 | Li and Liang, prefix tuning | Trainable key/value prefixes (12.2) |
| 2021 | Lester et al., prompt tuning | Input-only soft prompts scale (12.3) |
| 2022 | Hu et al., LoRA | Low-rank adaptation, $B=0$ init, $\alpha/r$ scaling (11) |
| 2022 | Liu et al., (IA)^3 | Learned activation rescaling (12.4) |
| 2022 | Ouyang et al., InstructGPT | SFT + reward model + PPO pipeline (13.1) |
| 2022 | Wortsman et al., model soups | Averaging fine-tunes of one base (15.1) |
| 2023 | Rafailov et al., DPO | Reward-model-free preference loss (13.2) |
| 2023 | Ilharco et al., task arithmetic | Weight-space task vectors (15.2) |
| 2023 | Dettmers et al., QLoRA | 4-bit frozen base + LoRA (16) |
| 2023 | Gao et al., reward over-optimization scaling | Gold vs proxy reward in KL (13.3) |
| 2024 | Azar et al., IPO; Ethayarajh et al., KTO | Preference-loss variants (13.4) |

## 24. Formula Summary

| Quantity | Formula |
|---|---|
| LoRA update | $\Delta W=\frac{\alpha}{r}BA$ |
| LoRA parameters per matrix | $r(d+k)$ |
| Merged weight | $W_0+\frac{\alpha}{r}BA$ |
| SFT loss | masked mean of $-\log\pi_\theta(y_t\mid x,y_{<t})$ over answer tokens |
| Bradley-Terry preference | $\sigma(r(x,y_w)-r(x,y_l))$ |
| KL-regularized optimum | $\pi^*\propto\pi_{\text{ref}}e^{r/\beta}$ |
| DPO loss | $-\log\sigma(\beta[\Delta\log\pi_\theta-\Delta\log\pi_{\text{ref}}])$ |
| EWC penalty | $\frac{\lambda}{2}\sum_i F_i(\theta_i-\theta^*_i)^2$ |
| Task vector | $\tau=\theta_{\text{ft}}-\theta_{\text{base}}$ |
| 4-bit weight storage | $\approx0.5$ bytes per parameter |

## 25. Final Self-Check

1. Can you compute a LoRA parameter count and its percentage for any layer (Problem 1)?
2. Can you merge an adapter by hand and prove merged equals unmerged (19.2)?
3. Can you derive DPO from the KL-regularized objective in five steps (19.1)?
4. Can you state DPO loss at a given movement and $\beta$, and why cross-$\beta$ comparisons fail (Problem 3)?
5. Can you report a fine-tune on two axes — task gain and base retention (14.2)?
6. Can you price a weight's movement under EWC from its Fisher value (Problem 8)?
7. Can you give the three-term QLoRA memory accounting without omitting activations (Problem 10)?


---

## 26. Budget Worksheets for a 7B-Class Model

Assumptions stated for rescaling: hidden size 4096, 32 layers, MLP intermediate 11008, attention projections square 4096x4096 (a LLaMA-7B-style geometry; the method, not the exact model, is the point).

### 26.1 LoRA parameters by target-module choice at $r=16$

| Target modules | Matrices per layer | Params per layer | 32 layers total | % of 7B |
|---|---:|---:|---:|---:|
| q, v only | 2 | 262,144 | 8,388,608 | 0.12% |
| q, k, v, o | 4 | 524,288 | 16,777,216 | 0.24% |
| q,k,v,o + MLP up/gate/down | 7 | 1,249,280 | 39,976,960 | 0.57% |

### 26.2 Adapter storage in bf16 (2 bytes per trainable parameter)

| Configuration (26.1) | Trainable | bf16 storage | With Adam moments in fp32 (+8 B/param) during training |
|---|---:|---:|---:|
| q, v only | 8,388,608 | 17 MB | 84 MB |
| q, k, v, o | 16,777,216 | 34 MB | 168 MB |
| q,k,v,o + MLP up/gate/down | 39,976,960 | 80 MB | 400 MB |

Storage is why Scenario D (ten tasks, one base) is routine with adapters and painful with full copies (Problem 6).

### 26.3 SFT token and step accounting

Dataset: 50,000 instruction examples, average answer length 180 tokens, average prompt 120 tokens (masked, Section 2.2).

| Quantity | Value | Arithmetic |
|---|---:|---|
| Answer tokens per epoch | 9.0M | 50,000 x 180 |
| Prompt tokens processed (not scored) per epoch | 6.0M | 50,000 x 120 |
| Steps per epoch at batch 128 examples | 391 | 50,000 / 128 |
| Answer tokens per optimizer step | ~23,000 | 128 x 180 |
| 3 epochs | 27M scored tokens, 1,172 steps | multiply by 3 |


The scored-token row, not the example count, is what connects SFT cost to the pretraining accounting of Training at Scale: fine-tuning is the same $6ND$-style arithmetic with $D$ equal to scored (plus processed prompt) tokens and $N$ replaced by the forward/backward cost through a mostly frozen base.

### 26.4 Epochs and repetition for small SFT sets

| Epochs on 50k examples | Scored tokens | Risk that rises | Mitigation |
|---:|---:|---|---|
| 1 | 9M | under-adaptation | check task eval before adding epochs |
| 2-3 | 18-27M | mild memorization of phrasing | hold out examples; watch base benchmarks (14.2) |
| 5+ | 45M+ | verbatim memorization, forgetting | replay mixing (14.2), lower LR, early stop |


These are heuristics for review, not laws: the deciding evidence is the two-axis report of Section 14.2 at each epoch count.

## 27. What Changes Between SFT, RLHF, and DPO: One Table

| | SFT | RLHF (PPO) | DPO |
|---|---|---|---|
| Training signal | gold responses | reward model + preference data | preference pairs directly |
| Extra models in memory | none | reward model + reference policy | reference policy |
| Key hyper-parameter beyond LR | mixture weights (2.5) | KL coefficient, PPO clip | $eta$ (13.2) |
| Main failure mode | forgetting, exposure to gold only | reward hacking (13.3) | over-confident pair fitting (13.4) |
| Movement measured in | epochs/tokens | KL from reference | implicit reward in nats (13.2) |
| Canonical reference | Ouyang et al. 2022 (SFT stage) | Ouyang et al. 2022; Schulman et al. 2017 | Rafailov et al. 2023 |

---

## 28. The InstructGPT Pipeline as an Accounting Exercise

Ouyang et al. (2022) is the canonical worked example of Section 13.1, and its three stages map onto this chapter's methods one-to-one.

Stage 1, SFT (Section 2): labeler-written demonstrations train a GPT-3 model with the answer-only-style objective of 2.2 for a small number of epochs.
Stage 2, reward modelling (13.1): labelers rank multiple sampled outputs per prompt; the Bradley-Terry loss fits a scalar reward to those rankings, with rankings expanded into pairwise comparisons.
Stage 3, PPO (13.1): the policy is optimized against the reward model with a per-token KL penalty to the SFT reference, plus a pretraining-data mix term to limit capability regression — replay (14.2) appearing inside the RL stage.

Their headline result, that a 1.3B InstructGPT was preferred to the 175B base GPT-3, is the strongest canonical evidence for this chapter's central claim: after pretraining, *adaptation quality is not primarily a parameter-count question*. It is also a caution in the other direction: the result depended on labeler selection and instructions, i.e. on $p_{\text{data}}$ for stages 1–2 — the method amplifies the preference data it is given, including its biases.

## 29. Evaluating Preference-Tuned Models Without Fooling Yourself

### 29.1 Win rate needs a judge protocol

A win rate is $\Pr(\text{judge prefers }A)$ over a prompt distribution. State: the prompt set, the judge (human pool or named model), position-order controls (judges prefer the first or longer answer at measurable rates), and a confidence interval $\sqrt{\hat q(1-\hat q)/n}$ as in Chapter 15 Section 05's Monte Carlo discussion. A 52% win rate on 100 prompts is within noise of a coin flip; on 2,500 prompts it is not.

### 29.2 Length is a confound, always

Preference judges (human and model) tend to reward longer answers. Report win rate *and* mean length for both systems, or a length-controlled variant; otherwise a "preference improvement" may be a verbosity update purchased with tokens. This is 13.3's Goodhart warning at the evaluation layer: the metric being maximized (judge approval) diverges from the goal (helpfulness) along a measurable axis.

### 29.3 Capability evals are the second axis

Every preference run should republish the base-capability line of 14.2's template. Alignment taxes are real and sometimes worth paying; an unreported tax is not a saving, it is an omission.

## 30. Debugging Fine-Tuning Runs

| Symptom | First check | Section |
|---|---|---|
| Task eval flat, training loss falls | Loss is being computed on prompt tokens or the wrong split | 2.2, Problem 4 |
| Output degenerates into repetition | LR too high for adapters; check $\alpha/r$ scaling | 11.3 |
| Base benchmarks collapse after 1 epoch | LR/epochs too aggressive; no replay | 14.2 |
| DPO loss falls, human eval worsens | Over-optimization in KL; inspect gold-vs-KL curve | 13.3 |
| Merged model worse than unmerged adapter | bf16 merge rounding or coefficient error | 11.4, 15.3 |
| One task's adapter hurts another after merging | Rank interference; lower merge coefficients or route instead | 15.3 |
| OOM despite frozen base | Activations, not weights, dominate | 6.3, Problem 10 |
| Step-0 outputs differ from base | Adapter initialization is not zero | 11.2 |

## 31. Notation for This Chapter's Expansion

| Symbol | Meaning |
|---|---|
| $W_0$ | Frozen pretrained weight |
| $A, B$ | LoRA down/up factors |
| $r$ | LoRA rank |
| $\alpha$ | LoRA scaling numerator in $\alpha/r$ |
| $\pi_\theta, \pi_{\text{ref}}$ | Trainable and reference policies |
| $r(x,y)$ | Reward function (not to confuse with LoRA rank; context decides) |
| $\beta$ | KL/preference temperature in RLHF and DPO |
| $y_w, y_l$ | Preferred and dispreferred responses |
| $F_i$ | Fisher information of parameter $i$ |
| $\tau$ | Task vector |

---

## 38. Evaluation Cadence Worksheet

| Checkpoint | Task eval | Base eval | Win-rate vs previous (29.1) | KL from reference (32.3) | Decision it informs |
|---|---|---|---|---|---|
| Step 0 (= base) | yes | yes | — | 0 | step-0 identity test (11.2) |
| 0.5 epoch | yes | no | optional | yes | early LR sanity |
| 1 epoch | yes | yes | yes | yes | continue vs stop |
| 2 epochs | yes | yes | yes | yes | forgetting check (14.2) |
| 3 epochs | yes | yes | yes | yes | diminishing-returns check (26.4) |
| Each merge candidate (15.3) | per-task | yes | per-task | n/a | merge or route |


The cadence is the practical form of the two-axis rule: base capability is measured often enough that forgetting is caught while it is still cheap to fix with replay or a lower LR.

## 39. Scaling Factor Table for LoRA Recipes

| $r$ | Common $lpha=2r$ choice | $lpha/r$ | Interpretation |
|---:|---:|---:|---|

| 1 | 2 | 2.0 | scale held constant while capacity grows (11.3) |
| 2 | 4 | 2.0 | scale held constant while capacity grows (11.3) |
| 4 | 8 | 2.0 | scale held constant while capacity grows (11.3) |
| 8 | 16 | 2.0 | scale held constant while capacity grows (11.3) |
| 16 | 32 | 2.0 | scale held constant while capacity grows (11.3) |
| 32 | 64 | 2.0 | scale held constant while capacity grows (11.3) |
| 64 | 128 | 2.0 | scale held constant while capacity grows (11.3) |
| 128 | 256 | 2.0 | scale held constant while capacity grows (11.3) |

Holding $lpha/r$ constant across the rank sweep is what makes Section 35's ablation attribute differences to capacity rather than scale.

## 40. Summary: Claims and Their Evidence Standard

| Claim you might make | Evidence this chapter requires |
|---|---|
| 'LoRA matched full tuning' | same data, same scored tokens, two-axis eval (14.2, 26.3) |
| 'DPO aligned the model' | win rate with judge protocol + KL axis (29.1, 32.3) |
| 'QLoRA saved memory' | three-term accounting including activations (Problem 10) |
| 'Merging helped' | per-task merged evals against unmerged adapters (15.3) |
| 'No forgetting' | base benchmark line, not an impression (14.2) |
| 'Rank 64 is better' | LR tuned first, tokens matched, full grid reported (35) |

---

## 41. One SFT Example, Token by Token

Prompt (masked, label = -100 in the usual convention):

```text
[INST] Name a primary color. [/INST]
```

Response (scored):

```text
Red.<EOS>
```

Training row construction, step by step:

1. Tokenize prompt and response separately so the boundary is known exactly.
2. Input ids are prompt ids followed by response ids.
3. Labels are -100 on every prompt position (Section 2.2).
4. Labels equal the response ids on response positions, shifted by the model's internal label shift (predict next token).
5. The EOS position is scored: the model must learn to *stop*, and an unscored EOS is the most common cause of rambling generations after SFT.
6. The loss for this row is the mean NLL over exactly the response-plus-EOS positions (Problem 4's arithmetic on real ids).
7. In a packed batch, this row's tokens must not attend into the previous row (document masking, Training-at-Scale 22.2).
8. At evaluation, generation stops at the same EOS id the loss scored in step 5 — train/test consistency of the stop token is a correctness property, not a detail.

## 42. A Learning-Rate Finder for Adapters, in Words and Numbers

Adapters tolerate larger learning rates than full fine-tuning because the base is frozen and the update is low-rank (11), but "larger" must still be found, not guessed.

Procedure:
1. Take a 5% slice of the SFT data, fixed across all trials below.
2. Train $r=16$, $\alpha=32$ adapters for 200 steps at each LR in $\{5\times10^{-5},10^{-4},2\times10^{-4},4\times10^{-4}\}$.
3. Log training loss *and* a 200-example task eval at step 200 for each (task eval, because SFT loss on your own demonstrations rewards imitation including its noise, 36).
4. Also log a base-capability spot check at the two largest LRs; adapter runs can damage base behaviour through high-LR layers before task eval notices (14).
5. Choose the largest LR whose step-200 task eval is within noise of the best and whose base spot check is unchanged; then run Section 35's rank sweep at that LR.
6. If the largest LR wins on task and already dents the base check, the binding constraint is forgetting, not optimization — answer with replay (14.2), not with more LR.

Worked reading of a hypothetical outcome (illustrative numbers, stated as such): losses 1.9/1.6/1.5/1.5 across the four LRs, task evals 61/68/70/70, base spot check unchanged until $4\times10^{-4}$ where it drops 3 points. Decision: $2\times10^{-4}$ — the last LR that buys task performance without spending base capability.

## 43. Storage and Serving Patterns for Adapters

| Pattern | What is stored | What is loaded per request | Use when |
|---|---|---|---|
| Merged single task | one merged model (11.4) | the model | one production behaviour |
| Adapter library, merged on deploy | base + adapters | merge once at deploy time | few tasks, static assignment |
| Adapter library, hot-swapped | base + adapters | base resident, adapter swapped in | many tasks, changing mix |
| Routed adapters | base + adapters + router | router picks adapter(s) | per-request task identity (15.3) |

The pattern choice is an inference-systems decision (Chapter 15 Section 13) made *possible* by the mathematics of 11.4 and 15.2: because an adapter is an additive, removable weight delta, all four patterns represent the same trained object.

---

## 44. Five Failure Scenarios and Their Diagnoses

### Scenario A: "The adapter learned nothing"
Symptoms: task eval at base level after two epochs, training loss decreasing slowly.
Diagnosis path: first verify the loss is scored on answer tokens at all (Problem 4); a mask that zeros the response makes the *reported* loss a prompt-modelling number while the adapter receives almost no task gradient.
Second, check that adapter parameters are actually in the optimizer's parameter groups (a freeze audit, 10.2): a LoRA module attached after optimizer construction trains nothing while logging normally.
Fix order: mask, parameter groups, then — only then — LR and rank.

### Scenario B: "It answers everything in the training set's voice, including facts it invents"
Symptoms: perfect style transfer, worse factuality.
Diagnosis: SFT taught the conditional distribution of the demonstrations (36), including their confidence; style and certainty were entangled in the data.
Fix: preference stage with honesty-weighted pairs (13), and capability/hallucination evals on the second axis (29.3). More SFT epochs intensify the problem they appear to solve.

### Scenario C: "DPO training loss went to 0.2 and the model got worse"
Symptoms: low preference loss, falling human win rate, very long outputs.
Diagnosis: over-optimization along KL (13.3) plus the length confound (29.2): the implicit reward (19.1) can be increased by verbosity the judge initially rewards.
Fix: early-stop on win rate versus KL (32.3), reduce $\beta$-driven movement, add length controls to judging.

### Scenario D: "The merged model is worse than either adapter alone"
Symptoms: two good task adapters, one mediocre merged model.
Diagnosis: interference in superposition (15.3), possibly aggravated by merging in bf16 (11.4).
Fix: merge in fp32, sweep coefficients from 32.2's table with per-task evals, and if interference persists, serve unmerged with routing (43).

### Scenario E: "Base model forgot how to code after dialogue SFT"
Symptoms: chat eval up, code benchmark down double digits.
Diagnosis: catastrophic forgetting through shared directions (14), accelerated by multiple epochs at adapter LRs chosen for the chat task alone (42, step 4).
Fix: replay with code data in the mixture (14.2), lower LR, fewer epochs, and put the code benchmark on the cadence table (38) so the trade is seen at epoch 1, not after deployment.

---

## 45. Implementation Audit Protocol

Run in order; each step isolates one layer of the fine-tuning stack.

1. Print one batch with prompt tokens, response tokens, and labels side by side (41).
2. Confirm every prompt label is the mask value and every response label, including EOS, is scored.
3. Confirm the label shift: position $i$ is scored for predicting token $i+1$, not token $i$.
4. List every trainable parameter name and count; compare the total to 18.1/26.1's prediction.
5. List every frozen parameter name; confirm the base and embeddings match the intended freeze policy (10.2).
6. Confirm $B=0$ (or adapter near-identity) by checking step-0 outputs against the base on 20 prompts (11.2).
7. Confirm adapter parameters are inside the optimizer's parameter groups (Scenario A).
8. Log LR per parameter group; adapters and any unfrozen base weights should not silently share one (10.3).
9. Verify $\alpha$ and $r$ in the saved config equal the values the scaling arithmetic assumed (11.3).
10. Evaluate the base checkpoint on the task and base benchmarks to establish both axes' origins (14.2).
11. Train 200 steps on a data slice and confirm loss falls without base spot-check damage (42).
12. Save a checkpoint and reload it; confirm outputs are unchanged after reload (merge/save correctness, 11.4).
13. If merging: merge in fp32, then eval per task (15.3).
14. If preference tuning: log an estimate of KL from reference from the first checkpoint onward (32.3).
15. Record data version, base hash, adapter config, and reference checkpoint together (22, item 15).

## 46. Equations, Read Aloud

- $W=W_0+\frac{\alpha}{r}BA$ says: the model you deploy is the pretrained model plus a low-rank opinion about your task.
- $-\log\pi_\theta(y\mid x)$ averaged over answer tokens says: SFT pays only for the words it asked the model to produce.
- $\sigma(r_w-r_l)$ says: a preference is evidence about a *difference*, never about absolute quality.
- $\beta\log\frac{\pi_\theta}{\pi_{\text{ref}}}$ says: DPO's reward is movement itself, priced by $\beta$.
- $\frac{\lambda}{2}F_i(\theta_i-\theta_i^*)^2$ says: forgetting insurance is priced per parameter by measured importance.
- $\theta_{\text{base}}+\sum_k c_k\tau_k$ says: merged behaviours are a weighted sum of edits, with all the interference linear algebra implies.

Reading the equations aloud in words is a reliable test: a fine-tuning configuration whose equations you cannot paraphrase is a configuration you are operating, not understanding.

---

## 47. Expanded Glossary

| Term | Meaning |
|---|---|
| Alignment tax | Capability lost in exchange for preference alignment (29.3) |
| Answer-only loss | SFT loss masked to response tokens (2.2) |
| Bias-corrected moments | Adam estimates divided by $1-\beta^t$ (Training at Scale) |
| Checkpoint cadence | Scheduled points at which both eval axes are measured (38) |
| Data mixture weight | Fraction of training tokens from a source (2.5) |
| Function-space intervention | A PEFT method viewed by which functions it can reach (12.4) |
| Gold reward | True preference quality, as opposed to a proxy reward model (13.3) |
| Hot-swapped adapter | Adapter loaded per request over a resident base (43) |
| Implicit reward | DPO's $\beta\log(\pi_\theta/\pi_{\text{ref}})$ (19.1) |
| Judge protocol | Prompt set, judge identity, and order controls behind a win rate (29.1) |
| KL budget | Allowed divergence from reference during preference tuning (32.3) |
| Length confound | Judge preference for longer answers masquerading as quality (29.2) |
| Low-rank opinion | Informal reading of $\frac{\alpha}{r}BA$ (46) |
| Merge coefficient | Scalar weight of a task vector in a merge (32.2) |
| Over-optimization | Proxy reward rising while gold reward falls (13.3) |
| Parameter group | Optimizer subset with its own LR (45, step 8) |
| Replay | Mixing earlier-distribution data to limit forgetting (14.2) |
| Reference policy | Frozen SFT model anchoring KL in RLHF/DPO (13) |
| Routed adapters | Per-request adapter selection (43) |
| Scored tokens | Tokens that enter the loss (26.3) |
| Steering task | Task solvable by redirecting existing capability (12.3) |
| Step-0 identity | Adapted model initially equals the base (11.2) |
| Two-axis report | Task score together with base retention (14.2) |

## 48. Closing Decision Summary

1. Steering or new computation? (12.4)
2. Smallest intervention that can express it? (11-12)
3. Loss scored on exactly the intended tokens? (41, 45)
4. Rank, alpha, and LR each doing one job? (11.3)
5. Both evaluation axes measured on a cadence? (38)
6. Preference movement tracked in KL, not steps? (32.3)
7. Merge verified per task, or not merged at all? (15.3)
8. Base, adapter, data, and reference stored as one versioned object? (22)

A fine-tuning run that answers these eight is a method. One that cannot is a hope with a learning rate.

---

## 49. Method Comparison, Extended

| Method | Trainable object | Starts at base function? | Removable after training? | Typical failure |
|---|---|---|---|---|
| Full fine-tuning | all weights | yes (it is the base) | no | forgetting (14) |
| LoRA | $A,B$ per targeted matrix | yes, $B=0$ (11.2) | yes, unmerge | rank/LR confounding (11.3) |
| Adapters | bottleneck modules | near, by initialization (12.1) | yes, bypass modules | inference module overhead |
| Prefix tuning | key/value prefixes | base weights untouched | yes, drop prefix | capacity limits on hard tasks |
| Prompt tuning | input soft prompt | base untouched | yes | weakest on small bases (12.3) |
| (IA)^3 | rescaling vectors | yes at unit scale | yes | cannot create new features (12.4) |
| QLoRA | LoRA on 4-bit base | yes | yes | activation memory surprise (16) |
| SFT (any method) | policy toward demonstrations | n/a | by checkpoint choice | data ceiling (36) |
| RLHF/PPO | policy vs reward + KL | at SFT reference | by checkpoint choice | reward hacking (13.3) |
| DPO | policy vs reference log-ratios | at reference | by checkpoint choice | pair overfitting (13.4) |

## 50. What to Log, and the Section That Explains Why

| Logged quantity | Cadence | Why |
|---|---|---|
| Training loss (answer tokens only) | every step, smoothed | optimization health (2.2) |
| Gradient norm | every step | LR and spike early warning (42) |
| LR per parameter group | on change | group mix-ups are silent (45) |
| Task eval | per 38's cadence | the point of the run |
| Base benchmark | per 38's cadence | forgetting axis (14.2) |
| KL from reference (preference runs) | every checkpoint | over-optimization axis (32.3) |
| Mean output length | every eval | length confound (29.2) |
| Adapter norm $\|BA\|$ | every checkpoint | merge and interference forecasting (15.3) |

---

## 51. Ten Review Sentences

- Every number claimed for a fine-tune should be traceable to a row in the tables of Sections 18, 26, 32, or 38.
- A fine-tuning result without its base-model row is an advertisement, not a measurement.
- A preference result without its KL axis is a trajectory without a map.
- An adapter result without its rank, alpha, target modules, and LR is not reproducible.
- A merged model without per-task evals is a hypothesis, not a release.
- A QLoRA result without activation accounting is a storage claim, not a memory claim.
- An SFT result without its data mixture is an objective without a definition.
- A win rate without its judge protocol is a percentage without a population.
- A forgetting claim without a base benchmark is an absence of evidence presented as evidence of absence.
- These sentences are the review checklist in prose form; Sections 22 and 45 are the same checklist as procedures.

## Common Mistakes

| Mistake | Why Wrong | Fix |
|---|---|---|
| Training loss on prompt tokens unintentionally | Teaches modelling the user, wastes capacity | Answer-only mask (2.2); audit the mask (10.1) |
| LoRA $\alpha$ and LR both changed between runs | $\alpha/r$ is itself a scale (11.3); effects confound | Fix $\alpha$, tune LR, then rank |
| $B$ initialized randomly nonzero | Model starts away from the base function | $B=0$ so $BA=0$ at step 0 (11.2) |
| Evaluating only the target task | Forgetting is invisible (14) | Report base-capability scores alongside task scores |
| DPO with $\beta$ copied from another model's recipe | Implicit reward is in nats of *your* policy movement (13.2) | Sweep $\beta$; track KL-from-reference |
| Chasing proxy reward for many steps | Gold reward peaks then falls (Gao et al. 2023) | Early-stop on gold eval vs KL, not steps |
| Merging adapters without per-task evals | Interference is silent (15.3) | Eval merged model on every ingredient task |
| Assuming QLoRA removes activation memory | Only weight storage shrinks (16) | Budget activations/checkpointing separately (6.3) |
| Full fine-tuning a capable base for a style task | Pays forgetting for steering (12.3, 14) | Try LoRA/prefix first; escalate on evidence |
| Reporting adapter parameter % of the wrong denominator | Rank cost is relative per matrix (11.1) | State per-matrix and total counts |

## Practice Exercises

1. Compute answer-only SFT loss using a binary mask.
2. Compute KL shift between base and tuned next-token distributions.
3. Count LoRA trainable parameters for a projection matrix.
4. Verify the shape of a LoRA update.
5. Approximate a matrix update with truncated SVD.
6. Count adapter bottleneck parameters.
7. Count prefix-tuning parameters for layers and heads.
8. Compute DPO loss for one preference pair.
9. Build a forgetting versus task-quality scorecard.
10. Write a parameter-freeze and masking checklist.

## Why This Matters for AI

| Concept | AI Impact |
|---|---|
| LoRA (Hu et al., 2022) | Cheap per-task adaptation of GPT-3-class models at ~0.01-1% trainable parameters |
| DPO (Rafailov et al., 2023) | Preference alignment without a fitted reward model or PPO loop |
| QLoRA (Dettmers et al., 2023) | 65B fine-tuning on a single 48 GB GPU at 4-bit base weights |
| Model soups / task vectors (Wortsman et al., 2022; Ilharco et al., 2023) | Composing and removing behaviours in weight space |
| Reward over-optimization (Gao et al., 2023) | KL-budgeted alignment instead of step-budgeted alignment |

Most applied LLM work is not pretraining from scratch. It is adaptation: making a capable base model behave correctly in a domain, workflow, policy regime, or product setting. Fine-tuning math keeps that adaptation honest. It tells you whether you are training the intended tokens, moving the intended parameters, using enough rank, preserving base capability, and measuring the right behavior.

## Bridge to Scaling Laws

The next section studies how loss changes with model size, data size, and compute. Fine-tuning adds another axis: adaptation capacity. A small low-rank update may be enough for format and style, while deeper domain shifts may require more data, rank, layers, or full-model movement.

## References

- Jeremy Howard and Sebastian Ruder, "Universal Language Model Fine-tuning for Text Classification", 2018: https://arxiv.org/abs/1801.06146
- Neil Houlsby et al., "Parameter-Efficient Transfer Learning for NLP", 2019: https://proceedings.mlr.press/v97/houlsby19a.html
- Xiang Lisa Li and Percy Liang, "Prefix-Tuning: Optimizing Continuous Prompts for Generation", 2021: https://arxiv.org/abs/2101.00190
- Edward J. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", 2021: https://arxiv.org/abs/2106.09685
- Long Ouyang et al., "Training language models to follow instructions with human feedback", 2022: https://arxiv.org/abs/2203.02155
- Tim Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023: https://arxiv.org/abs/2305.14314
- Rafael Rafailov et al., "Direct Preference Optimization: Your Language Model is Secretly a Reward Model", 2023: https://arxiv.org/abs/2305.18290
- John Schulman et al., "Proximal Policy Optimization Algorithms", 2017: https://arxiv.org/abs/1707.06347

---

## 52. Consolidated References for the Expansion

- Hu et al., LoRA, ICLR 2022, arXiv:2106.09685.
- Rafailov et al., Direct Preference Optimization, NeurIPS 2023, arXiv:2305.18290.
- Dettmers et al., QLoRA, NeurIPS 2023, arXiv:2305.14314.
- Ouyang et al., InstructGPT, NeurIPS 2022, arXiv:2203.02155.
- Gao, Schulman, and Hilton, Scaling Laws for Reward Model Overoptimization, ICML 2023, arXiv:2210.10760.
- Houlsby et al., Parameter-Efficient Transfer Learning for NLP, ICML 2019.
- Li and Liang, Prefix-Tuning, ACL 2021.
- Lester, Al-Rfou, and Constant, The Power of Scale for Parameter-Efficient Prompt Tuning, EMNLP 2021.
- Wortsman et al., Model Soups, ICML 2022.
- Ilharco et al., Editing Models with Task Arithmetic, ICLR 2023.
- Kirkpatrick et al., Overcoming Catastrophic Forgetting in Neural Networks, PNAS 2017.
- Liu et al., (IA)^3, NeurIPS 2022.
- Schulman et al., Proximal Policy Optimization Algorithms, 2017, arXiv:1707.06347.
- Azar et al., A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO), AISTATS 2024.
- Ethayarajh et al., KTO: Model Alignment as Prospect Theoretic Optimization, 2024.
- These references support only the claims made in Sections 11-51; no other results are attributed to them.
- Where a number is illustrative rather than reported, the text says so explicitly at the point of use.

- End of expansion.
- The original Sections 1-10 are unchanged above.
- Sections 11-52 are the phase-2 expansion.
- Exercises and their learner workspaces are unchanged.
- The theory notebook contains executable versions of the central calculations.
- Companion notebooks: theory.ipynb and exercises.ipynb in this directory.
- Notation follows docs/NOTATION_GUIDE.md.
- Plots in the companion notebook follow docs/VISUALIZATION_GUIDE.md.
- No external data files are required by the companion notebook.
- All random operations in new theory cells use fixed seeds.
- The next section, Scaling Laws, studies how the base models being fine-tuned here are sized.
- Fine-tuning changes the objective and the reachable function class, not the scaling arithmetic underneath.
