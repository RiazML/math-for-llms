[← Back to Math for Specific Models](../README.md) | [Next: Generative Models →](../07-Generative-Models/notes.md)

---

# Reinforcement Learning: Mathematical Foundations

> _"An agent learns not from labelled examples, but from the consequences of its own actions — a fundamentally different mathematical problem that requires dynamic programming, stochastic optimisation, and the theory of Markov chains."_

## Overview

Reinforcement learning (RL) formalises the problem of sequential decision-making under uncertainty. An **agent** interacts with an **environment** over discrete time steps: at each step, the agent observes a state, selects an action, receives a reward, and transitions to a new state. The goal is to learn a **policy** — a mapping from states to actions — that maximises the expected cumulative reward over time.

Unlike supervised learning, where the training signal is an explicit label for each input, the RL training signal is a scalar reward that may be delayed by many time steps. The agent must solve the **credit assignment problem**: which of the hundreds of actions it took actually caused the eventual reward? Unlike unsupervised learning, where there is no training signal at all, RL has a clear objective — but the data distribution depends on the agent's own behaviour, creating a feedback loop between learning and data collection.

The mathematical backbone of RL is the **Markov Decision Process** (MDP), which provides a formal model of the environment. The **Bellman equations** decompose the long-horizon optimisation problem into recursive one-step consistency conditions, enabling dynamic programming algorithms. **Temporal difference learning** combines ideas from Monte Carlo estimation and dynamic programming to learn from incomplete episodes. **Policy gradient methods** directly optimise the policy by estimating gradients of expected return through the log-derivative trick. **Actor-critic methods** combine policy gradients with value function estimation for variance reduction.

This section develops the full mathematical theory from first principles, culminating in the modern algorithms that power RLHF for LLM alignment: PPO, DPO, and reward modelling from human preferences.

## Prerequisites

- Probability: conditional probability, expectation, variance, Bayes' theorem (Chapters 06–07)
- Linear algebra: matrix operations, eigenvalues, fixed-point theory (Chapters 02–03)
- Calculus: partial derivatives, gradients, chain rule (Chapters 04–05)
- Optimisation: gradient descent, convexity, KL divergence (Chapters 09–10)
- Neural networks: forward pass, backpropagation (Section 14-02)
- Information theory: entropy, cross-entropy, KL divergence (Chapter 13)

## Companion Notebooks

| Notebook                           | Description                                                                                                                                                                              |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Interactive demos: gridworld value iteration, Q-learning convergence, policy gradient landscapes, TD vs MC comparison, PPO clipping visualisation, RLHF reward modelling                 |
| [exercises.ipynb](exercises.ipynb) | 10 graded problems: Bellman equations, value iteration, Q-learning, SARSA(λ), REINFORCE with baseline, actor-critic, DQN with replay, Double Q-learning, PPO clipping, experience replay |

## Learning Objectives

After completing this section, you will:

- Define an MDP formally and explain why the Markov property makes the problem tractable
- Derive the Bellman expectation and optimality equations and prove convergence via the contraction mapping theorem
- Implement policy iteration and value iteration, and analyse their convergence rates
- Explain the bias-variance tradeoff between Monte Carlo and TD methods, unified by TD(λ)
- Derive the policy gradient theorem from first principles and prove that baselines reduce variance without introducing bias
- Implement PPO's clipped surrogate objective and explain why it stabilises training
- Derive the RLHF pipeline: reward modelling → PPO fine-tuning → KL-constrained optimisation
- Explain DPO as implicit reward modelling and derive its loss function from the RLHF objective

---

## Table of Contents

- [1. Intuition and Motivation](#1-intuition-and-motivation)
  - [1.1 The Sequential Decision Problem](#11-the-sequential-decision-problem)
  - [1.2 The Reward Hypothesis](#12-the-reward-hypothesis)
  - [1.3 From Supervised to Reinforcement Learning](#13-from-supervised-to-reinforcement-learning)
  - [1.4 Historical Timeline](#14-historical-timeline)
- [2. Markov Decision Processes](#2-markov-decision-processes)
  - [2.1 MDP Definition](#21-mdp-definition)
  - [2.2 The Markov Property](#22-the-markov-property)
  - [2.3 Finite and Infinite Horizon](#23-finite-and-infinite-horizon)
  - [2.4 Discounting and Returns](#24-discounting-and-returns)
  - [2.5 Partially Observable MDPs](#25-partially-observable-mdps)
- [3. Policies and Value Functions](#3-policies-and-value-functions)
  - [3.1 Deterministic and Stochastic Policies](#31-deterministic-and-stochastic-policies)
  - [3.2 State-Value Function](#32-state-value-function)
  - [3.3 Action-Value Function](#33-action-value-function)
  - [3.4 The V–Q Relationship](#34-the-vq-relationship)
  - [3.5 Advantage Function](#35-advantage-function)
  - [3.6 Optimal Value Functions](#36-optimal-value-functions)
- [4. Bellman Equations](#4-bellman-equations)
  - [4.1 Bellman Expectation Equation](#41-bellman-expectation-equation)
  - [4.2 Bellman Optimality Equation](#42-bellman-optimality-equation)
  - [4.3 Contraction Mapping Theorem](#43-contraction-mapping-theorem)
  - [4.4 Matrix Form and Linear Systems](#44-matrix-form-and-linear-systems)
- [5. Dynamic Programming](#5-dynamic-programming)
  - [5.1 Policy Evaluation](#51-policy-evaluation)
  - [5.2 Policy Improvement Theorem](#52-policy-improvement-theorem)
  - [5.3 Policy Iteration](#53-policy-iteration)
  - [5.4 Value Iteration](#54-value-iteration)
  - [5.5 Convergence Rate Analysis](#55-convergence-rate-analysis)
- [6. Monte Carlo Methods](#6-monte-carlo-methods)
  - [6.1 First-Visit and Every-Visit MC](#61-first-visit-and-every-visit-mc)
  - [6.2 MC Control with Exploring Starts](#62-mc-control-with-exploring-starts)
  - [6.3 Off-Policy MC and Importance Sampling](#63-off-policy-mc-and-importance-sampling)
  - [6.4 Bias-Variance in MC Estimation](#64-bias-variance-in-mc-estimation)
- [7. Temporal Difference Learning](#7-temporal-difference-learning)
  - [7.1 TD(0) Prediction](#71-td0-prediction)
  - [7.2 The TD Error as Surprise](#72-the-td-error-as-surprise)
  - [7.3 TD vs MC: Bias-Variance Tradeoff](#73-td-vs-mc-bias-variance-tradeoff)
  - [7.4 N-Step TD Returns](#74-n-step-td-returns)
  - [7.5 TD(λ) and Eligibility Traces](#75-tdλ-and-eligibility-traces)
  - [7.6 Forward and Backward Views](#76-forward-and-backward-views)
- [8. Q-Learning and SARSA](#8-q-learning-and-sarsa)
  - [8.1 SARSA: On-Policy TD Control](#81-sarsa-on-policy-td-control)
  - [8.2 Q-Learning: Off-Policy TD Control](#82-q-learning-off-policy-td-control)
  - [8.3 On-Policy vs Off-Policy](#83-on-policy-vs-off-policy)
  - [8.4 Maximisation Bias and Double Q-Learning](#84-maximisation-bias-and-double-q-learning)
  - [8.5 Convergence of Q-Learning](#85-convergence-of-q-learning)
- [9. Function Approximation and Deep RL](#9-function-approximation-and-deep-rl)
  - [9.1 Why Tables Fail](#91-why-tables-fail)
  - [9.2 Linear Function Approximation](#92-linear-function-approximation)
  - [9.3 The Deadly Triad](#93-the-deadly-triad)
  - [9.4 Deep Q-Networks (DQN)](#94-deep-q-networks-dqn)
  - [9.5 Experience Replay and Target Networks](#95-experience-replay-and-target-networks)
  - [9.6 Rainbow: Combining Improvements](#96-rainbow-combining-improvements)
- [10. Policy Gradient Methods](#10-policy-gradient-methods)
  - [10.1 Policy Parameterisation](#101-policy-parameterisation)
  - [10.2 The Policy Gradient Theorem](#102-the-policy-gradient-theorem)
  - [10.3 REINFORCE](#103-reinforce)
  - [10.4 Variance Reduction: Baselines](#104-variance-reduction-baselines)
  - [10.5 Natural Policy Gradient](#105-natural-policy-gradient)
- [11. Actor-Critic and PPO](#11-actor-critic-and-ppo)
  - [11.1 Actor-Critic Architecture](#111-actor-critic-architecture)
  - [11.2 Generalised Advantage Estimation](#112-generalised-advantage-estimation)
  - [11.3 Trust Regions and TRPO](#113-trust-regions-and-trpo)
  - [11.4 PPO: Clipped Surrogate Objective](#114-ppo-clipped-surrogate-objective)
  - [11.5 Implementation Details That Matter](#115-implementation-details-that-matter)
- [12. Continuous Action Spaces](#12-continuous-action-spaces)
  - [12.1 Deterministic Policy Gradient](#121-deterministic-policy-gradient)
  - [12.2 DDPG and TD3](#122-ddpg-and-td3)
  - [12.3 Maximum Entropy RL](#123-maximum-entropy-rl)
  - [12.4 Soft Actor-Critic](#124-soft-actor-critic)
- [13. RLHF and LLM Alignment](#13-rlhf-and-llm-alignment)
  - [13.1 Reward Modelling from Human Preferences](#131-reward-modelling-from-human-preferences)
  - [13.2 The RLHF Pipeline](#132-the-rlhf-pipeline)
  - [13.3 PPO for Language Models](#133-ppo-for-language-models)
  - [13.4 DPO: Direct Preference Optimisation](#134-dpo-direct-preference-optimisation)
  - [13.5 KL Divergence Constraint](#135-kl-divergence-constraint)
  - [13.6 Constitutional AI and RLAIF](#136-constitutional-ai-and-rlaif)
- [14. Exploration Strategies](#14-exploration-strategies)
  - [14.1 ε-Greedy and Boltzmann Exploration](#141-epsilon-greedy-and-boltzmann-exploration)
  - [14.2 UCB and Optimism](#142-ucb-and-optimism)
  - [14.3 Thompson Sampling](#143-thompson-sampling)
  - [14.4 Intrinsic Motivation and Curiosity](#144-intrinsic-motivation-and-curiosity)
- [15. Model-Based RL](#15-model-based-rl)
  - [15.1 Learned Dynamics Models](#151-learned-dynamics-models)
  - [15.2 Dyna-Q: Integrating Learning and Planning](#152-dyna-q-integrating-learning-and-planning)
  - [15.3 Model Predictive Control](#153-model-predictive-control)
  - [15.4 World Models](#154-world-models)
- [16. Common Mistakes](#16-common-mistakes)
- [17. Exercises](#17-exercises)
- [18. Why This Matters for AI (2026)](#18-why-this-matters-for-ai-2026)
- [19. Conceptual Bridge](#19-conceptual-bridge)

---

## 1. Intuition and Motivation

### 1.1 The Sequential Decision Problem

Consider a robot navigating a maze toward a goal. At each intersection, it must choose a direction. Some paths lead to dead ends, some to the exit. The robot receives no map — it only discovers what happens after taking each step. Crucially, the consequences of a decision may not be immediately apparent: a left turn now might seem fine but lead to a dead end three intersections later. This is the **sequential decision problem**: the agent must reason about how current actions affect future outcomes.

Formally, the problem has four defining characteristics:

1. **Sequential**: Actions are taken over multiple time steps, not all at once.
2. **Stochastic**: The environment may respond nondeterministically — the same action in the same state can lead to different outcomes.
3. **Evaluative**: The agent receives a reward signal telling it how good the outcome was, but not what the optimal action would have been (contrast with supervised learning, where the correct answer is provided).
4. **Non-stationary data**: The data distribution changes as the policy changes — a better policy visits different states, generating different training data.

**The credit assignment problem.** When an agent wins a chess game after 40 moves, which moves deserve credit? The final checkmate clearly contributed, but what about the opening that created the positional advantage? RL must solve this temporal credit assignment problem: propagating reward information backwards through time to determine which state-action pairs were actually responsible for outcomes.

**For AI:** Every LLM trained with RLHF faces exactly this problem. When a human rates a response as "helpful," the reward applies to the entire generated sequence — hundreds of tokens. The RL optimiser must determine which tokens (actions) in which contexts (states) contributed to the high rating.

### 1.2 The Reward Hypothesis

> _"All of what we mean by goals and purposes can be well thought of as the maximisation of the expected value of the cumulative sum of a received scalar signal (called reward)."_
> — Sutton & Barto (2018)

The **reward hypothesis** is the foundational assumption of RL: that all goals can be expressed as maximising cumulative scalar reward. This is a strong claim. Consider:

- **Game playing**: reward = score or win/loss. Natural fit.
- **Robot navigation**: reward = +1 at goal, −1 per step. Straightforward.
- **LLM alignment**: reward = human preference rating. Much harder to define correctly.

**Reward shaping.** The reward function implicitly defines the task. A poorly designed reward creates **reward hacking**: the agent finds an unintended strategy that maximises reward without achieving the designer's true goal. A cleaning robot rewarded for "not seeing dirt" might learn to close its eyes. An LLM rewarded for "sounding confident" might learn to hallucinate assertively.

**Sparse vs dense rewards.** In many problems (chess, Go, complex robotics), the reward is **sparse**: zero on most steps, nonzero only at episode end. This makes credit assignment extremely difficult. Dense reward shaping (providing intermediate rewards) can accelerate learning but risks introducing bias if the shaping function is imperfect.

### 1.3 From Supervised to Reinforcement Learning

| Aspect            | Supervised Learning             | Reinforcement Learning                  |
| ----------------- | ------------------------------- | --------------------------------------- |
| Training signal   | Correct label for each input    | Scalar reward, possibly delayed         |
| Data distribution | Fixed (i.i.d. from dataset)     | Changes with policy (non-stationary)    |
| Feedback          | Instructive ("the answer is X") | Evaluative ("that was +3 good")         |
| Exploration       | Not needed (data is given)      | Essential (must try actions to learn)   |
| Credit assignment | Immediate (loss per example)    | Temporal (which actions caused reward?) |
| Objective         | Minimise empirical risk         | Maximise expected cumulative reward     |

The key mathematical distinction: in supervised learning, the loss gradient $\nabla_\theta \mathcal{L}$ is straightforward because the data does not depend on $\theta$. In RL, the trajectory distribution $p_\theta(\tau)$ depends on the policy $\pi_\theta$, so the gradient of expected return requires the **log-derivative trick**:

$$\nabla_\theta \mathbb{E}_{\tau \sim p_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim p_\theta}[R(\tau) \nabla_\theta \log p_\theta(\tau)]$$

This is the policy gradient — the mathematical bridge from RL to optimisation.

### 1.4 Historical Timeline

```text
REINFORCEMENT LEARNING TIMELINE
════════════════════════════════════════════════════════════════════════

  1950  Bellman              Dynamic programming, Bellman equation
  1957  Bellman              Markov Decision Processes formalised
  1972  Klopf                Adaptive critic elements
  1988  Sutton               TD(λ), temporal difference learning
  1989  Watkins              Q-learning (off-policy TD control)
  1992  Tesauro              TD-Gammon: backgammon via TD learning
  1994  Rummery & Niranjan   SARSA (on-policy TD control)
  1999  Sutton et al.        Policy gradient theorem
  2004  Kakade               Natural policy gradient
  2013  Mnih et al.          DQN: Atari from pixels
  2015  Schulman et al.      TRPO: trust region policy optimisation
  2016  Silver et al.        AlphaGo: Monte Carlo tree search + RL
  2017  Schulman et al.      PPO: proximal policy optimisation
  2017  Haarnoja et al.      Soft Actor-Critic (SAC)
  2017  Christiano et al.    RLHF: learning from human preferences
  2022  Ouyang et al.        InstructGPT: RLHF for LLMs (PPO)
  2023  Rafailov et al.      DPO: direct preference optimisation
  2024  DeepSeek             GRPO: group relative policy optimisation
  2025  Anthropic             Constitutional AI / RLAIF at scale

════════════════════════════════════════════════════════════════════════
```

---

## 2. Markov Decision Processes

### 2.1 MDP Definition

A **Markov Decision Process** is a tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$:

- $\mathcal{S}$: **State space** — the set of all possible states the environment can be in. May be finite (gridworld), countably infinite (queuing systems), or continuous ($\mathbb{R}^d$ for robotics).
- $\mathcal{A}$: **Action space** — the set of all possible actions. May depend on state: $\mathcal{A}(s)$. Discrete (left/right/up/down) or continuous (torque, velocity).
- $P: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to [0, 1]$: **Transition dynamics** — $P(s' \mid s, a)$ is the probability of transitioning to state $s'$ given the agent takes action $a$ in state $s$. Satisfies $\sum_{s' \in \mathcal{S}} P(s' \mid s, a) = 1$ for all $s, a$.
- $R: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{R}$: **Reward function** — $R(s, a, s')$ is the immediate reward received upon transitioning from $s$ to $s'$ via action $a$. Often simplified to $R(s, a)$ or $R(s)$.
- $\gamma \in [0, 1)$: **Discount factor** — controls the relative importance of future vs immediate rewards.

**For AI:** An LLM generating text can be modelled as an MDP where the state is the prompt plus all tokens generated so far, the action is the next token chosen from the vocabulary $\mathcal{V}$, the transition is deterministic (state = previous state + chosen token), and the reward comes from a reward model (or human evaluator) at the end of generation.

### 2.2 The Markov Property

The defining assumption of an MDP is the **Markov property** (memorylessness):

$$P(S_{t+1} = s' \mid S_t = s, A_t = a, S_{t-1}, A_{t-1}, \ldots, S_0, A_0) = P(S_{t+1} = s' \mid S_t = s, A_t = a)$$

The future is conditionally independent of the past given the present state. The state $S_t$ contains all information needed to predict the future — no history is required.

**When Markov fails.** Many real problems are **non-Markov**. A poker game where you cannot see opponents' cards. A robot with noisy sensors. An LLM where the reward depends on user intent not captured in the prompt. In these cases, the true state is partially observed, and we need POMDPs (Section 2.5) or must engineer the state representation to be approximately Markov (e.g., including history in the state via frame stacking in Atari, or the full conversation context in LLMs).

**Why Markov matters mathematically.** The Markov property converts the full trajectory optimisation problem into a recursive problem. Without it, we would need to condition on entire histories, making the state space grow exponentially with time. With it, we can write value functions as functions of a single state $V(s)$ rather than entire histories $V(s_0, a_0, s_1, a_1, \ldots, s_t)$.

### 2.3 Finite and Infinite Horizon

**Finite horizon** ($T < \infty$): The agent acts for exactly $T$ steps. The objective is:

$$J(\pi) = \mathbb{E}_\pi\left[\sum_{t=0}^{T-1} R(S_t, A_t)\right]$$

No discounting is needed because the sum is finite. However, the optimal policy may be **non-stationary**: the best action at state $s$ depends on how many steps remain. At step $T-1$, the agent should be greedy; at step 0, it should plan ahead.

**Infinite horizon** ($T = \infty$) with discounting: The objective becomes:

$$J(\pi) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R(S_t, A_t)\right]$$

The discount factor $\gamma < 1$ ensures the sum converges (assuming rewards are bounded: $|R| \le R_{\max}$):

$$\left|\sum_{t=0}^{\infty} \gamma^t R_t\right| \le R_{\max} \sum_{t=0}^{\infty} \gamma^t = \frac{R_{\max}}{1 - \gamma}$$

**Interpretation of $\gamma$.** The discount factor has multiple interpretations: (1) **Time preference** — future rewards are worth less than immediate ones. (2) **Probability of continuing** — if the episode terminates with probability $1 - \gamma$ at each step, then $\gamma^t$ is the probability of reaching step $t$. (3) **Mathematical convenience** — ensures convergence and makes the Bellman operator a contraction.

### 2.4 Discounting and Returns

The **return** $G_t$ is the discounted sum of future rewards from time step $t$:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

The crucial **recursive structure**:

$$G_t = R_{t+1} + \gamma G_{t+1}$$

This recursive decomposition is what makes the Bellman equations possible. Every RL algorithm exploits this structure in some way: dynamic programming uses it to compute values exactly, TD learning uses it to bootstrap estimates, and Monte Carlo methods compute it by summing actual rewards.

**Effective horizon.** The discount factor $\gamma$ defines an effective planning horizon of $\frac{1}{1 - \gamma}$ steps. For $\gamma = 0.99$, the effective horizon is 100 steps — rewards beyond 100 steps are attenuated by a factor $> e^{-1}$. For $\gamma = 0.999$, the horizon extends to 1000 steps. Choosing $\gamma$ is choosing how far ahead the agent plans.

| $\gamma$ | Effective horizon | Character                               |
| -------- | ----------------- | --------------------------------------- |
| 0.9      | 10 steps          | Myopic — focuses on immediate rewards   |
| 0.99     | 100 steps         | Moderate — balances near and far        |
| 0.999    | 1000 steps        | Far-sighted — plans over long sequences |
| 1.0      | $\infty$          | Undiscounted — only finite episodes     |

### 2.5 Partially Observable MDPs

When the agent cannot observe the full state, we have a **Partially Observable MDP** (POMDP), defined by $(\mathcal{S}, \mathcal{A}, P, R, \gamma, \Omega, O)$ where $\Omega$ is the observation space and $O(o \mid s, a)$ is the observation function.

The agent maintains a **belief state** $b_t \in \Delta(\mathcal{S})$ — a probability distribution over states — updated via Bayes' rule:

$$b_{t+1}(s') = \frac{O(o_{t+1} \mid s', a_t) \sum_{s} P(s' \mid s, a_t) b_t(s)}{\sum_{s''} O(o_{t+1} \mid s'', a_t) \sum_{s} P(s'' \mid s, a_t) b_t(s)}$$

The belief MDP (over belief states) is Markov, but the belief space is continuous and high-dimensional, making exact solutions intractable for most problems. In practice, deep RL handles partial observability by using recurrent networks (LSTMs) or Transformers that condition on observation histories, effectively learning an implicit belief representation.

**For AI:** An LLM chatbot operates in a POMDP: the true "state" includes the user's intent, knowledge, and emotional state, but the model only observes the text of the conversation. The Transformer's ability to condition on the entire conversation history is its mechanism for maintaining an implicit belief state.

---

## 3. Policies and Value Functions

### 3.1 Deterministic and Stochastic Policies

A **policy** $\pi$ is a rule for selecting actions. Two forms:

**Deterministic policy:** $a = \pi(s)$ — a function mapping each state to a single action.

**Stochastic policy:** $\pi(a \mid s) = P(A_t = a \mid S_t = s)$ — a conditional distribution over actions given a state. Satisfies $\pi(a \mid s) \ge 0$ and $\sum_{a \in \mathcal{A}} \pi(a \mid s) = 1$ for all $s$.

Stochastic policies are more general (every deterministic policy is a degenerate stochastic policy), and they are essential for:

1. **Exploration:** A deterministic policy cannot explore — it always takes the same action in the same state.
2. **Optimality in POMDPs:** Under partial observability, stochastic policies can be strictly better than deterministic ones.
3. **Policy gradient methods:** Parameterising $\pi_\theta(a \mid s)$ as a differentiable distribution (e.g., softmax over logits) enables gradient-based optimisation.

**For AI:** An LLM's output is a stochastic policy: $\pi_\theta(a \mid s) = P(\text{next token} = a \mid \text{context} = s)$, parameterised by the softmax output layer. Temperature sampling adjusts the entropy of this policy.

### 3.2 State-Value Function

The **state-value function** $V^\pi(s)$ measures how good it is to be in state $s$ under policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\Big|\; S_t = s\right]$$

This is the expected return starting from state $s$ and following policy $\pi$ thereafter. The expectation is over both the stochasticity of the policy and the stochasticity of the environment transitions.

**Properties:**

- $V^\pi(s) \in [-R_{\max}/(1-\gamma), \; R_{\max}/(1-\gamma)]$ for bounded rewards.
- For a terminal state $s_{\text{term}}$: $V^\pi(s_{\text{term}}) = 0$.
- $V^\pi$ is the unique fixed point of the Bellman expectation operator for policy $\pi$ (Section 4).

### 3.3 Action-Value Function

The **action-value function** (Q-function) $Q^\pi(s, a)$ measures how good it is to take action $a$ in state $s$ and then follow policy $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\Big|\; S_t = s, A_t = a\right]$$

$Q^\pi$ is more useful than $V^\pi$ for control because it directly tells us the value of each action — we can improve the policy by choosing $\arg\max_a Q^\pi(s, a)$ without knowing the transition dynamics $P$.

### 3.4 The V–Q Relationship

The value functions are related by marginalising over actions:

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \, Q^\pi(s, a)$$

The state-value is the expected action-value under the policy. Conversely, the Q-function decomposes into immediate reward plus discounted next-state value:

$$Q^\pi(s, a) = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^\pi(s')\right]$$

These two equations together give the **Bellman expectation equation** — the most important equation in RL (Section 4).

```text
VALUE FUNCTION RELATIONSHIPS
════════════════════════════════════════════════════════════════════════

  V^π(s) ──── π(a|s) ────→  Q^π(s,a)
    │                          │
    │                          │ P(s'|s,a), R(s,a,s')
    │                          ↓
    │                     R + γ V^π(s')
    │                          │
    └──────────────────────────┘

  V = Σ_a π(a|s) Q(s,a)           (marginalize over actions)
  Q = Σ_s' P(s'|s,a) [R + γ V(s')]  (one-step lookahead)

════════════════════════════════════════════════════════════════════════
```

### 3.5 Advantage Function

The **advantage function** measures how much better action $a$ is compared to the average action under policy $\pi$:

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

**Properties:**

- $\sum_a \pi(a \mid s) A^\pi(s, a) = 0$ — the expected advantage is zero by construction.
- $A^\pi(s, a) > 0$ means action $a$ is better than average.
- $A^\pi(s, a) < 0$ means action $a$ is worse than average.

The advantage function is central to modern policy gradient methods. Using advantages instead of raw returns dramatically reduces variance in gradient estimates (Section 10.4). PPO, A2C, and GAE all operate on advantage estimates.

### 3.6 Optimal Value Functions

The **optimal state-value function** $V^*(s)$ is the maximum value achievable from any state:

$$V^*(s) = \max_\pi V^\pi(s) \qquad \forall s \in \mathcal{S}$$

The **optimal action-value function** $Q^*(s, a)$:

$$Q^*(s, a) = \max_\pi Q^\pi(s, a) \qquad \forall s, a$$

**Theorem.** For any finite MDP, there exists at least one **optimal policy** $\pi^*$ that simultaneously maximises $V^\pi(s)$ for all states $s$. This policy can be extracted from $Q^*$:

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

The remarkable fact is that a single policy can be optimal for all states simultaneously — there is no need to trade off performance in one state against another. This follows from the Bellman optimality equation (Section 4.2).

---

## 4. Bellman Equations

### 4.1 Bellman Expectation Equation

Substituting the V–Q relationships from Section 3.4 into each other yields the **Bellman expectation equations**:

**For $V^\pi$:**

$$V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^\pi(s')\right]$$

_Derivation._ Start from $V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$. Use the recursive return $G_t = R_{t+1} + \gamma G_{t+1}$:

$$V^\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma G_{t+1} \mid S_t = s]$$

$$= \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s,a,s') + \gamma \,\mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s']\right]$$

$$= \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s,a,s') + \gamma V^\pi(s')\right]$$

The second step uses the law of total expectation and the Markov property (conditioning on $S_{t+1} = s'$ suffices). The third step recognises $\mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s'] = V^\pi(s')$.

**For $Q^\pi$:**

$$Q^\pi(s, a) = \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')\right]$$

**Interpretation:** The Bellman expectation equation says: the value of a state equals the expected immediate reward plus the discounted value of the next state. This is a **consistency condition** — any correct value function must satisfy it. The equation does not tell you how to compute $V^\pi$; it tells you what $V^\pi$ must satisfy.

### 4.2 Bellman Optimality Equation

For the **optimal** value functions, the policy is replaced by a $\max$:

**For $V^*$:**

$$V^*(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^*(s')\right]$$

**For $Q^*$:**

$$Q^*(s, a) = \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma \max_{a'} Q^*(s', a')\right]$$

The Bellman expectation equation is **linear** in $V^\pi$ (for a fixed policy $\pi$). The Bellman optimality equation is **nonlinear** due to the $\max$ operator. This nonlinearity means we cannot solve it by matrix inversion — we need iterative methods.

### 4.3 Contraction Mapping Theorem

**Definition.** An operator $\mathcal{T}$ on a complete metric space is a **contraction mapping** with modulus $\gamma < 1$ if:

$$\lVert \mathcal{T}f - \mathcal{T}g \rVert_\infty \le \gamma \lVert f - g \rVert_\infty \qquad \forall f, g$$

**Banach Fixed-Point Theorem.** Every contraction mapping has a unique fixed point $f^*$ satisfying $\mathcal{T}f^* = f^*$, and the sequence $f_{k+1} = \mathcal{T}f_k$ converges to $f^*$ from any starting point $f_0$.

**Bellman optimality operator.** Define $\mathcal{T}^*$ acting on value functions:

$$(\mathcal{T}^* V)(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V(s')\right]$$

**Theorem.** $\mathcal{T}^*$ is a contraction with modulus $\gamma$ in the $\ell^\infty$-norm.

_Proof._ For any two value functions $V_1, V_2$:

$$|(\mathcal{T}^* V_1)(s) - (\mathcal{T}^* V_2)(s)| = \left|\max_a \sum_{s'} P(s' \mid s, a) [R + \gamma V_1(s')] - \max_a \sum_{s'} P(s' \mid s, a) [R + \gamma V_2(s')]\right|$$

Using $|\max_a f(a) - \max_a g(a)| \le \max_a |f(a) - g(a)|$:

$$\le \max_a \sum_{s'} P(s' \mid s, a) \gamma |V_1(s') - V_2(s')| \le \gamma \lVert V_1 - V_2 \rVert_\infty$$

since $\sum_{s'} P(s' \mid s, a) = 1$. Therefore $\lVert \mathcal{T}^* V_1 - \mathcal{T}^* V_2 \rVert_\infty \le \gamma \lVert V_1 - V_2 \rVert_\infty$. $\square$

**Consequence:** Value iteration converges to $V^*$ at a geometric rate. After $k$ iterations: $\lVert V_k - V^* \rVert_\infty \le \gamma^k \lVert V_0 - V^* \rVert_\infty$. For $\gamma = 0.99$, we need about $k = 460$ iterations to reduce the error by a factor of $100$.

### 4.4 Matrix Form and Linear Systems

For a **fixed policy** $\pi$ with finite state and action spaces, the Bellman expectation equation is a linear system. Define:

- $\mathbf{v} \in \mathbb{R}^{|\mathcal{S}|}$: vector of state values
- $\mathbf{r}^\pi \in \mathbb{R}^{|\mathcal{S}|}$: expected immediate reward vector, $r^\pi_s = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) R(s, a, s')$
- $P^\pi \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}$: transition matrix under $\pi$, $P^\pi_{ss'} = \sum_a \pi(a \mid s) P(s' \mid s, a)$

The Bellman expectation equation becomes:

$$\mathbf{v} = \mathbf{r}^\pi + \gamma P^\pi \mathbf{v}$$

Solving: $(I - \gamma P^\pi) \mathbf{v} = \mathbf{r}^\pi$, so:

$$\mathbf{v} = (I - \gamma P^\pi)^{-1} \mathbf{r}^\pi$$

This gives the **exact** value function in $O(|\mathcal{S}|^3)$ time (matrix inversion). For small state spaces this is practical; for large state spaces, iterative methods (value iteration, TD learning) are necessary.

**Why $(I - \gamma P^\pi)$ is invertible:** Since $P^\pi$ is a stochastic matrix, its spectral radius is $\rho(P^\pi) \le 1$. With $\gamma < 1$, we have $\rho(\gamma P^\pi) < 1$, so $(I - \gamma P^\pi)$ is non-singular. In fact, $(I - \gamma P^\pi)^{-1} = \sum_{k=0}^{\infty} (\gamma P^\pi)^k$ — the Neumann series converges because $\gamma < 1$.

---

## 5. Dynamic Programming

Dynamic programming (DP) methods compute optimal policies given complete knowledge of the MDP (i.e., $P$ and $R$ are known). They serve as the theoretical foundation for all RL algorithms, even those that work without a model.

### 5.1 Policy Evaluation

**Problem:** Given a policy $\pi$, compute its value function $V^\pi$.

**Iterative policy evaluation** repeatedly applies the Bellman expectation operator:

$$V_{k+1}(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V_k(s')\right]$$

Starting from any $V_0$, this converges to $V^\pi$ because the Bellman expectation operator (for a fixed policy) is a contraction with modulus $\gamma$.

**Convergence criterion:** Stop when $\lVert V_{k+1} - V_k \rVert_\infty < \theta$ for a small threshold $\theta$.

**Complexity:** Each sweep costs $O(|\mathcal{S}|^2 |\mathcal{A}|)$ — for each state, we loop over all actions and all next states.

### 5.2 Policy Improvement Theorem

**Theorem (Policy Improvement).** Let $\pi$ be a policy with value function $V^\pi$. Define a new policy $\pi'$ that is greedy with respect to $V^\pi$:

$$\pi'(s) = \arg\max_a \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^\pi(s')\right] = \arg\max_a Q^\pi(s, a)$$

Then $V^{\pi'}(s) \ge V^\pi(s)$ for all $s \in \mathcal{S}$, with equality if and only if $\pi$ is already optimal.

_Proof sketch._ For any state $s$:

$$V^\pi(s) \le Q^\pi(s, \pi'(s)) = \sum_{s'} P(s' \mid s, \pi'(s)) [R + \gamma V^\pi(s')]$$

$$\le \sum_{s'} P(s' \mid s, \pi'(s)) [R + \gamma Q^\pi(s', \pi'(s'))]$$

Repeating this expansion and using the Bellman equation for $\pi'$:

$$\le V^{\pi'}(s)$$

The inequality $V^\pi(s) \le Q^\pi(s, \pi'(s))$ holds because $\pi'(s)$ maximises $Q^\pi(s, \cdot)$, which is at least as good as the average $\sum_a \pi(a \mid s) Q^\pi(s, a) = V^\pi(s)$.

### 5.3 Policy Iteration

**Algorithm:**

1. **Initialise** $\pi_0$ arbitrarily.
2. **Policy evaluation:** Compute $V^{\pi_k}$ (solve the Bellman expectation equation for the current policy).
3. **Policy improvement:** $\pi_{k+1}(s) = \arg\max_a Q^{\pi_k}(s, a)$ for all $s$.
4. If $\pi_{k+1} = \pi_k$, stop (policy is optimal). Otherwise go to step 2.

**Convergence:** By the policy improvement theorem, $V^{\pi_{k+1}}(s) \ge V^{\pi_k}(s)$ for all $s$. Since there are finitely many deterministic policies ($|\mathcal{A}|^{|\mathcal{S}|}$), the sequence must converge in a finite number of iterations. In practice, convergence is remarkably fast — often in fewer than 10 iterations even for large state spaces.

### 5.4 Value Iteration

Value iteration combines the evaluation and improvement steps into a single update:

$$V_{k+1}(s) = \max_a \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V_k(s')\right]$$

This is equivalent to performing one sweep of policy evaluation followed by one step of policy improvement, repeated. It converges to $V^*$ by the contraction mapping theorem (Section 4.3).

**Extract policy after convergence:**

$$\pi^*(s) = \arg\max_a \sum_{s'} P(s' \mid s, a) \left[R(s, a, s') + \gamma V^*(s')\right]$$

**Value iteration vs policy iteration:** Value iteration performs a single Bellman backup per sweep and requires many sweeps ($\sim \frac{1}{1-\gamma}$). Policy iteration performs full policy evaluation (many backups) but requires very few outer iterations. In practice, the choice depends on the problem structure — policy iteration often wins for small to moderate state spaces.

### 5.5 Convergence Rate Analysis

**Value iteration convergence rate.** After $k$ iterations:

$$\lVert V_k - V^* \rVert_\infty \le \gamma^k \lVert V_0 - V^* \rVert_\infty \le \gamma^k \cdot \frac{2 R_{\max}}{1 - \gamma}$$

To achieve $\epsilon$-accuracy: $k \ge \frac{1}{1 - \gamma} \ln\!\left(\frac{2 R_{\max}}{\epsilon(1 - \gamma)}\right)$. For $\gamma = 0.99$, $R_{\max} = 1$, $\epsilon = 0.01$: $k \ge 100 \ln(20000) \approx 990$ iterations.

**Policy iteration convergence rate.** In the worst case, policy iteration converges in $O(|\mathcal{A}|^{|\mathcal{S}|})$ iterations (trying all policies), but this is never observed in practice. Ye (2011) showed that policy iteration converges in $O(|\mathcal{S}| |\mathcal{A}| / (1-\gamma))$ iterations — polynomial, not exponential. Empirically, convergence is typically much faster than this bound suggests.

| Algorithm         | Per-iteration cost                                                                 | Iterations to converge                          | Model required? |
| ----------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------- | --------------- |
| Policy evaluation | $O(\lvert\mathcal{S}\rvert^2 \lvert\mathcal{A}\rvert)$                             | $O(\frac{1}{1-\gamma} \log \frac{1}{\epsilon})$ | Yes             |
| Value iteration   | $O(\lvert\mathcal{S}\rvert^2 \lvert\mathcal{A}\rvert)$                             | $O(\frac{1}{1-\gamma} \log \frac{1}{\epsilon})$ | Yes             |
| Policy iteration  | $O(\lvert\mathcal{S}\rvert^3 + \lvert\mathcal{S}\rvert^2 \lvert\mathcal{A}\rvert)$ | Very few (typically $< 20$)                     | Yes             |
| Direct solve      | $O(\lvert\mathcal{S}\rvert^3)$                                                     | 1 (matrix inversion)                            | Yes             |

---

## 6. Monte Carlo Methods

DP requires a model ($P$ and $R$). Monte Carlo (MC) methods learn from experience — actual episodes of interaction with the environment — without knowing the transition dynamics.

### 6.1 First-Visit and Every-Visit MC

**First-visit MC prediction:** To estimate $V^\pi(s)$, collect many episodes under policy $\pi$. For each episode, find the **first** time step $t$ at which state $s$ is visited, compute the return $G_t$ from that point, and average across episodes:

$$V^\pi(s) \approx \frac{1}{N(s)} \sum_{i=1}^{N(s)} G_t^{(i)}$$

**Every-visit MC:** Uses every visit to state $s$ within each episode, not just the first. Both are consistent estimators — they converge to $V^\pi(s)$ as the number of episodes $\to \infty$. First-visit MC is unbiased; every-visit MC is biased (correlated samples within an episode) but often has lower variance in practice.

**Incremental update.** Rather than storing all returns and averaging, we can update incrementally:

$$V(s) \leftarrow V(s) + \frac{1}{N(s)} \left[G_t - V(s)\right]$$

Or with a fixed learning rate $\alpha$ (which forgets old data and adapts to non-stationarity):

$$V(s) \leftarrow V(s) + \alpha \left[G_t - V(s)\right]$$

### 6.2 MC Control with Exploring Starts

To find the optimal policy, we need MC for Q-values (not just V-values), because improving a policy from V requires knowing $P$, but improving from Q does not.

**MC control with exploring starts:**

1. Initialise $Q(s, a)$ and $\pi$ arbitrarily.
2. For each episode: choose a random starting state-action pair (exploring starts), then follow $\pi$.
3. For each $(s, a)$ in the episode, update $Q(s, a)$ using first-visit MC.
4. For each $s$ visited, set $\pi(s) = \arg\max_a Q(s, a)$.

The **exploring starts** assumption ensures every state-action pair is visited infinitely often. This is unrealistic for most applications — we cannot choose the starting conditions of a robot or an LLM conversation. The solution is to use stochastic policies that ensure exploration ($\epsilon$-greedy, softmax).

### 6.3 Off-Policy MC and Importance Sampling

**Off-policy learning** uses data from a **behaviour policy** $b(a \mid s)$ (which explores) to evaluate a **target policy** $\pi(a \mid s)$ (which may be greedy). This requires correcting for the distributional mismatch via **importance sampling**.

For a trajectory $\tau = (S_0, A_0, R_1, S_1, \ldots, S_T)$, the importance sampling ratio is:

$$\rho_{t:T-1} = \prod_{k=t}^{T-1} \frac{\pi(A_k \mid S_k)}{b(A_k \mid S_k)}$$

The off-policy estimate:

$$V^\pi(s) \approx \frac{\sum_{t \in \mathcal{T}(s)} \rho_{t:T-1} G_t}{|\mathcal{T}(s)|}$$

**Variance explosion.** The product of ratios can be enormous if $\pi$ and $b$ differ significantly. For a trajectory of length $T$ with $\pi/b \approx 2$ at each step, $\rho \approx 2^T$ — exponentially large. This makes ordinary importance sampling impractical for long episodes.

**Weighted importance sampling** uses:

$$V^\pi(s) \approx \frac{\sum_{t} \rho_{t:T-1} G_t}{\sum_{t} \rho_{t:T-1}}$$

This is biased but has dramatically lower variance, making it the standard choice.

### 6.4 Bias-Variance in MC Estimation

MC methods have a characteristic bias-variance profile:

- **Unbiased:** First-visit MC gives an unbiased estimate of $V^\pi(s)$ (the return $G_t$ is an unbiased sample of $\mathbb{E}_\pi[G_t \mid S_t = s]$).
- **High variance:** The return depends on the entire trajectory after time $t$ — all the stochasticity of actions and transitions accumulates. For a trajectory of length $T$, $\operatorname{Var}(G_t)$ grows with $T$.
- **No bootstrap bias:** Unlike TD methods (Section 7), MC does not use value estimates as targets, so there is no bias from inaccurate bootstrapping.

This tradeoff — unbiased but high variance — is the fundamental tension between MC and TD methods, resolved by the TD(λ) spectrum (Section 7.5).

---

## 7. Temporal Difference Learning

TD learning is the most important algorithmic idea in RL. It combines the bootstrapping of DP (using value estimates as targets) with the model-free, sample-based approach of MC.

### 7.1 TD(0) Prediction

**TD(0) update** for estimating $V^\pi$:

$$V(S_t) \leftarrow V(S_t) + \alpha \left[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\right]$$

Compare with MC: $V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$

The key difference: MC uses the actual return $G_t$ (waits until the end of the episode), while TD(0) uses $R_{t+1} + \gamma V(S_{t+1})$ — the **TD target** — which is available after a single step. TD(0) bootstraps: it uses the current estimate $V(S_{t+1})$ as a stand-in for the true expected future return.

### 7.2 The TD Error as Surprise

The **TD error** $\delta_t$ measures the surprise at each time step:

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

**Interpretation:** Before taking action, the agent expected to receive $V(S_t)$ worth of future return from state $S_t$. After one step, it received $R_{t+1}$ and ended up in state $S_{t+1}$ with estimated future value $V(S_{t+1})$. The TD error is the discrepancy between what was expected and what was observed.

- $\delta_t > 0$: Things went better than expected. Increase $V(S_t)$.
- $\delta_t < 0$: Things went worse than expected. Decrease $V(S_t)$.
- $\delta_t = 0$: Prediction was correct. No update needed.

**Connection to dopamine.** Neuroscience research by Schultz et al. (1997) showed that dopaminergic neurons in the brain fire in a pattern strikingly similar to the TD error: they fire when reward is unexpectedly received (positive $\delta$), decrease firing when expected reward is omitted (negative $\delta$), and are silent when reward is received as expected ($\delta = 0$). This is one of the most striking parallels between computational RL and neuroscience.

**Key property:** If $V = V^\pi$ (the true value function), then $\mathbb{E}_\pi[\delta_t \mid S_t = s] = 0$ — the TD error has zero mean. This means TD(0) is a stochastic approximation of the Bellman equation, and convergence follows from the Robbins-Monro conditions on the learning rate: $\sum_t \alpha_t = \infty$ and $\sum_t \alpha_t^2 < \infty$.

### 7.3 TD vs MC: Bias-Variance Tradeoff

|                            | Monte Carlo                               | TD(0)                               |
| -------------------------- | ----------------------------------------- | ----------------------------------- |
| Target                     | $G_t = R_{t+1} + \gamma R_{t+2} + \cdots$ | $R_{t+1} + \gamma V(S_{t+1})$       |
| Bias                       | Unbiased (uses true returns)              | Biased (uses estimate $V(S_{t+1})$) |
| Variance                   | High (accumulates over episode)           | Low (single-step randomness)        |
| Requires episodes?         | Yes (must wait for episode end)           | No (updates after every step)       |
| Works in continuing tasks? | No                                        | Yes                                 |
| Converges to $V^\pi$?      | Yes ($\alpha_t$ conditions)               | Yes ($\alpha_t$ conditions)         |

**Bias of TD(0):** The TD target $R_{t+1} + \gamma V(S_{t+1})$ uses the current estimate $V(S_{t+1})$, which may be wrong. This introduces bias — the target is not an unbiased estimate of $V^\pi(S_t)$. However, this bias decreases as $V$ approaches $V^\pi$, and in the limit TD(0) converges to the correct values.

**Why TD is preferred.** Despite the bias, TD methods are preferred in practice because: (1) they can learn before episode ends (online), (2) they work in continuing (non-episodic) tasks, (3) their lower variance means faster convergence in practice, and (4) they are computationally cheaper per step.

### 7.4 N-Step TD Returns

The n-step return bridges MC and TD(0):

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

$$V(S_t) \leftarrow V(S_t) + \alpha \left[G_t^{(n)} - V(S_t)\right]$$

- $n = 1$: TD(0) — one step of real reward, then bootstrap.
- $n = 2$: Two steps of real reward, then bootstrap.
- $n = \infty$: MC — use only real rewards, no bootstrapping.

The optimal $n$ is problem-dependent. Small $n$ has low variance but high bias; large $n$ has low bias but high variance. TD(λ) provides a principled way to combine all n-step returns.

### 7.5 TD(λ) and Eligibility Traces

**TD(λ)** combines all n-step returns using exponential weighting:

$$G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

The weights $(1 - \lambda)\lambda^{n-1}$ form a geometric distribution that sums to 1. This is the **forward view**: the λ-return is an exponentially-weighted average of all n-step returns.

- $\lambda = 0$: $G_t^0 = G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})$ → TD(0).
- $\lambda = 1$: $G_t^1 = G_t$ → MC (full return).
- $0 < \lambda < 1$: A smooth blend between TD and MC.

The **backward view** uses **eligibility traces** $\mathbf{e}_t \in \mathbb{R}^{|\mathcal{S}|}$:

$$e_t(s) = \gamma \lambda \, e_{t-1}(s) + \mathbf{1}(S_t = s)$$

$$V(s) \leftarrow V(s) + \alpha \, \delta_t \, e_t(s) \qquad \forall s$$

The trace $e_t(s)$ records how recently and frequently state $s$ was visited. When a TD error $\delta_t$ occurs, all recently visited states are updated proportionally to their eligibility. This is computationally efficient: one sweep per time step, rather than waiting for the episode to end.

### 7.6 Forward and Backward Views

**Theorem (Equivalence).** For online TD(λ) with accumulating traces, the total update to each state over a complete episode equals the update that would be computed using the forward-view λ-return:

$$\sum_{t=0}^{T-1} \alpha \delta_t e_t(s) = \sum_{t=0}^{T-1} \alpha \left[G_t^\lambda - V(S_t)\right] \mathbf{1}(S_t = s)$$

This equivalence holds exactly for offline updates (batch at episode end) and approximately for online updates (step-by-step). The practical significance: the backward view with eligibility traces is an efficient, online implementation of the mathematically elegant forward view.

**Replacing traces vs accumulating traces.** The update $e_t(s) = \gamma \lambda \, e_{t-1}(s) + \mathbf{1}(S_t = s)$ is the **accumulating trace** — it adds 1 each time $s$ is visited. The **replacing trace** sets $e_t(s) = 1$ when $S_t = s$ (capping the trace at 1). Replacing traces often work better empirically, as accumulating traces can become very large in states visited frequently.

---

## 8. Q-Learning and SARSA

### 8.1 SARSA: On-Policy TD Control

**SARSA** (State-Action-Reward-State-Action) is the on-policy TD control algorithm:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\right]$$

The name comes from the quintuple $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ used in each update. Crucially, $A_{t+1}$ is the action **actually taken** by the agent in state $S_{t+1}$ — not the greedy action.

**On-policy** means SARSA evaluates and improves the same policy it uses to collect data. If the agent uses $\epsilon$-greedy exploration, SARSA learns the value of the $\epsilon$-greedy policy — which is slightly worse than the optimal policy because it includes random actions. This is actually desirable in certain settings (e.g., cliff walking) where the optimal policy is dangerous during exploration.

**SARSA(λ)** extends SARSA with eligibility traces for faster credit assignment:

$$e_t(s, a) = \gamma \lambda \, e_{t-1}(s, a) + \mathbf{1}(S_t = s, A_t = a)$$

$$Q(s, a) \leftarrow Q(s, a) + \alpha \, \delta_t \, e_t(s, a) \qquad \forall s, a$$

### 8.2 Q-Learning: Off-Policy TD Control

**Q-learning** (Watkins, 1989) is the off-policy TD control algorithm:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)\right]$$

The critical difference from SARSA: the target uses $\max_a Q(S_{t+1}, a)$ — the **best** possible action in the next state, regardless of what the agent actually did. This makes Q-learning **off-policy**: it learns $Q^*$ (the optimal Q-function) regardless of the exploration policy.

**Why this works.** The Q-learning update is a stochastic approximation of the Bellman optimality equation: $Q^*(s, a) = \mathbb{E}[R + \gamma \max_{a'} Q^*(S', a')]$. Each update moves $Q$ toward satisfying this equation for the sampled transition.

### 8.3 On-Policy vs Off-Policy

The cliff-walking example illustrates the difference clearly:

```text
CLIFF WALKING ENVIRONMENT
════════════════════════════════════════════════════════════════════════

  ┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
  │   │   │   │   │   │   │   │   │   │   │   │   │
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
  │   │   │   │   │   │   │   │   │   │   │   │   │
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
  │ → │ → │ → │ → │ → │ → │ → │ → │ → │ → │ → │ ↓ │  ← SARSA (safe)
  ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
  │ S │ ☠ │ ☠ │ ☠ │ ☠ │ ☠ │ ☠ │ ☠ │ ☠ │ ☠ │ ☠ │ G │
  └───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
                    ↑ Cliff (reward = -100)

  Q-learning finds the optimal path (along the cliff edge)
  but during ε-greedy exploration, the agent often falls off.

  SARSA finds a safer path (one row above the cliff)
  because it accounts for its own exploration noise.

════════════════════════════════════════════════════════════════════════
```

**Q-learning** finds the optimal policy (along the cliff) but suffers during training because $\epsilon$-greedy occasionally steps into the cliff. **SARSA** finds a safer suboptimal policy (one row above the cliff) because it learns the value of the $\epsilon$-greedy policy, which accounts for the risk of random cliff-falling.

### 8.4 Maximisation Bias and Double Q-Learning

**The problem.** Q-learning uses $\max_a Q(s, a)$ as the target. When Q-values are noisy estimates, $\max_a Q(s, a)$ is a **biased overestimate** of $\max_a Q^*(s, a)$. This is because $\mathbb{E}[\max_a Q(s, a)] \ge \max_a \mathbb{E}[Q(s, a)]$ (Jensen's inequality applied to the convex $\max$ function).

**Example.** Consider a state with 10 actions, all with true value 0. If $Q(s, a) \sim \mathcal{N}(0, 1)$ for each action, then $\mathbb{E}[\max_a Q(s, a)] \approx 1.54$ — a substantial overestimate of the true max (0).

**Double Q-learning** (van Hasselt, 2010) fixes this by maintaining two Q-functions $Q_1$ and $Q_2$. Use $Q_1$ to **select** the best action, and $Q_2$ to **evaluate** it:

$$Q_1(S, A) \leftarrow Q_1(S, A) + \alpha \left[R + \gamma Q_2\!\left(S', \arg\max_a Q_1(S', a)\right) - Q_1(S, A)\right]$$

With probability 0.5, swap the roles of $Q_1$ and $Q_2$. Since the selection and evaluation use different (independent) estimates, the overestimation bias is eliminated.

### 8.5 Convergence of Q-Learning

**Theorem (Watkins & Dayan, 1992).** Tabular Q-learning converges to $Q^*$ with probability 1, provided:

1. All state-action pairs are visited infinitely often.
2. The learning rate $\alpha_t(s, a)$ satisfies: $\sum_t \alpha_t(s, a) = \infty$ and $\sum_t \alpha_t^2(s, a) < \infty$.

_Proof sketch._ Define the noise $\omega_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - \mathbb{E}[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) \mid S_t, A_t]$. The Q-learning update is:

$$Q_{t+1}(S_t, A_t) = (1 - \alpha_t) Q_t(S_t, A_t) + \alpha_t (\mathcal{T}^* Q_t)(S_t, A_t) + \alpha_t \omega_t$$

This is a stochastic approximation of the Bellman optimality operator $\mathcal{T}^*$, which is a $\gamma$-contraction (Section 4.3). Convergence then follows from the Robbins-Monro theorem for stochastic fixed-point iterations with contractive operators.

**Practical learning rates.** The theoretical $\alpha_t = 1/N_t(s, a)$ (visit count) converges but is too slow. In practice, a fixed $\alpha \in [0.01, 0.1]$ is used, sacrificing the theoretical guarantee of convergence to $Q^*$ in exchange for faster adaptation and the ability to track non-stationary environments.

---

## 9. Function Approximation and Deep RL

### 9.1 Why Tables Fail

Tabular methods store one value per state (or state-action pair). For a gridworld with $100 \times 100$ cells and 4 actions, we need $40{,}000$ entries — feasible. For Atari from pixels ($210 \times 160 \times 3$ RGB frames), the state space has $256^{100{,}800} \approx 10^{242{,}000}$ states — more than the atoms in the observable universe. For LLMs, the state (prompt + generated tokens) is a variable-length sequence from a vocabulary of $32{,}000$ tokens — effectively infinite.

Function approximation replaces the table with a parameterised function $\hat{V}(s; \mathbf{w})$ or $\hat{Q}(s, a; \mathbf{w})$ that generalises across states, sharing information between similar states.

### 9.2 Linear Function Approximation

The simplest approximation: $\hat{V}(s; \mathbf{w}) = \mathbf{w}^\top \boldsymbol{\phi}(s)$, where $\boldsymbol{\phi}(s) \in \mathbb{R}^d$ is a feature vector for state $s$.

**TD(0) with linear approximation:**

$$\mathbf{w} \leftarrow \mathbf{w} + \alpha \, \delta_t \, \boldsymbol{\phi}(S_t)$$

where $\delta_t = R_{t+1} + \gamma \mathbf{w}^\top \boldsymbol{\phi}(S_{t+1}) - \mathbf{w}^\top \boldsymbol{\phi}(S_t)$.

**Convergence guarantee.** Linear TD(0) converges to a fixed point $\mathbf{w}^*$ satisfying $\mathbf{w}^* = A^{-1} \mathbf{b}$ where $A = \mathbb{E}[\boldsymbol{\phi}(S_t)(\boldsymbol{\phi}(S_t) - \gamma \boldsymbol{\phi}(S_{t+1}))^\top]$ and $\mathbf{b} = \mathbb{E}[R_{t+1} \boldsymbol{\phi}(S_t)]$. The error is bounded: $\lVert V_{\mathbf{w}^*} - V^\pi \rVert_\mu \le \frac{1}{\sqrt{1 - \gamma^2}} \min_\mathbf{w} \lVert V_\mathbf{w} - V^\pi \rVert_\mu$, where $\mu$ is the stationary distribution. Linear TD finds a value function that is within a factor $\frac{1}{\sqrt{1-\gamma^2}}$ of the best possible linear approximation.

### 9.3 The Deadly Triad

Convergence of TD with function approximation is not guaranteed in general. Divergence can occur when all three of the following are present simultaneously:

1. **Function approximation** (instead of tabular representation)
2. **Bootstrapping** (TD-style updates using value estimates)
3. **Off-policy learning** (learning about a policy different from the one generating data)

Any two are fine: MC + function approximation + off-policy converges (no bootstrapping). TD + tabular + off-policy converges (no function approximation). TD + function approximation + on-policy converges (no off-policy). But all three together can diverge — the famous **Baird counterexample** demonstrates this for linear function approximation with off-policy TD.

**Why this matters for deep RL.** DQN uses all three components: neural function approximation, bootstrapping (TD target), and off-policy learning (experience replay). It should diverge! The key stabilising innovations are experience replay and target networks (Section 9.5).

### 9.4 Deep Q-Networks (DQN)

**DQN** (Mnih et al., 2013, 2015) was the breakthrough that launched deep RL. A neural network $Q(s, a; \theta)$ takes raw pixel frames as input and outputs Q-values for each action.

**Loss function:**

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[\left(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta)\right)^2\right]$$

where $\mathcal{D}$ is the replay buffer and $\theta^-$ is the target network parameters.

### 9.5 Experience Replay and Target Networks

**Experience replay** stores transitions $(s, a, r, s')$ in a buffer and samples mini-batches uniformly for training. This breaks temporal correlations (consecutive transitions are highly correlated, violating the i.i.d. assumption of SGD) and improves data efficiency (each transition is used multiple times).

**Target network** $Q(s, a; \theta^-)$ is a copy of the Q-network with **frozen** parameters, updated periodically: $\theta^- \leftarrow \theta$ every $C$ steps. This stabilises training by preventing the moving target problem: without a target network, the TD target $r + \gamma \max_{a'} Q(s', a'; \theta)$ changes with every gradient step, creating a non-stationary regression problem.

**Soft updates** (Polyak averaging) provide smoother target evolution:

$$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$$

with $\tau \ll 1$ (typically $\tau = 0.005$).

### 9.6 Rainbow: Combining Improvements

**Rainbow** (Hessel et al., 2018) combines six DQN improvements, each addressing a different limitation:

| Component               | Addresses                     | Key idea                                    |
| ----------------------- | ----------------------------- | ------------------------------------------- |
| Double DQN              | Maximisation bias             | Decouple selection and evaluation           |
| Prioritised replay      | Uniform sampling inefficiency | Sample high-TD-error transitions more often |
| Dueling architecture    | Value vs advantage            | $Q(s,a) = V(s) + A(s,a) - \bar{A}$          |
| Multi-step returns      | Single-step bootstrap bias    | Use $n$-step TD targets                     |
| Distributional RL (C51) | Scalar value limitation       | Learn the full return distribution          |
| Noisy nets              | $\epsilon$-greedy exploration | Learned exploration via parameter noise     |

The combination outperforms any individual component — a rare example of orthogonal improvements stacking multiplicatively.

---

## 10. Policy Gradient Methods

Value-based methods (Q-learning, DQN) learn a value function and derive a policy indirectly. Policy gradient methods directly parameterise and optimise the policy $\pi_\theta$.

### 10.1 Policy Parameterisation

**Discrete actions (softmax policy):**

$$\pi_\theta(a \mid s) = \frac{\exp(h(s, a; \theta))}{\sum_{a'} \exp(h(s, a'; \theta))}$$

where $h(s, a; \theta)$ are action preferences (logits), computed by a neural network.

**Continuous actions (Gaussian policy):**

$$\pi_\theta(a \mid s) = \mathcal{N}(a \mid \mu_\theta(s), \sigma_\theta^2(s))$$

where both the mean $\mu_\theta(s)$ and variance $\sigma_\theta^2(s)$ are outputs of a neural network.

**For AI:** An LLM uses the softmax policy parameterisation directly — the final layer outputs logits over the vocabulary, and softmax converts them to a probability distribution over next tokens.

### 10.2 The Policy Gradient Theorem

**Objective:** Maximise expected return $J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau)]$ where $R(\tau) = \sum_{t=0}^{T} \gamma^t r_t$.

**The problem:** The expectation is over trajectories $\tau$ sampled from $p_\theta(\tau)$, which depends on $\theta$ through the policy. We cannot simply differentiate through the sampling process.

**Policy gradient theorem** (Sutton et al., 1999):

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t \mid S_t) \, Q^{\pi_\theta}(S_t, A_t)\right]$$

_Proof._ Start from $J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau)] = \sum_\tau p_\theta(\tau) R(\tau)$.

$$\nabla_\theta J = \sum_\tau \nabla_\theta p_\theta(\tau) R(\tau) = \sum_\tau p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} R(\tau)$$

Using the **log-derivative trick**: $\frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} = \nabla_\theta \log p_\theta(\tau)$:

$$= \mathbb{E}_{\tau \sim p_\theta}\left[R(\tau) \nabla_\theta \log p_\theta(\tau)\right]$$

Now, $p_\theta(\tau) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)$, so:

$$\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

The transition dynamics $P$ and initial state $p(s_0)$ do not depend on $\theta$, so they vanish from the gradient. This is remarkable: we can optimise the policy without knowing the environment dynamics.

Finally, using the fact that future actions do not affect past rewards:

$$\nabla_\theta J = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t \mid S_t) \, G_t\right]$$

where $G_t = \sum_{k=t}^{T} \gamma^{k-t} R_{k+1}$ is the return from time $t$. A more refined version uses $Q^{\pi_\theta}(S_t, A_t)$ in place of $G_t$.

### 10.3 REINFORCE

**REINFORCE** (Williams, 1992) is the simplest policy gradient algorithm. It uses Monte Carlo returns as an unbiased estimate of $Q^\pi$:

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \gamma^t G_t \nabla_\theta \log \pi_\theta(A_t \mid S_t)$$

**Properties:**

- Unbiased gradient estimate.
- Very high variance — $G_t$ accumulates all randomness from time $t$ onward.
- Requires complete episodes (cannot learn online).
- Sample inefficient — each trajectory is used once.

### 10.4 Variance Reduction: Baselines

**Theorem.** Subtracting any function $b(s)$ that depends only on the state (not the action) from the return does not change the expected gradient:

$$\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a \mid s) \cdot b(s)\right] = 0$$

_Proof._ $\sum_a \nabla_\theta \pi_\theta(a \mid s) \cdot b(s) = b(s) \nabla_\theta \sum_a \pi_\theta(a \mid s) = b(s) \nabla_\theta 1 = 0$. $\square$

**Practical impact:** Using $b(s) = V^\pi(s)$ converts the policy gradient to:

$$\nabla_\theta J = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(A_t \mid S_t) \, A^\pi(S_t, A_t)\right]$$

where $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$ is the **advantage**. Since advantages are centred around zero (positive for above-average actions, negative for below-average), the gradient signal is much less noisy than using raw returns.

**Optimal baseline** (minimising variance): $b^*(s) = \frac{\mathbb{E}[\lVert \nabla_\theta \log \pi \rVert^2 Q^\pi]}{\mathbb{E}[\lVert \nabla_\theta \log \pi \rVert^2]}$. In practice, $V^\pi(s)$ is nearly optimal and much simpler.

### 10.5 Natural Policy Gradient

The standard gradient $\nabla_\theta J$ depends on the parameterisation — reparameterising $\theta$ changes the gradient direction even though the policy distribution is unchanged. The **natural policy gradient** fixes this by using the Fisher information metric:

$$\tilde{\nabla}_\theta J = F^{-1} \nabla_\theta J$$

where $F = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta \, \nabla_\theta \log \pi_\theta^\top]$ is the Fisher information matrix. The natural gradient moves in the direction of steepest ascent in the space of distributions (measured by KL divergence), not in parameter space.

**Connection to trust regions:** A natural gradient step of size $\epsilon$ produces a policy update with $D_{\text{KL}}(\pi_{\theta_{\text{old}}} \| \pi_{\theta_{\text{new}}}) \approx \epsilon$ — a controlled, bounded change in the policy distribution. This is the theoretical foundation for TRPO (Section 11.3).

---

## 11. Actor-Critic and PPO

### 11.1 Actor-Critic Architecture

Actor-critic methods combine policy gradients (actor) with value function estimation (critic), getting the best of both worlds:

- **Actor** $\pi_\theta(a \mid s)$: the policy, updated via policy gradients.
- **Critic** $V_\phi(s)$ (or $Q_\phi(s, a)$): a learned value function, used to compute advantage estimates for the actor.

The critic replaces the high-variance Monte Carlo return $G_t$ with a lower-variance (but biased) TD estimate. The simplest actor-critic uses the one-step TD error as the advantage:

$$\hat{A}_t = R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t) = \delta_t$$

**Actor update:** $\theta \leftarrow \theta + \alpha_\theta \, \delta_t \, \nabla_\theta \log \pi_\theta(A_t \mid S_t)$

**Critic update:** $\phi \leftarrow \phi + \alpha_\phi \, \delta_t \, \nabla_\phi V_\phi(S_t)$

**A2C (Advantage Actor-Critic)** is the synchronous version where multiple parallel environments collect data simultaneously, reducing variance through averaging.

### 11.2 Generalised Advantage Estimation

**GAE** (Schulman et al., 2016) applies the TD(lambda) idea to advantage estimation:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t)$.

This is computed recursively: $\hat{A}_t = \delta_t + \gamma \lambda \, \hat{A}_{t+1}$ (starting from $\hat{A}_T = 0$).

- $\lambda = 0$: $\hat{A}_t = \delta_t$ — one-step TD advantage. Low variance, high bias.
- $\lambda = 1$: $\hat{A}_t = G_t - V_\phi(S_t)$ — MC advantage. No bias, high variance.
- $\lambda \in (0, 1)$: Smooth tradeoff. Typical choice: $\lambda = 0.95$.

GAE is used in nearly all modern policy gradient implementations, including PPO for RLHF.

### 11.3 Trust Regions and TRPO

The fundamental problem with vanilla policy gradients: a learning rate too large can cause a catastrophic policy update — the new policy performs terribly, generating bad data, causing further degradation (a death spiral).

**Trust Region Policy Optimisation** (TRPO, Schulman et al., 2015) constrains the policy update to stay within a trust region:

$$\max_\theta \; \mathbb{E}_{s \sim \rho_{\theta_{\text{old}}}}\left[\frac{\pi_\theta(a \mid s)}{\pi_{\theta_{\text{old}}}(a \mid s)} \hat{A}_t\right] \quad \text{s.t.} \quad \mathbb{E}_s\left[D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot \mid s) \,\|\, \pi_\theta(\cdot \mid s))\right] \le \delta$$

The KL constraint ensures the new policy is close to the old one in distribution space. TRPO solves this constrained optimisation using conjugate gradients and line search — effective but computationally expensive.

### 11.4 PPO: Clipped Surrogate Objective

**PPO** (Schulman et al., 2017) achieves similar stability to TRPO with a much simpler implementation. Instead of a KL constraint, PPO clips the objective:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[\min\!\left(r_t(\theta) \hat{A}_t, \; \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(A_t \mid S_t)}{\pi_{\theta_{\text{old}}}(A_t \mid S_t)}$ is the importance ratio and $\epsilon$ is the clipping parameter (typically 0.1-0.2).

**How clipping works:**

- When $\hat{A}_t > 0$ (good action): We want to increase $\pi_\theta(A_t \mid S_t)$, which increases $r_t$. But the clip prevents $r_t$ from exceeding $1 + \epsilon$ — stopping the optimisation from making the action too much more likely.
- When $\hat{A}_t < 0$ (bad action): We want to decrease $\pi_\theta(A_t \mid S_t)$, decreasing $r_t$. The clip prevents $r_t$ from falling below $1 - \epsilon$.

```text
PPO CLIPPING MECHANISM
════════════════════════════════════════════════════════════════════════

  L^CLIP(r)                    L^CLIP(r)
  (A > 0: good action)        (A < 0: bad action)

  │        ┌──────             │
  │       ╱                    │──────┐
  │      ╱                     │       ╲
  │     ╱                      │        ╲
  ──┼────┼──────── r           ──┼────┼──────── r
  │  1-ε  1  1+ε               │  1-ε  1  1+ε
  │                            │

  Gradient stops when r > 1+ε   Gradient stops when r < 1-ε
  (prevents too much increase)  (prevents too much decrease)

════════════════════════════════════════════════════════════════════════
```

### 11.5 Implementation Details That Matter

PPO's success depends on implementation details not in the paper but critical in practice:

1. **Advantage normalisation:** $\hat{A}_t \leftarrow (\hat{A}_t - \bar{A}) / (\sigma_A + \epsilon)$ across the batch.
2. **Value function clipping:** Similar clipping applied to the value loss.
3. **Entropy bonus:** $L = L^{\text{CLIP}} + c_1 L^{\text{VF}} + c_2 H(\pi_\theta)$. The entropy term encourages exploration.
4. **Multiple epochs:** PPO reuses each batch of data for 3-10 gradient updates before collecting new data — much more sample-efficient than REINFORCE.
5. **Mini-batch SGD:** Each epoch shuffles and splits the batch into mini-batches.

**For AI:** PPO is the standard algorithm for RLHF in LLMs. OpenAI's InstructGPT, Anthropic's Claude, and Meta's LLaMA-2 all use PPO (or variants) to fine-tune from human preference data.

---

## 12. Continuous Action Spaces

### 12.1 Deterministic Policy Gradient

For continuous action spaces $\mathcal{A} \subseteq \mathbb{R}^d$, we cannot enumerate actions. The **deterministic policy gradient** (DPG, Silver et al., 2014) provides a gradient for deterministic policies $a = \mu_\theta(s)$:

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\mu}\left[\nabla_\theta \mu_\theta(s) \, \nabla_a Q^\mu(s, a)\big|_{a = \mu_\theta(s)}\right]$$

This is a chain rule: $\nabla_\theta Q = (\nabla_a Q)(\nabla_\theta \mu)$. The critic provides $\nabla_a Q$ (how value changes with action), and the actor provides $\nabla_\theta \mu$ (how action changes with parameters).

**Key advantage:** DPG does not require integrating over actions (no $\sum_a$ or $\int_a$), which would be intractable in high-dimensional continuous spaces.

### 12.2 DDPG and TD3

**DDPG** (Deep Deterministic Policy Gradient, Lillicrap et al., 2016) combines DPG with DQN techniques:

- Experience replay buffer for off-policy learning.
- Target networks $\mu_{\theta^-}$ and $Q_{\phi^-}$ with Polyak averaging.
- Exploration via Ornstein-Uhlenbeck noise added to the deterministic action.

**TD3** (Twin Delayed DDPG, Fujimoto et al., 2018) addresses DDPG's overestimation bias with three fixes:

1. **Twin critics:** Two Q-networks $Q_{\phi_1}$, $Q_{\phi_2}$. Target uses the minimum: $y = r + \gamma \min_{i=1,2} Q_{\phi_i^-}(s', \mu_{\theta^-}(s'))$. This is the continuous analogue of Double Q-learning.
2. **Delayed policy updates:** Update the actor less frequently than the critics (every $d$ steps). This lets the critics stabilise before the actor changes.
3. **Target policy smoothing:** Add noise to the target action: $a' = \mu_{\theta^-}(s') + \epsilon$, $\epsilon \sim \operatorname{clip}(\mathcal{N}(0, \sigma), -c, c)$. This acts as a regulariser, preventing the policy from exploiting narrow peaks in the Q-function.

### 12.3 Maximum Entropy RL

Standard RL maximises expected return. **Maximum entropy RL** adds an entropy bonus to the reward:

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}\left[r(S_t, A_t) + \alpha \, \mathcal{H}(\pi(\cdot \mid S_t))\right]$$

where $\mathcal{H}(\pi(\cdot \mid s)) = -\sum_a \pi(a \mid s) \log \pi(a \mid s)$ is the policy entropy and $\alpha > 0$ is the temperature parameter.

**Why entropy maximisation?**

1. **Exploration:** Higher entropy means more randomness in action selection — the agent explores more broadly.
2. **Robustness:** The optimal max-entropy policy captures all near-optimal behaviours rather than committing to a single mode. This makes it robust to perturbations.
3. **Composability:** Max-entropy policies can be composed to solve new tasks more easily (pre-training benefits).
4. **Connection to inference:** Max-entropy RL is equivalent to probabilistic inference, where the optimal policy is the posterior distribution over actions given the goal of maximising reward.

**Soft Bellman equation:**

$$Q^*(s, a) = r(s, a) + \gamma \, \mathbb{E}_{s'}\left[V^*(s')\right]$$

where $V^*(s) = \alpha \log \sum_a \exp(Q^*(s, a) / \alpha)$ — a soft maximum (log-sum-exp) rather than a hard maximum.

### 12.4 Soft Actor-Critic

**SAC** (Haarnoja et al., 2018) is the state-of-the-art off-policy algorithm for continuous control. It combines max-entropy RL with actor-critic:

**Critic update:** Minimise the soft Bellman residual:

$$\mathcal{L}_Q(\phi) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[\left(Q_\phi(s, a) - r - \gamma(Q_{\phi^-}(s', a') - \alpha \log \pi_\theta(a' \mid s'))\right)^2\right]$$

where $a' \sim \pi_\theta(\cdot \mid s')$.

**Actor update:** Minimise the expected KL divergence:

$$\mathcal{L}_\pi(\theta) = \mathbb{E}_{s \sim \mathcal{D}}\left[D_{\text{KL}}\!\left(\pi_\theta(\cdot \mid s) \;\big\|\; \frac{\exp(Q_\phi(s, \cdot) / \alpha)}{Z(s)}\right)\right]$$

In practice, this is computed using the reparameterisation trick: $a = f_\theta(\epsilon; s)$ where $\epsilon \sim \mathcal{N}(0, I)$, making the gradient flow through the sampling.

**Automatic temperature tuning:** SAC can automatically adjust $\alpha$ to maintain a target entropy $\bar{\mathcal{H}}$:

$$\mathcal{L}(\alpha) = \mathbb{E}_{a \sim \pi_\theta}\left[-\alpha \log \pi_\theta(a \mid s) - \alpha \bar{\mathcal{H}}\right]$$

This eliminates one of the most sensitive hyperparameters.

---

## 13. RLHF and LLM Alignment

This is arguably the most important section for understanding modern AI. Reinforcement Learning from Human Feedback (RLHF) is how raw language models are transformed into helpful, harmless, and honest assistants.

### 13.1 Reward Modelling from Human Preferences

Humans cannot write down a reward function for "be helpful and harmless." Instead, RLHF **learns** a reward function from human comparisons.

**Data collection.** Given a prompt $x$, generate two responses $y_1, y_2$ from the LLM. A human annotator chooses which response is better: $y_w \succ y_l$ (winner vs loser).

**Bradley-Terry model.** The probability that $y_1$ is preferred over $y_2$ is modelled as:

$$P(y_1 \succ y_2 \mid x) = \sigma(r_\psi(x, y_1) - r_\psi(x, y_2))$$

where $\sigma$ is the sigmoid function and $r_\psi(x, y)$ is a learned reward model (typically the LLM itself with a scalar head replacing the vocabulary head).

**Reward model loss:**

$$\mathcal{L}_{\text{RM}}(\psi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[\log \sigma(r_\psi(x, y_w) - r_\psi(x, y_l))\right]$$

This is binary cross-entropy on pairwise comparisons. The reward model learns to assign higher scalar scores to preferred responses.

**Challenges:** (1) Human annotators disagree — inter-annotator agreement is typically 70-80%. (2) The reward model can be wrong — it is a learned approximation. (3) The RL agent can exploit errors in the reward model (reward hacking).

### 13.2 The RLHF Pipeline

```text
RLHF PIPELINE
════════════════════════════════════════════════════════════════════════

  Step 1: Supervised Fine-Tuning (SFT)
  ┌──────────────────────────────────────────┐
  │  Pretrained LLM  →  Fine-tune on        │
  │                      human demonstrations│
  │                      (prompt, response)   │
  └──────────────────────────────────────────┘
                      │
                      ▼
  Step 2: Reward Model Training
  ┌──────────────────────────────────────────┐
  │  SFT model generates pairs (y1, y2)      │
  │  Humans label preferences: yw ≻ yl      │
  │  Train r_ψ via Bradley-Terry loss        │
  └──────────────────────────────────────────┘
                      │
                      ▼
  Step 3: RL Fine-Tuning (PPO)
  ┌──────────────────────────────────────────┐
  │  Optimise π_θ to maximise r_ψ(x,y)      │
  │  Subject to KL constraint from π_SFT    │
  │  Using PPO with GAE                      │
  └──────────────────────────────────────────┘

════════════════════════════════════════════════════════════════════════
```

### 13.3 PPO for Language Models

The RLHF objective is:

$$\max_\theta \; \mathbb{E}_{x \sim \mathcal{D}, \, y \sim \pi_\theta(\cdot \mid x)} \left[r_\psi(x, y)\right] - \beta \, D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

where $\pi_{\text{ref}}$ is the SFT model (reference policy) and $\beta$ controls the KL penalty.

**Per-token reward shaping.** The reward $r_\psi(x, y)$ is a single scalar for the entire sequence. To apply PPO (which operates per-step), the reward is distributed: $r_t = 0$ for all tokens except the last, where $r_T = r_\psi(x, y)$. The KL penalty $-\beta \log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}$ is added at each token.

**In RL terms:**

- **State** $s_t$: the prompt $x$ plus tokens generated so far $y_{<t}$.
- **Action** $a_t$: the next token $y_t \in \mathcal{V}$.
- **Policy** $\pi_\theta(a_t \mid s_t)$: the LLM's next-token distribution.
- **Reward** $r_t$: the KL penalty at each step, plus the reward model score at the final step.
- **Value function** $V_\phi(s_t)$: a learned critic (often a separate head on the LLM).

### 13.4 DPO: Direct Preference Optimisation

**DPO** (Rafailov et al., 2023) eliminates the need for a separate reward model and RL training loop. The key insight: the optimal policy under the KL-constrained RLHF objective has a closed-form solution:

$$\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x) \exp(r(x, y) / \beta)}{Z(x)}$$

Solving for the reward: $r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$.

Substituting into the Bradley-Terry preference model and cancelling the partition function $Z(x)$:

$$P(y_w \succ y_l \mid x) = \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$$

**DPO loss:**

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

**Advantages of DPO:**

- No reward model training needed.
- No RL loop (PPO is complex and unstable for LLMs).
- Simple supervised loss — just binary cross-entropy on preference pairs.
- Mathematically equivalent to RLHF under the Bradley-Terry model.

**Disadvantages:** DPO can overfit to the preference dataset, and it implicitly assumes the Bradley-Terry model is correct. It also lacks the online data collection that PPO provides.

### 13.5 KL Divergence Constraint

The KL penalty $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$ is essential for preventing **reward hacking** — without it, the policy would find degenerate outputs that exploit errors in the reward model (e.g., generating nonsensical text that happens to score high).

$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\right] = \sum_{y} \pi_\theta(y \mid x) \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

**The KL-reward tradeoff:** Increasing $\beta$ keeps $\pi_\theta$ close to $\pi_{\text{ref}}$ (safe but less optimised for reward). Decreasing $\beta$ allows more deviation (higher reward but risk of reward hacking). In practice, $\beta$ is tuned to achieve a KL divergence of roughly 5-15 nats from the reference policy.

### 13.6 Constitutional AI and RLAIF

**Constitutional AI** (Bai et al., 2022) replaces human annotators with AI feedback. Instead of humans comparing responses, an AI model evaluates responses against a set of principles (the "constitution"):

1. Generate responses to a prompt.
2. Ask an AI: "Which response better follows these principles?" (with specific rules about helpfulness, harmlessness, honesty).
3. Use the AI preferences to train a reward model (or DPO directly).

**RLAIF (RL from AI Feedback)** generalises this: any AI-generated preference signal. This scales beyond human annotation capacity and enables iterative improvement where the evaluating model improves alongside the trained model.

**GRPO (Group Relative Policy Optimisation, DeepSeek, 2024)** eliminates the critic entirely. Instead of learning a value baseline, GRPO samples a group of responses per prompt and uses the group mean reward as the baseline — a simple Monte Carlo approach that avoids the complexity of value function training.

---

## 14. Exploration Strategies

The **exploration-exploitation tradeoff** is fundamental: the agent must exploit known high-reward actions to accumulate reward, but must also explore unknown actions to discover potentially better strategies.

### 14.1 Epsilon-Greedy and Boltzmann Exploration

**Epsilon-greedy:** With probability $1 - \epsilon$, take the greedy action $\arg\max_a Q(s, a)$. With probability $\epsilon$, take a random action.

$$\pi(a \mid s) = \begin{cases} 1 - \epsilon + \epsilon / |\mathcal{A}| & \text{if } a = \arg\max_{a'} Q(s, a') \\ \epsilon / |\mathcal{A}| & \text{otherwise} \end{cases}$$

Simple but undirected — random exploration wastes effort on actions already known to be bad.

**Boltzmann (softmax) exploration:** Choose actions proportionally to their exponentiated Q-values:

$$\pi(a \mid s) = \frac{\exp(Q(s, a) / \tau)}{\sum_{a'} \exp(Q(s, a') / \tau)}$$

Temperature $\tau$ controls exploration: high $\tau \to$ uniform (explore), low $\tau \to$ greedy (exploit). Unlike $\epsilon$-greedy, Boltzmann exploration is **directed**: it prefers higher-valued actions even during exploration.

### 14.2 UCB and Optimism

**Upper Confidence Bound** (UCB) implements the principle of "optimism in the face of uncertainty":

$$A_t = \arg\max_a \left[Q(s, a) + c \sqrt{\frac{\ln t}{N_t(s, a)}}\right]$$

The bonus term $c\sqrt{\ln t / N_t(s, a)}$ is large for rarely-tried actions and decreases as they are explored. This ensures every action is tried infinitely often, while focusing on promising actions.

**Theoretical guarantee:** UCB achieves logarithmic regret: $\text{Regret}(T) = O(\sqrt{KT \ln T})$ where $K$ is the number of actions — optimal up to the $\sqrt{\ln T}$ factor.

**Connection to Bayesian methods:** UCB can be derived from a Bayesian perspective where the bonus approximates the upper end of a confidence interval for the true Q-value.

### 14.3 Thompson Sampling

**Thompson sampling** maintains a posterior distribution over Q-values and samples from it:

1. For each action $a$, maintain a posterior $P(Q(s, a) \mid \text{data})$.
2. Sample $\tilde{Q}(s, a) \sim P(Q(s, a) \mid \text{data})$ for each action.
3. Take action $a = \arg\max_a \tilde{Q}(s, a)$.

For Bernoulli rewards (each action gives reward 1 with probability $p_a$), the posterior is $\text{Beta}(\alpha_a, \beta_a)$ where $\alpha_a$ counts successes and $\beta_a$ counts failures.

Thompson sampling naturally balances exploration and exploitation: uncertain actions have wide posteriors, so they are occasionally sampled with high values (triggering exploration); well-understood good actions have concentrated posteriors near their true values (exploitation).

**Theoretical guarantee:** Thompson sampling achieves Bayesian-optimal regret bounds — it is optimal among all algorithms that use the given prior.

### 14.4 Intrinsic Motivation and Curiosity

For sparse-reward environments, external rewards are insufficient to guide exploration. **Intrinsic motivation** provides additional reward for visiting novel or surprising states.

**Prediction error (curiosity):** Train a forward dynamics model $\hat{s}_{t+1} = f_\phi(s_t, a_t)$. The intrinsic reward is the prediction error:

$$r_t^i = \lVert f_\phi(s_t, a_t) - s_{t+1} \rVert^2$$

High prediction error means the state transition was surprising — the agent is in unfamiliar territory. This drives the agent toward novel states.

**Count-based exploration:** Maintain visit counts $N(s)$ and provide a bonus:

$$r_t^+ = \frac{\beta}{\sqrt{N(s_t)}}$$

Rarely visited states get larger bonuses. For continuous state spaces, pseudo-counts or density models approximate the visit count.

**Random Network Distillation (RND, Burda et al., 2019):** Use a fixed random network $f(s)$ and a trainable network $\hat{f}_\phi(s)$. The intrinsic reward is $\lVert f(s) - \hat{f}_\phi(s) \rVert^2$. For frequently visited states, $\hat{f}$ learns to match $f$, reducing the bonus. For novel states, the error is high.

---

## 15. Model-Based RL

All methods so far are **model-free**: they learn value functions or policies directly from experience without building a model of the environment. **Model-based RL** learns a dynamics model $\hat{P}(s' \mid s, a)$ and uses it for planning.

### 15.1 Learned Dynamics Models

A dynamics model predicts the next state (and optionally reward) given the current state and action:

$$\hat{s}_{t+1} = f_\phi(s_t, a_t) \qquad \hat{r}_{t+1} = g_\phi(s_t, a_t)$$

For deterministic models, $f_\phi$ is a neural network trained to minimise $\lVert f_\phi(s_t, a_t) - s_{t+1} \rVert^2$. For stochastic models, $f_\phi$ outputs parameters of a distribution (e.g., Gaussian mean and variance).

**Compounding error.** Multi-step predictions $\hat{s}_{t+k} = f_\phi(\hat{s}_{t+k-1}, a_{t+k-1})$ accumulate errors — each step's prediction error feeds into the next step's input. After $H$ steps, the error can grow exponentially. This is the fundamental challenge of model-based RL: the model is useful for short-horizon planning but unreliable for long-horizon predictions.

### 15.2 Dyna-Q: Integrating Learning and Planning

**Dyna-Q** (Sutton, 1991) interleaves real experience with simulated experience from the learned model:

1. Take action $a$ in real environment, observe $(s, a, r, s')$.
2. **Direct RL:** Update Q-values from the real transition.
3. **Model learning:** Update the dynamics model with $(s, a) \to (r, s')$.
4. **Planning:** Repeat $k$ times: sample a previously visited $(s, a)$, simulate $(r, s')$ from the model, update Q-values.

The key insight: each real interaction generates $k$ additional learning updates through the model, dramatically improving sample efficiency. The planning steps are computationally cheap (model inference) compared to real environment interactions (which may involve expensive simulations or physical systems).

### 15.3 Model Predictive Control

**Model Predictive Control (MPC)** uses the learned model for online planning:

$$a_t^* = \arg\max_{\{a_t, \ldots, a_{t+H-1}\}} \sum_{k=0}^{H-1} \gamma^k r(\hat{s}_{t+k}, a_{t+k})$$

At each time step, plan the best action sequence over a horizon $H$, execute only the first action, observe the actual next state, and re-plan. This **receding horizon** approach mitigates compounding model errors because the plan is always re-grounded in the true state.

**Solving the planning problem:** For discrete actions, tree search (Monte Carlo Tree Search as in AlphaGo). For continuous actions, sampling-based methods (CEM — Cross-Entropy Method, random shooting) or gradient-based optimisation through the differentiable model.

### 15.4 World Models

**World models** (Ha & Schmidhuber, 2018) learn a compact latent representation of the environment dynamics:

1. **Encoder** $z = \text{enc}(s)$: compress high-dimensional observations (images) into a latent state $z$.
2. **Dynamics model** $z' = f(z, a)$: predict the next latent state.
3. **Reward predictor** $\hat{r} = g(z, a)$: predict reward from latent state.
4. **Policy** $\pi(a \mid z)$: act in latent space.

The policy can be trained entirely "in the dream" — in the latent space using the learned dynamics — without interacting with the real environment. This is extremely sample-efficient when the model is accurate.

**Dreamer** (Hafner et al., 2020, 2023) is the state-of-the-art world model approach, achieving competitive performance with model-free methods while using 10-100x fewer environment interactions. DreamerV3 generalises across diverse domains (Atari, robotics, Minecraft) with a single set of hyperparameters.

**For AI:** LLMs themselves can be viewed as world models — they learn a compressed model of language dynamics from internet text. The connection between world models in RL and next-token prediction in LLMs is an active area of research.

---

## 16. Common Mistakes

1. **Confusing on-policy and off-policy.** SARSA is on-policy (evaluates the policy being followed); Q-learning is off-policy (evaluates the greedy policy regardless of what is followed). Using off-policy data with on-policy algorithms (without importance sampling) gives wrong value estimates.

2. **Ignoring the discount factor in return calculations.** The return is $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$, not $\sum R$. Forgetting $\gamma$ in implementations causes value estimates to diverge.

3. **Using $\max$ in SARSA.** SARSA uses $Q(S_{t+1}, A_{t+1})$ (the action actually taken); Q-learning uses $\max_a Q(S_{t+1}, a)$. Swapping them changes the algorithm's fundamental character.

4. **Not normalising advantages in PPO.** Raw advantages can vary by orders of magnitude between batches, causing training instability. Always normalise: $\hat{A} \leftarrow (\hat{A} - \mu) / (\sigma + \epsilon)$.

5. **Forgetting the KL penalty in RLHF.** Without the KL constraint, the policy collapses to degenerate outputs that exploit reward model errors. The reference policy anchor is not optional.

6. **Confusing the Bellman equation with a learning algorithm.** The Bellman equation is a mathematical identity that the true value function satisfies. TD learning and value iteration are algorithms that use the Bellman equation as an update rule to converge to the true values.

7. **Applying tabular convergence proofs to deep RL.** Tabular Q-learning converges to $Q^*$ under mild conditions. DQN has no such guarantee — the deadly triad (function approximation + bootstrapping + off-policy) means divergence is possible. Experience replay and target networks are heuristic stabilisers, not convergence guarantees.

8. **Reward hacking through poor reward design.** The reward function defines the task. If $r = \text{score}$ in a game, the agent maximises score. If $r = \text{clicks}$ on a website, the agent maximises clicks (not user satisfaction). Reward misspecification is the most common failure mode in applied RL.

---

## 17. Exercises

1. **Bellman equation as linear system.** Consider a 4-state MDP with given transition probabilities and rewards. Write the Bellman expectation equation for a uniform random policy as a system of linear equations $\mathbf{v} = \mathbf{r}^\pi + \gamma P^\pi \mathbf{v}$. Solve by matrix inversion and verify by running iterative policy evaluation.

2. **TD(0) vs Monte Carlo convergence.** Implement both TD(0) and first-visit MC prediction on a 5-state random walk. Run 100 episodes and plot the RMSE vs episodes for both methods. Explain which converges faster and why.

3. **Q-learning vs SARSA on cliff walking.** Implement both algorithms on the $4 \times 12$ cliff walking environment. Plot the reward per episode during training for both. Show that Q-learning finds the optimal path but SARSA finds a safer path, and explain why.

4. **Policy gradient theorem derivation.** Starting from $J(\theta) = \mathbb{E}_{\tau \sim p_\theta}[R(\tau)]$, derive the REINFORCE gradient estimator step by step. Show that the transition dynamics $P(s' \mid s, a)$ cancel out. Then prove that subtracting a state-dependent baseline $b(s)$ does not change the expected gradient.

5. **PPO clipping analysis.** For the PPO clipped objective, draw the effective loss landscape $L^{\text{CLIP}}(r)$ as a function of the importance ratio $r$ for both $\hat{A} > 0$ and $\hat{A} < 0$. Identify the regions where the gradient is zero (clipped) and non-zero (active). Explain how this prevents catastrophically large updates.

6. **Double Q-learning on maximisation bias MDP.** Implement the maximisation bias example from Sutton & Barto: a two-state MDP where state B has 10 actions with $\mathcal{N}(-0.1, 1)$ rewards. Run standard Q-learning and Double Q-learning for 300 episodes. Plot the percentage of time each algorithm chooses the optimal action (left from state A).

7. **DPO loss derivation.** Starting from the KL-constrained RLHF objective, derive the closed-form optimal policy. Then substitute into the Bradley-Terry preference model to derive the DPO loss. Show that the partition function $Z(x)$ cancels.

8. **GAE computation.** Given a trajectory of 5 time steps with rewards $[0, 0, 0, 1, 0]$ and value estimates $[0.5, 0.4, 0.3, 0.8, 0.1]$, compute the GAE advantages for $\lambda = 0$, $\lambda = 0.5$, and $\lambda = 1$. Verify that $\lambda = 0$ gives the TD error and $\lambda = 1$ gives the MC advantage.

9. **Experience replay analysis.** Explain why experience replay (a) breaks temporal correlations, (b) improves sample efficiency, and (c) enables off-policy learning. Then explain why prioritised experience replay (sampling high-TD-error transitions more often) requires importance sampling weights for unbiased updates.

10. **Contraction mapping proof.** Prove that the Bellman optimality operator $\mathcal{T}^*$ is a $\gamma$-contraction in the $\ell^\infty$ norm. Use this to prove that value iteration converges to $V^*$ and derive the convergence rate.

---

## 18. Why This Matters for AI (2026)

Reinforcement learning is no longer a niche subfield — it is the critical final training stage that transforms raw language models into useful AI assistants.

**RLHF is how LLMs are aligned.** GPT-4, Claude, Gemini, and LLaMA all use RL-based methods (PPO, DPO, or variants) to fine-tune from human preferences. Without RLHF, a pretrained LLM is a raw text generator that can produce toxic, harmful, or dishonest content. RLHF teaches it to be helpful, harmless, and honest — the most commercially and socially important application of RL to date.

**Reward modelling is the bottleneck.** The reward model is a learned approximation of human preferences, and it can be wrong. Reward hacking — where the RL agent exploits errors in the reward model — is an active research problem. Constitutional AI and RLAIF are attempts to scale reward modelling beyond human annotation capacity.

**DPO simplified the pipeline.** The discovery that RLHF can be reformulated as a simple supervised loss (DPO) has democratised alignment — smaller teams can now fine-tune models without the engineering complexity of PPO. However, online RL methods (PPO, GRPO) still tend to produce stronger results because they can explore beyond the training distribution.

**RL for reasoning.** DeepSeek-R1 and similar reasoning models use RL (specifically GRPO) to train LLMs to produce chain-of-thought reasoning. The reward model evaluates the final answer, and RL discovers effective reasoning strategies — a new application of RL to cognitive capabilities.

**Multi-agent RL and AI safety.** As AI systems interact with each other (e.g., multiple AI assistants negotiating), multi-agent RL becomes relevant. Game theory, mechanism design, and multi-agent training stability are emerging research frontiers with direct implications for AI safety.

---

## 19. Conceptual Bridge

**Where we have been:** This section developed RL from the MDP formalism through Bellman equations, dynamic programming, TD learning, Q-learning, policy gradients, and actor-critic methods, culminating in the modern RLHF pipeline for LLM alignment.

**Where this connects:**

| Concept                  | Connects to                                             |
| ------------------------ | ------------------------------------------------------- |
| Bellman equations        | Dynamic programming (Sections 09, 12-02)                |
| Policy gradient theorem  | Log-derivative trick, score functions (Section 06-03)   |
| Advantage function       | Variance reduction, control variates (Section 06-06)    |
| KL divergence constraint | Information theory (Chapter 13)                         |
| Softmax policy           | Boltzmann distribution, attention (Section 14-05)       |
| Function approximation   | Neural networks (Section 14-02)                         |
| Bradley-Terry model      | Logistic regression, maximum likelihood (Section 14-01) |
| Contraction mapping      | Fixed-point theory, spectral radius (Section 02-06)     |
| Importance sampling      | Monte Carlo methods, variance (Section 06-04)           |
| RLHF / DPO               | LLM training, alignment (applied ML)                    |

**What comes next:** Section 14-07 (Generative Models) develops the mathematics of VAEs, GANs, and diffusion models — another family of models where the training objective involves optimising an intractable expectation, solved by similar tricks (reparameterisation, score functions, variational bounds) to those used in policy gradients.

---

## References

1. Sutton, R. S. & Barto, A. G. (2018). _Reinforcement Learning: An Introduction_ (2nd ed.). MIT Press.
2. Watkins, C. J. & Dayan, P. (1992). Q-learning. _Machine Learning_, 8(3-4), 279-292.
3. Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. _Nature_, 518, 529-533.
4. Schulman, J. et al. (2015). Trust region policy optimization. _ICML_.
5. Schulman, J. et al. (2016). High-dimensional continuous control using generalised advantage estimation. _ICLR_.
6. Schulman, J. et al. (2017). Proximal policy optimization algorithms. _arXiv:1707.06347_.
7. Haarnoja, T. et al. (2018). Soft actor-critic: Off-policy maximum entropy deep RL. _ICML_.
8. Silver, D. et al. (2014). Deterministic policy gradient algorithms. _ICML_.
9. Christiano, P. et al. (2017). Deep reinforcement learning from human preferences. _NeurIPS_.
10. Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback. _NeurIPS_.
11. Rafailov, R. et al. (2023). Direct preference optimization: Your language model is secretly a reward model. _NeurIPS_.
12. Bai, Y. et al. (2022). Constitutional AI: Harmlessness from AI feedback. _arXiv:2212.08073_.
13. Fujimoto, S. et al. (2018). Addressing function approximation error in actor-critic methods. _ICML_.
14. Hessel, M. et al. (2018). Rainbow: Combining improvements in deep reinforcement learning. _AAAI_.
15. Hafner, D. et al. (2023). Mastering diverse domains through world models. _arXiv:2301.04104_.
