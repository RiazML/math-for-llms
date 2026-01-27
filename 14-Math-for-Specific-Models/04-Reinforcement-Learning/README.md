# Reinforcement Learning: Mathematical Foundations

## Overview

Reinforcement Learning (RL) provides a mathematical framework for sequential decision-making under uncertainty. An agent learns to maximize cumulative reward through interaction with an environment, balancing exploration and exploitation.

## 1. Markov Decision Process (MDP)

### Definition

An MDP is a tuple $(S, A, P, R, \gamma)$:

- $S$: State space
- $A$: Action space
- $P(s'|s, a)$: Transition dynamics
- $R(s, a, s')$: Reward function
- $\gamma \in [0, 1)$: Discount factor

### Markov Property

$$P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, \ldots) = P(s_{t+1} | s_t, a_t)$$

### Return

**Discounted return**:
$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

**Recursive**: $G_t = R_{t+1} + \gamma G_{t+1}$

## 2. Policies and Value Functions

### Policy

**Deterministic**: $a = \pi(s)$

**Stochastic**: $\pi(a|s) = P(A_t = a | S_t = s)$

### State-Value Function

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \Big| S_t = s\right]$$

### Action-Value Function

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

### Relationship

$$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)$$
$$Q^\pi(s, a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

## 3. Bellman Equations

### Bellman Expectation Equation

**For V**:
$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

**For Q**:
$$Q^\pi(s, a) = \sum_{s'} P(s'|s,a)\left[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s', a')\right]$$

### Bellman Optimality Equation

**Optimal Value**:
$$V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]$$

**Optimal Q**:
$$Q^*(s, a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \max_{a'} Q^*(s', a')]$$

### Optimal Policy

$$\pi^*(s) = \arg\max_a Q^*(s, a)$$

## 4. Dynamic Programming

### Policy Evaluation

Iteratively compute $V^\pi$:
$$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_k(s')]$$

### Policy Improvement

Greedy policy improvement:
$$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

### Policy Iteration

1. Initialize $\pi$
2. **Evaluate**: Compute $V^\pi$
3. **Improve**: $\pi' = \text{greedy}(V^\pi)$
4. If $\pi' = \pi$, done; else $\pi \leftarrow \pi'$, goto 2

### Value Iteration

Combine evaluation and improvement:
$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_k(s')]$$

**Convergence**: $\|V_{k+1} - V_k\|_\infty < \epsilon(1-\gamma)/\gamma$

## 5. Temporal Difference Learning

### TD(0) for Value Function

$$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

**TD Error**: $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$

### TD(λ) - Eligibility Traces

**Forward view**:
$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

where $G_t^{(n)}$ is the n-step return.

**Backward view**:
$$e_t(s) = \gamma \lambda e_{t-1}(s) + \mathbf{1}(S_t = s)$$
$$V(s) \leftarrow V(s) + \alpha \delta_t e_t(s)$$

## 6. Q-Learning

### Off-Policy TD Control

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$$

### Properties

- **Off-policy**: Learns optimal Q regardless of behavior policy
- **Convergence**: Converges to $Q^*$ under conditions on $\alpha$
- **ε-greedy exploration**: With probability ε, random action

### Double Q-Learning

Addresses maximization bias:
$$Q_1(S, A) \leftarrow Q_1(S, A) + \alpha[R + \gamma Q_2(S', \arg\max_a Q_1(S', a)) - Q_1(S, A)]$$

## 7. SARSA

### On-Policy TD Control

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

### SARSA(λ)

With eligibility traces:
$$e_t(s, a) = \gamma \lambda e_{t-1}(s, a) + \mathbf{1}(S_t = s, A_t = a)$$
$$Q(s, a) \leftarrow Q(s, a) + \alpha \delta_t e_t(s, a)$$

## 8. Policy Gradient Methods

### Policy Parameterization

$$\pi_\theta(a|s) = \frac{\exp(h(s, a, \theta))}{\sum_{a'} \exp(h(s, a', \theta))}$$

### Policy Gradient Theorem

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) G_t\right]$$

### REINFORCE

$$\theta \leftarrow \theta + \alpha \gamma^t G_t \nabla_\theta \log \pi_\theta(A_t|S_t)$$

### Baseline Subtraction

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) (Q^\pi(s, a) - b(s))\right]$$

Optimal baseline: $b(s) = V^\pi(s)$

## 9. Actor-Critic Methods

### Architecture

- **Actor**: Policy $\pi_\theta(a|s)$
- **Critic**: Value function $V_w(s)$ or $Q_w(s, a)$

### Advantage Function

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

**TD estimate**: $\hat{A}_t = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t)$

### A2C Update

**Critic**:
$$w \leftarrow w + \alpha_w \delta_t \nabla_w V_w(S_t)$$

**Actor**:
$$\theta \leftarrow \theta + \alpha_\theta \delta_t \nabla_\theta \log \pi_\theta(A_t|S_t)$$

### A3C (Asynchronous)

Multiple parallel actors update a shared model asynchronously.

## 10. Proximal Policy Optimization (PPO)

### Clipped Objective

$$L^{CLIP}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

### Properties

- **Trust region**: Limits policy change
- **Simple**: No second-order optimization
- **Sample efficient**: Multiple epochs per batch

## 11. Deep Q-Networks (DQN)

### Neural Q-Function

$$Q(s, a; \theta) \approx Q^*(s, a)$$

### Key Innovations

**Experience Replay**:
$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2\right]$$

**Target Network**: $\theta^-$ updated slowly

### Double DQN

$$y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$$

### Dueling DQN

$$Q(s, a; \theta) = V(s; \theta) + A(s, a; \theta) - \frac{1}{|A|}\sum_{a'} A(s, a'; \theta)$$

## 12. Continuous Action Spaces

### Deterministic Policy Gradient (DPG)

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s, a)|_{a=\mu_\theta(s)}\right]$$

### DDPG (Deep DPG)

Combines DPG with DQN techniques:

- Experience replay
- Target networks
- Soft updates: $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$

### SAC (Soft Actor-Critic)

**Maximum entropy objective**:
$$J(\pi) = \sum_t \mathbb{E}\left[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]$$

**Soft Bellman**:
$$Q(s, a) = r + \gamma \mathbb{E}[Q(s', a') - \alpha \log \pi(a'|s')]$$

## 13. Model-Based RL

### Learned Dynamics Model

$$\hat{s}_{t+1} = f_\phi(s_t, a_t)$$

### Planning with Model

**Model Predictive Control (MPC)**:
$$a_t^* = \arg\max_{\{a_t, \ldots, a_{t+H}\}} \sum_{k=0}^{H} \gamma^k r(\hat{s}_{t+k}, a_{t+k})$$

### Dyna-Q

Integrates learning and planning:

1. Take action, observe $(s, a, r, s')$
2. Update Q with real experience
3. Update model with $(s, a) \to (r, s')$
4. Repeat: sample from model, update Q

## 14. Exploration Strategies

### ε-Greedy

$$a = \begin{cases} \arg\max_a Q(s, a) & \text{with prob } 1-\epsilon \\ \text{random action} & \text{with prob } \epsilon \end{cases}$$

### UCB (Upper Confidence Bound)

$$a = \arg\max_a \left[Q(s, a) + c\sqrt{\frac{\ln t}{N_t(s, a)}}\right]$$

### Entropy Regularization

$$\pi^*(a|s) = \frac{\exp(Q(s, a)/\alpha)}{\sum_{a'} \exp(Q(s, a')/\alpha)}$$

### Intrinsic Motivation

**Curiosity**: $r^i_t = \|\hat{s}_{t+1} - s_{t+1}\|^2$

**Count-based**: $r^+ = \beta / \sqrt{N(s)}$

## Key Equations Summary

| Concept         | Equation                                                                |
| --------------- | ----------------------------------------------------------------------- | ------------------------- | ----------------------- |
| Bellman (V)     | $V(s) = \sum_a \pi(a                                                    | s)\sum\_{s'} P(s'         | s,a)[R + \gamma V(s')]$ |
| Bellman Opt     | $V^\*(s) = \max*a \sum*{s'} P(s'                                        | s,a)[R + \gamma V^*(s')]$ |
| TD(0)           | $V(s) \leftarrow V(s) + \alpha[R + \gamma V(s') - V(s)]$                |
| Q-Learning      | $Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma \max_a Q(s',a) - Q(s,a)]$ |
| Policy Gradient | $\nabla J = \mathbb{E}[\nabla \log \pi(a                                | s) Q(s,a)]$               |
| Advantage       | $A(s,a) = Q(s,a) - V(s)$                                                |
| PPO Clip        | $\min(r\hat{A}, \text{clip}(r, 1\pm\epsilon)\hat{A})$                   |

## ML Connections

| RL Concept      | Application                     |
| --------------- | ------------------------------- |
| Q-Learning      | Game playing, control           |
| Policy Gradient | Robotics, continuous control    |
| A3C             | Distributed training            |
| PPO             | Fine-tuning LLMs (RLHF)         |
| SAC             | Sample-efficient robot learning |
| Model-Based     | Data-efficient learning         |

## References

1. Sutton & Barto - "Reinforcement Learning: An Introduction"
2. Schulman et al. - "Proximal Policy Optimization"
3. Mnih et al. - "Human-level control through deep RL"
4. Haarnoja et al. - "Soft Actor-Critic"
