# Reinforcement Learning: Mathematical Foundations

[← Previous: Transformer Architecture](../05-Transformer-Architecture) | [Next: Generative Models →](../07-Generative-Models)

## Overview

Reinforcement Learning (RL) provides a mathematical framework for sequential decision-making under uncertainty. An agent learns to maximize cumulative reward through interaction with an environment, balancing exploration and exploitation.

## Files in This Section

| File                               | Description                              |
| ---------------------------------- | ---------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Interactive examples with visualizations |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions         |

## Why This Matters for Machine Learning

Reinforcement learning addresses a class of problems fundamentally different from supervised learning: the agent must discover which actions lead to reward through trial and error, while the data distribution itself depends on the agent's own behavior. The Bellman equations — recursive relationships between value functions at successive time steps — provide the mathematical backbone for this entire field. They decompose a long-horizon planning problem into a series of one-step consistency conditions, making dynamic programming, Q-learning, and policy optimization all possible.

Policy gradient methods bridge the gap between optimization and probability. The policy gradient theorem shows that the gradient of expected return equals the expected gradient of log-probability weighted by returns — a result that relies on the log-derivative trick from probability theory. This same trick appears in variational inference and score function estimators, making policy gradients a gateway to a broader family of gradient estimation techniques. Variance reduction through baselines (typically the value function) is what makes these estimators practical.

Temporal difference (TD) learning combines ideas from Monte Carlo estimation and dynamic programming, bootstrapping value estimates from other value estimates rather than waiting for complete episodes. This creates a spectrum from TD(0) (maximum bootstrapping) to Monte Carlo (no bootstrapping), unified by TD($\lambda$) and eligibility traces. Understanding this spectrum clarifies the bias-variance tradeoff in RL: more bootstrapping reduces variance but introduces bias from inaccurate value estimates.

## Chapter Roadmap

- **Markov Decision Processes**: States, actions, transitions, rewards, and the Markov property
- **Policies and Value Functions**: State-value $V^\pi$, action-value $Q^\pi$, and their relationships
- **Bellman Equations**: Expectation and optimality equations as recursive consistency conditions
- **Dynamic Programming**: Policy evaluation, policy improvement, and value iteration
- **Temporal Difference Learning**: TD(0), TD($\lambda$), and eligibility traces
- **Q-Learning**: Off-policy TD control, Double Q-learning, and convergence guarantees
- **SARSA**: On-policy TD control with eligibility traces
- **Policy Gradient Methods**: The policy gradient theorem, REINFORCE, and baseline subtraction
- **Actor-Critic Methods**: Combining policy and value function learning with advantage estimation
- **PPO**: Clipped surrogate objectives for stable policy updates
- **Deep Q-Networks**: Neural function approximation, experience replay, and target networks
- **Continuous Action Spaces**: DPG, DDPG, and Soft Actor-Critic with maximum entropy
- **Model-Based RL**: Learned dynamics, planning, and Dyna-Q
- **Exploration Strategies**: $\epsilon$-greedy, UCB, entropy regularization, and intrinsic motivation

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

> 💡 **Insight:** The Bellman equations are the algebraic spine of RL. They say that the value of a state is the immediate reward plus the discounted value of the next state — a one-step consistency condition. Both value iteration and Q-learning exploit this by repeatedly enforcing local consistency until it propagates globally. The optimality equation adds a $\max$ over actions, turning the linear system into a nonlinear fixed-point problem. The contraction mapping theorem guarantees convergence: each Bellman backup brings you closer to the true value function by a factor of $\gamma$.

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

> 💡 **Insight:** Temporal difference learning is a remarkable hybrid: it updates value estimates using _other_ value estimates (bootstrapping), unlike Monte Carlo methods that wait for actual returns. The TD error $\delta_t$ measures surprise — the difference between the received reward plus estimated future value and the current estimate. When TD errors are consistently positive, the agent has been underestimating value in that region; when negative, overestimating. TD(0) uses a single step of bootstrapping, while TD($\lambda$) blends all n-step returns via exponential weighting, offering a smooth bias-variance tradeoff parametrized by $\lambda \in [0,1]$.

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

> 💡 **Insight:** The policy gradient theorem is beautiful in its generality: $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a)]$. The $\nabla \log \pi$ term points in the direction that makes the chosen action more likely, and $Q^\pi(s,a)$ scales that direction by how good the action turned out to be. Subtracting a baseline $b(s)$ doesn't change the expected gradient (since $\mathbb{E}[\nabla \log \pi \cdot b(s)] = 0$) but dramatically reduces variance. Using $b(s) = V^\pi(s)$ converts the weighting from $Q$ to the advantage $A = Q - V$, which centers the signal around zero: positive advantages reinforce actions, negative advantages suppress them.

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

## Key Takeaways

- The MDP formalism captures sequential decision-making through the tuple $(S, A, P, R, \gamma)$; the Markov property makes the problem tractable by ensuring the future depends only on the current state and action.
- Bellman equations provide recursive consistency conditions for value functions; value iteration and policy iteration both converge to optimal policies through repeated application of these equations.
- TD learning bootstraps value estimates from other estimates, blending Monte Carlo and dynamic programming ideas; TD($\lambda$) unifies the spectrum from one-step to full-return estimation.
- Q-learning is off-policy (learns $Q^*$ regardless of the behavior policy), while SARSA is on-policy (evaluates and improves the policy being followed) — the distinction matters for safety and exploration.
- The policy gradient theorem provides an unbiased gradient estimator for the expected return; variance reduction through baselines (especially $V^\pi$) and advantage estimation are critical for practical performance.
- PPO stabilizes policy updates by clipping the importance ratio, preventing destructively large updates without requiring second-order optimization.
- Soft Actor-Critic adds entropy to the reward, encouraging exploration and providing a principled connection between RL and probabilistic inference.

## Exercises

1. **Bellman Backup**: For a simple 3-state MDP with transition probabilities and rewards given in a table, compute $V^\pi(s)$ for a uniform random policy by solving the Bellman expectation equations as a system of linear equations. Verify by running value iteration until convergence.

2. **TD(0) vs Monte Carlo**: Generate 100 episodes from a random walk MDP (5 states, terminal at each end). Implement both TD(0) and Monte Carlo prediction for $V^\pi$. Plot learning curves (RMSE vs episodes) for both methods and explain why TD(0) converges faster in this setting.

3. **Policy Gradient Derivation**: From the definition $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$, derive the REINFORCE gradient estimator. Show that subtracting a state-dependent baseline $b(s)$ preserves the expected gradient (prove $\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = 0$).

4. **Q-Learning Convergence**: Implement tabular Q-learning on a $4 \times 4$ gridworld. Experiment with $\epsilon$-greedy exploration using $\epsilon \in \{0.01, 0.1, 0.3\}$ and learning rates $\alpha \in \{0.1, 0.5, 1.0\}$. Plot the number of episodes to convergence and discuss the interaction between exploration and learning rate.

5. **PPO Clipping**: Explain geometrically what the PPO clipped objective $\min(r\hat{A}, \text{clip}(r, 1-\epsilon, 1+\epsilon)\hat{A})$ does when: (a) $\hat{A} > 0$ and $r > 1+\epsilon$, (b) $\hat{A} < 0$ and $r < 1-\epsilon$. Why does this prevent catastrophically large policy updates?

## References

1. Sutton & Barto - "Reinforcement Learning: An Introduction"
2. Schulman et al. - "Proximal Policy Optimization"
3. Mnih et al. - "Human-level control through deep RL"
4. Haarnoja et al. - "Soft Actor-Critic"
