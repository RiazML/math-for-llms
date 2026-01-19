"""
Reinforcement Learning: Exercises
================================

Practice implementing RL algorithms from scratch.
"""

import numpy as np
from typing import Tuple, List, Dict, Callable, Optional
from dataclasses import dataclass


# =============================================================================
# Exercise 1: Implement Value Iteration from Scratch
# =============================================================================

def exercise1_value_iteration():
    """
    Implement value iteration for a simple MDP.
    
    MDP Definition:
    - States: {0, 1, 2, 3, 4} (terminal: 4)
    - Actions: {left, right}
    - Transitions: Deterministic
    - Rewards: -1 per step, +10 at goal
    
    Tasks:
    1. Implement Bellman optimality update
    2. Extract optimal policy
    3. Handle terminal states correctly
    """
    
    # MDP definition
    n_states = 5
    n_actions = 2  # 0: left, 1: right
    terminal_state = 4
    
    # Transition function: P[s, a] = next_state
    # State 4 is terminal
    transitions = np.array([
        [0, 1],  # State 0: left->0, right->1
        [0, 2],  # State 1: left->0, right->2
        [1, 3],  # State 2: left->1, right->3
        [2, 4],  # State 3: left->2, right->4 (goal)
        [4, 4],  # State 4: terminal
    ])
    
    # Reward function: R[s, a, s']
    def get_reward(s, a, s_next):
        if s_next == terminal_state:
            return 10.0
        return -1.0
    
    gamma = 0.99
    
    def value_iteration(theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implement value iteration.
        
        Returns:
            V: Optimal value function
            policy: Optimal policy
        """
        # YOUR CODE HERE
        pass
    
    # Test
    V, policy = value_iteration()
    print(f"Optimal Values: {V}")
    print(f"Optimal Policy: {policy}")


def solution1_value_iteration():
    """Solution for Exercise 1."""
    
    n_states = 5
    n_actions = 2
    terminal_state = 4
    
    transitions = np.array([
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [4, 4],
    ])
    
    def get_reward(s, a, s_next):
        if s_next == terminal_state:
            return 10.0
        return -1.0
    
    gamma = 0.99
    
    def value_iteration(theta: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """Value iteration implementation."""
        V = np.zeros(n_states)
        
        while True:
            delta = 0
            V_new = np.zeros(n_states)
            
            for s in range(n_states):
                if s == terminal_state:
                    V_new[s] = 0
                    continue
                
                # Compute Q-values
                q_values = []
                for a in range(n_actions):
                    s_next = transitions[s, a]
                    r = get_reward(s, a, s_next)
                    q = r + gamma * V[s_next]
                    q_values.append(q)
                
                V_new[s] = max(q_values)
                delta = max(delta, abs(V_new[s] - V[s]))
            
            V = V_new
            
            if delta < theta:
                break
        
        # Extract policy
        policy = np.zeros(n_states, dtype=int)
        for s in range(n_states):
            if s == terminal_state:
                continue
            
            q_values = []
            for a in range(n_actions):
                s_next = transitions[s, a]
                r = get_reward(s, a, s_next)
                q = r + gamma * V[s_next]
                q_values.append(q)
            
            policy[s] = np.argmax(q_values)
        
        return V, policy
    
    V, policy = value_iteration()
    print(f"Optimal Values: {V.round(2)}")
    print(f"Optimal Policy: {policy} (0=left, 1=right)")
    
    # Verify: From state 0, optimal path is 0->1->2->3->4
    # Value should be: -1 + γ*(-1) + γ²*(-1) + γ³*(10) ≈ 6.88
    expected_v0 = -1 + 0.99*(-1) + 0.99**2*(-1) + 0.99**3*10
    print(f"Expected V[0]: {expected_v0:.2f}, Got: {V[0]:.2f}")


# =============================================================================
# Exercise 2: Implement Q-Learning with Exploration Decay
# =============================================================================

def exercise2_q_learning():
    """
    Implement Q-Learning with decaying exploration.
    
    Tasks:
    1. Implement ε-greedy with decay
    2. Implement Q-learning update
    3. Track learning progress
    """
    
    class CliffWalking:
        """
        Cliff Walking environment.
        
        4x12 grid. Start at bottom-left, goal at bottom-right.
        Bottom row (except start/goal) is a cliff.
        """
        
        def __init__(self):
            self.height = 4
            self.width = 12
            self.start = (3, 0)
            self.goal = (3, 11)
            self.cliff = [(3, i) for i in range(1, 11)]
            
            self.state = None
            self.reset()
        
        def reset(self) -> Tuple[int, int]:
            self.state = self.start
            return self.state
        
        def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
            """
            Actions: 0=up, 1=down, 2=left, 3=right
            """
            dy, dx = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
            new_y = max(0, min(self.height - 1, self.state[0] + dy))
            new_x = max(0, min(self.width - 1, self.state[1] + dx))
            
            self.state = (new_y, new_x)
            
            if self.state in self.cliff:
                self.state = self.start
                return self.state, -100.0, False
            
            if self.state == self.goal:
                return self.state, 0.0, True
            
            return self.state, -1.0, False
    
    def q_learning(
        env: CliffWalking,
        n_episodes: int = 500,
        alpha: float = 0.1,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Q-Learning with decaying exploration.
        
        Returns:
            Q: Learned Q-values
            rewards: Episode rewards
        """
        # YOUR CODE HERE
        pass
    
    # Test
    env = CliffWalking()
    # Q, rewards = q_learning(env)
    print("Test Q-learning implementation...")


def solution2_q_learning():
    """Solution for Exercise 2."""
    
    class CliffWalking:
        def __init__(self):
            self.height = 4
            self.width = 12
            self.start = (3, 0)
            self.goal = (3, 11)
            self.cliff = [(3, i) for i in range(1, 11)]
            self.state = None
            self.reset()
        
        def reset(self) -> Tuple[int, int]:
            self.state = self.start
            return self.state
        
        def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
            dy, dx = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
            new_y = max(0, min(self.height - 1, self.state[0] + dy))
            new_x = max(0, min(self.width - 1, self.state[1] + dx))
            
            self.state = (new_y, new_x)
            
            if self.state in self.cliff:
                self.state = self.start
                return self.state, -100.0, False
            
            if self.state == self.goal:
                return self.state, 0.0, True
            
            return self.state, -1.0, False
    
    def q_learning(
        env: CliffWalking,
        n_episodes: int = 500,
        alpha: float = 0.1,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995
    ) -> Tuple[np.ndarray, List[float]]:
        """Q-Learning with decaying exploration."""
        
        Q = np.zeros((env.height, env.width, 4))
        rewards_history = []
        epsilon = epsilon_start
        
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            
            for _ in range(1000):  # Max steps
                # ε-greedy action selection
                if np.random.random() < epsilon:
                    action = np.random.randint(4)
                else:
                    action = np.argmax(Q[state[0], state[1]])
                
                next_state, reward, done = env.step(action)
                
                # Q-learning update
                best_next = np.max(Q[next_state[0], next_state[1]])
                td_target = reward + gamma * best_next * (1 - done)
                td_error = td_target - Q[state[0], state[1], action]
                Q[state[0], state[1], action] += alpha * td_error
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            rewards_history.append(total_reward)
            
            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        return Q, rewards_history
    
    env = CliffWalking()
    Q, rewards = q_learning(env, n_episodes=500)
    
    print(f"Average reward (last 50): {np.mean(rewards[-50:]):.1f}")
    
    # Show optimal path
    action_names = ['↑', '↓', '←', '→']
    print("\nLearned policy (bottom row safe path):")
    for x in range(12):
        a = np.argmax(Q[2, x])
        print(f"  ({2},{x}): {action_names[a]}", end="")
    print()


# =============================================================================
# Exercise 3: Implement SARSA with Eligibility Traces
# =============================================================================

def exercise3_sarsa_lambda():
    """
    Implement SARSA(λ) with eligibility traces.
    
    Tasks:
    1. Implement eligibility trace updates
    2. Implement SARSA with traces
    3. Compare with standard SARSA
    """
    
    def sarsa_lambda(
        env,  # GridWorld-like environment
        n_episodes: int = 500,
        alpha: float = 0.1,
        gamma: float = 0.99,
        lambda_: float = 0.9,
        epsilon: float = 0.1
    ) -> Tuple[np.ndarray, List[float]]:
        """
        SARSA(λ) implementation.
        
        e_t(s,a) = γλ e_{t-1}(s,a) + 1(S_t=s, A_t=a)
        Q(s,a) ← Q(s,a) + α δ e(s,a)
        
        Returns:
            Q: Learned Q-values
            rewards: Episode rewards
        """
        # YOUR CODE HERE
        pass
    
    print("Test SARSA(λ) implementation...")


def solution3_sarsa_lambda():
    """Solution for Exercise 3."""
    
    class SimpleEnv:
        """Simple corridor environment."""
        def __init__(self, length: int = 10):
            self.length = length
            self.state = 0
        
        def reset(self) -> int:
            self.state = 0
            return self.state
        
        def step(self, action: int) -> Tuple[int, float, bool]:
            # action: 0=left, 1=right
            if action == 1:
                self.state = min(self.length - 1, self.state + 1)
            else:
                self.state = max(0, self.state - 1)
            
            done = self.state == self.length - 1
            reward = 1.0 if done else 0.0
            return self.state, reward, done
    
    def sarsa_lambda(
        env,
        n_episodes: int = 500,
        alpha: float = 0.1,
        gamma: float = 0.99,
        lambda_: float = 0.9,
        epsilon: float = 0.1
    ) -> Tuple[np.ndarray, List[float]]:
        """SARSA(λ) implementation."""
        
        n_states = env.length
        n_actions = 2
        Q = np.zeros((n_states, n_actions))
        rewards_history = []
        
        for episode in range(n_episodes):
            state = env.reset()
            
            # Select action
            if np.random.random() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax(Q[state])
            
            # Initialize eligibility traces
            e = np.zeros((n_states, n_actions))
            
            total_reward = 0
            
            for _ in range(500):
                next_state, reward, done = env.step(action)
                
                # Select next action
                if np.random.random() < epsilon:
                    next_action = np.random.randint(n_actions)
                else:
                    next_action = np.argmax(Q[next_state])
                
                # TD error
                if done:
                    delta = reward - Q[state, action]
                else:
                    delta = reward + gamma * Q[next_state, next_action] - Q[state, action]
                
                # Update eligibility trace (accumulating)
                e[state, action] += 1
                
                # Update all Q-values
                Q += alpha * delta * e
                
                # Decay traces
                e *= gamma * lambda_
                
                total_reward += reward
                state = next_state
                action = next_action
                
                if done:
                    break
            
            rewards_history.append(total_reward)
        
        return Q, rewards_history
    
    env = SimpleEnv(length=10)
    
    # Compare SARSA(0) and SARSA(0.9)
    Q0, rewards0 = sarsa_lambda(env, lambda_=0.0, n_episodes=200)
    Q9, rewards9 = sarsa_lambda(env, lambda_=0.9, n_episodes=200)
    
    print(f"SARSA(0) avg reward (last 50): {np.mean(rewards0[-50:]):.3f}")
    print(f"SARSA(0.9) avg reward (last 50): {np.mean(rewards9[-50:]):.3f}")
    
    # Check learned Q-values
    print(f"\nQ-values at state 8 (one step from goal):")
    print(f"  SARSA(0): left={Q0[8,0]:.3f}, right={Q0[8,1]:.3f}")
    print(f"  SARSA(0.9): left={Q9[8,0]:.3f}, right={Q9[8,1]:.3f}")


# =============================================================================
# Exercise 4: Implement REINFORCE with Baseline
# =============================================================================

def exercise4_reinforce_baseline():
    """
    Implement REINFORCE with baseline (value function).
    
    Tasks:
    1. Implement policy network
    2. Implement value network (baseline)
    3. Compute advantage estimates
    4. Update both networks
    """
    
    class PolicyNetwork:
        """Simple softmax policy."""
        
        def __init__(self, n_states: int, n_actions: int):
            self.theta = np.zeros((n_states, n_actions))
        
        def get_probs(self, state: int) -> np.ndarray:
            """Get action probabilities."""
            # YOUR CODE HERE
            pass
        
        def sample_action(self, state: int) -> int:
            """Sample action from policy."""
            # YOUR CODE HERE
            pass
        
        def update(self, state: int, action: int, advantage: float, lr: float):
            """Policy gradient update."""
            # YOUR CODE HERE
            pass
    
    class ValueNetwork:
        """Simple value function."""
        
        def __init__(self, n_states: int):
            self.w = np.zeros(n_states)
        
        def get_value(self, state: int) -> float:
            """Get state value."""
            # YOUR CODE HERE
            pass
        
        def update(self, state: int, target: float, lr: float):
            """Value update."""
            # YOUR CODE HERE
            pass
    
    def reinforce_baseline(
        env,
        n_episodes: int = 1000,
        lr_policy: float = 0.01,
        lr_value: float = 0.1,
        gamma: float = 0.99
    ) -> Tuple[PolicyNetwork, ValueNetwork, List[float]]:
        """
        REINFORCE with baseline.
        
        Returns:
            policy, value, rewards
        """
        # YOUR CODE HERE
        pass
    
    print("Test REINFORCE with baseline...")


def solution4_reinforce_baseline():
    """Solution for Exercise 4."""
    
    class SimpleEnv:
        """Simple environment."""
        def __init__(self, n_states: int = 10):
            self.n_states = n_states
            self.goal = n_states - 1
            self.state = 0
        
        def reset(self) -> int:
            self.state = 0
            return self.state
        
        def step(self, action: int) -> Tuple[int, float, bool]:
            if action == 1:  # right
                self.state = min(self.goal, self.state + 1)
            else:  # left
                self.state = max(0, self.state - 1)
            
            done = self.state == self.goal
            reward = 1.0 if done else -0.01
            return self.state, reward, done
    
    class PolicyNetwork:
        def __init__(self, n_states: int, n_actions: int):
            self.n_states = n_states
            self.n_actions = n_actions
            self.theta = np.zeros((n_states, n_actions))
        
        def get_probs(self, state: int) -> np.ndarray:
            logits = self.theta[state]
            exp_logits = np.exp(logits - np.max(logits))
            return exp_logits / np.sum(exp_logits)
        
        def sample_action(self, state: int) -> int:
            probs = self.get_probs(state)
            return np.random.choice(self.n_actions, p=probs)
        
        def update(self, state: int, action: int, advantage: float, lr: float):
            probs = self.get_probs(state)
            grad = -probs
            grad[action] += 1
            self.theta[state] += lr * advantage * grad
    
    class ValueNetwork:
        def __init__(self, n_states: int):
            self.w = np.zeros(n_states)
        
        def get_value(self, state: int) -> float:
            return self.w[state]
        
        def update(self, state: int, target: float, lr: float):
            error = target - self.w[state]
            self.w[state] += lr * error
    
    def reinforce_baseline(
        env,
        n_episodes: int = 1000,
        lr_policy: float = 0.01,
        lr_value: float = 0.1,
        gamma: float = 0.99
    ) -> Tuple[PolicyNetwork, ValueNetwork, List[float]]:
        """REINFORCE with baseline."""
        
        policy = PolicyNetwork(env.n_states, 2)
        value = ValueNetwork(env.n_states)
        rewards_history = []
        
        for episode in range(n_episodes):
            state = env.reset()
            states, actions, rewards = [], [], []
            
            # Collect trajectory
            for _ in range(500):
                action = policy.sample_action(state)
                next_state, reward, done = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state
                if done:
                    break
            
            # Compute returns
            T = len(rewards)
            G = np.zeros(T)
            G[T-1] = rewards[T-1]
            for t in range(T-2, -1, -1):
                G[t] = rewards[t] + gamma * G[t+1]
            
            # Update policy and value
            for t in range(T):
                s = states[t]
                a = actions[t]
                
                # Advantage = return - baseline
                advantage = G[t] - value.get_value(s)
                
                # Update policy
                policy.update(s, a, advantage * (gamma ** t), lr_policy)
                
                # Update value function
                value.update(s, G[t], lr_value)
            
            rewards_history.append(sum(rewards))
        
        return policy, value, rewards_history
    
    env = SimpleEnv(n_states=10)
    policy, value, rewards = reinforce_baseline(env, n_episodes=500)
    
    print(f"Average reward (last 50): {np.mean(rewards[-50:]):.3f}")
    
    # Show learned value function
    print("\nLearned value function:")
    for s in range(env.n_states):
        print(f"  V({s}) = {value.get_value(s):.3f}")


# =============================================================================
# Exercise 5: Implement Actor-Critic with TD(0)
# =============================================================================

def exercise5_actor_critic():
    """
    Implement one-step Actor-Critic.
    
    Tasks:
    1. Actor: stochastic policy π_θ(a|s)
    2. Critic: value function V_w(s)
    3. Advantage: δ = r + γV(s') - V(s)
    4. Update both actor and critic online
    """
    
    def actor_critic(
        env,
        n_episodes: int = 1000,
        lr_actor: float = 0.01,
        lr_critic: float = 0.1,
        gamma: float = 0.99
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        One-step Actor-Critic.
        
        Returns:
            theta: Policy parameters
            w: Value parameters
            rewards: Episode rewards
        """
        # YOUR CODE HERE
        pass
    
    print("Test Actor-Critic implementation...")


def solution5_actor_critic():
    """Solution for Exercise 5."""
    
    class SimpleEnv:
        def __init__(self, n_states: int = 10):
            self.n_states = n_states
            self.goal = n_states - 1
            self.state = 0
        
        def reset(self) -> int:
            self.state = 0
            return self.state
        
        def step(self, action: int) -> Tuple[int, float, bool]:
            if action == 1:
                self.state = min(self.goal, self.state + 1)
            else:
                self.state = max(0, self.state - 1)
            
            done = self.state == self.goal
            reward = 1.0 if done else -0.01
            return self.state, reward, done
    
    def actor_critic(
        env,
        n_episodes: int = 1000,
        lr_actor: float = 0.01,
        lr_critic: float = 0.1,
        gamma: float = 0.99
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """One-step Actor-Critic."""
        
        n_states = env.n_states
        n_actions = 2
        
        # Actor parameters
        theta = np.zeros((n_states, n_actions))
        
        # Critic parameters
        w = np.zeros(n_states)
        
        def get_probs(state):
            logits = theta[state]
            exp_logits = np.exp(logits - np.max(logits))
            return exp_logits / np.sum(exp_logits)
        
        def sample_action(state):
            probs = get_probs(state)
            return np.random.choice(n_actions, p=probs)
        
        rewards_history = []
        
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            
            for _ in range(500):
                action = sample_action(state)
                next_state, reward, done = env.step(action)
                
                # TD error (advantage estimate)
                if done:
                    delta = reward - w[state]
                else:
                    delta = reward + gamma * w[next_state] - w[state]
                
                # Critic update
                w[state] += lr_critic * delta
                
                # Actor update
                probs = get_probs(state)
                grad = -probs
                grad[action] += 1
                theta[state] += lr_actor * delta * grad
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            rewards_history.append(total_reward)
        
        return theta, w, rewards_history
    
    env = SimpleEnv(n_states=10)
    theta, w, rewards = actor_critic(env, n_episodes=500)
    
    print(f"Average reward (last 50): {np.mean(rewards[-50:]):.3f}")
    
    # Show policy at each state
    print("\nLearned policy (right action probability):")
    for s in range(env.n_states):
        logits = theta[s]
        probs = np.exp(logits - np.max(logits))
        probs /= np.sum(probs)
        print(f"  State {s}: P(right) = {probs[1]:.3f}")


# =============================================================================
# Exercise 6: Implement N-step TD
# =============================================================================

def exercise6_n_step_td():
    """
    Implement n-step TD for value estimation.
    
    G_t^{(n)} = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n V(S_{t+n})
    V(S_t) ← V(S_t) + α[G_t^{(n)} - V(S_t)]
    
    Tasks:
    1. Implement n-step return calculation
    2. Handle episode boundaries
    3. Compare different values of n
    """
    
    def n_step_td(
        env,
        policy: np.ndarray,  # deterministic policy
        n: int = 4,
        n_episodes: int = 100,
        alpha: float = 0.1,
        gamma: float = 0.99
    ) -> np.ndarray:
        """
        N-step TD for policy evaluation.
        
        Returns:
            V: Estimated value function
        """
        # YOUR CODE HERE
        pass
    
    print("Test n-step TD implementation...")


def solution6_n_step_td():
    """Solution for Exercise 6."""
    
    class SimpleEnv:
        def __init__(self, n_states: int = 10):
            self.n_states = n_states
            self.goal = n_states - 1
            self.state = 0
        
        def reset(self) -> int:
            self.state = 0
            return self.state
        
        def step(self, action: int) -> Tuple[int, float, bool]:
            if action == 1:
                self.state = min(self.goal, self.state + 1)
            else:
                self.state = max(0, self.state - 1)
            
            done = self.state == self.goal
            reward = 1.0 if done else 0.0
            return self.state, reward, done
    
    def n_step_td(
        env,
        policy: np.ndarray,
        n: int = 4,
        n_episodes: int = 100,
        alpha: float = 0.1,
        gamma: float = 0.99
    ) -> np.ndarray:
        """N-step TD for policy evaluation."""
        
        V = np.zeros(env.n_states)
        
        for episode in range(n_episodes):
            # Store trajectory
            states = [env.reset()]
            rewards = [0]  # Placeholder for R_0
            
            T = float('inf')
            t = 0
            
            while True:
                if t < T:
                    action = policy[states[t]]
                    next_state, reward, done = env.step(action)
                    states.append(next_state)
                    rewards.append(reward)
                    
                    if done:
                        T = t + 1
                
                # Update time
                tau = t - n + 1
                
                if tau >= 0:
                    # Compute n-step return
                    G = 0
                    for i in range(tau + 1, min(tau + n, T) + 1):
                        G += (gamma ** (i - tau - 1)) * rewards[i]
                    
                    if tau + n < T:
                        G += (gamma ** n) * V[states[tau + n]]
                    
                    # Update
                    V[states[tau]] += alpha * (G - V[states[tau]])
                
                t += 1
                
                if tau == T - 1:
                    break
        
        return V
    
    env = SimpleEnv(n_states=10)
    
    # Optimal policy: always go right
    policy = np.ones(env.n_states, dtype=int)
    
    # Compare different n values
    for n in [1, 2, 4, 8]:
        V = n_step_td(env, policy, n=n, n_episodes=200)
        print(f"n={n}: V = {V.round(3)}")


# =============================================================================
# Exercise 7: Implement Double Q-Learning
# =============================================================================

def exercise7_double_q_learning():
    """
    Implement Double Q-Learning to reduce maximization bias.
    
    Use two Q-functions Q1 and Q2.
    Update rule:
        Q1(S, A) ← Q1(S, A) + α[R + γQ2(S', argmax_a Q1(S', a)) - Q1(S, A)]
    
    With probability 0.5, swap Q1 and Q2.
    
    Tasks:
    1. Implement double Q-learning
    2. Compare with standard Q-learning
    3. Analyze maximization bias reduction
    """
    
    def double_q_learning(
        env,
        n_episodes: int = 1000,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Double Q-Learning.
        
        Returns:
            Q1, Q2, rewards
        """
        # YOUR CODE HERE
        pass
    
    print("Test Double Q-Learning implementation...")


def solution7_double_q_learning():
    """Solution for Exercise 7."""
    
    class MaxBiasEnv:
        """
        Environment that demonstrates maximization bias.
        
        Two states: A (start), B (terminal with many noisy actions)
        From A: left->terminal(0), right->B
        From B: many actions with mean 0 but high variance
        """
        def __init__(self, n_b_actions: int = 10):
            self.n_b_actions = n_b_actions
            self.state = 'A'
        
        def reset(self) -> str:
            self.state = 'A'
            return self.state
        
        def step(self, action: int) -> Tuple[str, float, bool]:
            if self.state == 'A':
                if action == 0:  # left
                    return 'terminal', 0.0, True
                else:  # right
                    self.state = 'B'
                    return self.state, 0.0, False
            else:  # state B
                # Reward: N(-0.1, 1)
                reward = np.random.normal(-0.1, 1.0)
                return 'terminal', reward, True
    
    def q_learning(env, n_episodes=300, alpha=0.1, gamma=1.0, epsilon=0.1):
        """Standard Q-learning."""
        Q_A = np.zeros(2)  # left, right
        Q_B = np.zeros(env.n_b_actions)
        
        left_counts = []
        
        for episode in range(n_episodes):
            state = env.reset()
            
            while True:
                if state == 'A':
                    if np.random.random() < epsilon:
                        action = np.random.randint(2)
                    else:
                        action = np.argmax(Q_A)
                    
                    next_state, reward, done = env.step(action)
                    
                    if next_state == 'terminal':
                        Q_A[action] += alpha * (reward - Q_A[action])
                    else:
                        Q_A[action] += alpha * (reward + gamma * np.max(Q_B) - Q_A[action])
                else:  # state B
                    if np.random.random() < epsilon:
                        action = np.random.randint(env.n_b_actions)
                    else:
                        action = np.argmax(Q_B)
                    
                    next_state, reward, done = env.step(action)
                    Q_B[action] += alpha * (reward - Q_B[action])
                
                state = next_state
                if done:
                    break
            
            # Track how often "left" is chosen from A
            left_counts.append(int(np.argmax(Q_A) == 0))
        
        return Q_A, Q_B, left_counts
    
    def double_q_learning(env, n_episodes=300, alpha=0.1, gamma=1.0, epsilon=0.1):
        """Double Q-learning."""
        Q1_A = np.zeros(2)
        Q1_B = np.zeros(env.n_b_actions)
        Q2_A = np.zeros(2)
        Q2_B = np.zeros(env.n_b_actions)
        
        left_counts = []
        
        for episode in range(n_episodes):
            state = env.reset()
            
            while True:
                if state == 'A':
                    Q_combined = Q1_A + Q2_A
                    if np.random.random() < epsilon:
                        action = np.random.randint(2)
                    else:
                        action = np.argmax(Q_combined)
                    
                    next_state, reward, done = env.step(action)
                    
                    if np.random.random() < 0.5:
                        # Update Q1
                        if next_state == 'terminal':
                            Q1_A[action] += alpha * (reward - Q1_A[action])
                        else:
                            best_a = np.argmax(Q1_B)
                            Q1_A[action] += alpha * (reward + gamma * Q2_B[best_a] - Q1_A[action])
                    else:
                        # Update Q2
                        if next_state == 'terminal':
                            Q2_A[action] += alpha * (reward - Q2_A[action])
                        else:
                            best_a = np.argmax(Q2_B)
                            Q2_A[action] += alpha * (reward + gamma * Q1_B[best_a] - Q2_A[action])
                else:
                    Q_combined = Q1_B + Q2_B
                    if np.random.random() < epsilon:
                        action = np.random.randint(env.n_b_actions)
                    else:
                        action = np.argmax(Q_combined)
                    
                    next_state, reward, done = env.step(action)
                    
                    if np.random.random() < 0.5:
                        Q1_B[action] += alpha * (reward - Q1_B[action])
                    else:
                        Q2_B[action] += alpha * (reward - Q2_B[action])
                
                state = next_state
                if done:
                    break
            
            left_counts.append(int(np.argmax(Q1_A + Q2_A) == 0))
        
        return Q1_A + Q2_A, Q1_B + Q2_B, left_counts
    
    # Compare
    np.random.seed(42)
    env = MaxBiasEnv(n_b_actions=10)
    
    # Optimal: go left (reward 0 vs expected -0.1)
    
    _, _, q_left = q_learning(env, n_episodes=300)
    _, _, dq_left = double_q_learning(env, n_episodes=300)
    
    print("Percentage choosing optimal action (left) over episodes:")
    print(f"  Q-Learning: {100 * np.mean(q_left[-100:]):.1f}%")
    print(f"  Double Q-Learning: {100 * np.mean(dq_left[-100:]):.1f}%")


# =============================================================================
# Exercise 8: Implement PPO (Simplified)
# =============================================================================

def exercise8_ppo():
    """
    Implement simplified Proximal Policy Optimization.
    
    L^{CLIP} = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
    
    where r_t = π_θ(a|s) / π_θ_old(a|s)
    
    Tasks:
    1. Implement clipped objective
    2. Implement advantage estimation
    3. Multiple epochs per batch
    """
    
    class SimplePPO:
        def __init__(
            self,
            n_states: int,
            n_actions: int,
            clip_epsilon: float = 0.2,
            lr: float = 0.01,
            gamma: float = 0.99
        ):
            self.n_states = n_states
            self.n_actions = n_actions
            self.clip_epsilon = clip_epsilon
            self.lr = lr
            self.gamma = gamma
            
            self.theta = np.zeros((n_states, n_actions))
        
        def get_probs(self, state: int) -> np.ndarray:
            """Get action probabilities."""
            # YOUR CODE HERE
            pass
        
        def compute_advantages(
            self,
            rewards: List[float],
            values: List[float],
            dones: List[bool]
        ) -> np.ndarray:
            """Compute generalized advantage estimates."""
            # YOUR CODE HERE
            pass
        
        def update(
            self,
            states: List[int],
            actions: List[int],
            advantages: np.ndarray,
            old_probs: List[np.ndarray],
            n_epochs: int = 4
        ):
            """PPO update with clipped objective."""
            # YOUR CODE HERE
            pass
    
    print("Test PPO implementation...")


def solution8_ppo():
    """Solution for Exercise 8."""
    
    class SimpleEnv:
        def __init__(self, n_states: int = 10):
            self.n_states = n_states
            self.goal = n_states - 1
            self.state = 0
        
        def reset(self) -> int:
            self.state = 0
            return self.state
        
        def step(self, action: int) -> Tuple[int, float, bool]:
            if action == 1:
                self.state = min(self.goal, self.state + 1)
            else:
                self.state = max(0, self.state - 1)
            
            done = self.state == self.goal
            reward = 1.0 if done else -0.01
            return self.state, reward, done
    
    class SimplePPO:
        def __init__(
            self,
            n_states: int,
            n_actions: int,
            clip_epsilon: float = 0.2,
            lr: float = 0.01,
            gamma: float = 0.99,
            lam: float = 0.95
        ):
            self.n_states = n_states
            self.n_actions = n_actions
            self.clip_epsilon = clip_epsilon
            self.lr = lr
            self.gamma = gamma
            self.lam = lam
            
            # Policy parameters
            self.theta = np.zeros((n_states, n_actions))
            
            # Value function
            self.w = np.zeros(n_states)
        
        def get_probs(self, state: int) -> np.ndarray:
            logits = self.theta[state]
            exp_logits = np.exp(logits - np.max(logits))
            return exp_logits / np.sum(exp_logits)
        
        def sample_action(self, state: int) -> int:
            probs = self.get_probs(state)
            return np.random.choice(self.n_actions, p=probs)
        
        def compute_advantages(
            self,
            rewards: List[float],
            states: List[int],
            dones: List[bool]
        ) -> np.ndarray:
            """GAE computation."""
            T = len(rewards)
            advantages = np.zeros(T)
            
            last_adv = 0
            for t in reversed(range(T)):
                if t == T - 1 or dones[t]:
                    next_value = 0
                else:
                    next_value = self.w[states[t + 1]]
                
                delta = rewards[t] + self.gamma * next_value - self.w[states[t]]
                advantages[t] = delta + self.gamma * self.lam * last_adv * (1 - dones[t])
                last_adv = advantages[t]
            
            return advantages
        
        def update(
            self,
            states: List[int],
            actions: List[int],
            advantages: np.ndarray,
            old_probs: List[np.ndarray],
            n_epochs: int = 4
        ):
            """PPO clipped update."""
            T = len(states)
            returns = advantages + np.array([self.w[s] for s in states])
            
            # Normalize advantages
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            
            for _ in range(n_epochs):
                for t in range(T):
                    s = states[t]
                    a = actions[t]
                    adv = advantages[t]
                    
                    # Current probability
                    new_probs = self.get_probs(s)
                    
                    # Probability ratio
                    ratio = new_probs[a] / (old_probs[t][a] + 1e-8)
                    
                    # Clipped objective
                    obj1 = ratio * adv
                    obj2 = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv
                    
                    # Policy gradient (maximize min(obj1, obj2))
                    if obj1 < obj2:
                        grad = -new_probs
                        grad[a] += 1
                        self.theta[s] += self.lr * adv * grad
                    elif ratio < 1 - self.clip_epsilon or ratio > 1 + self.clip_epsilon:
                        pass  # Clipped, no gradient
                    else:
                        grad = -new_probs
                        grad[a] += 1
                        self.theta[s] += self.lr * adv * grad
                    
                    # Value update
                    self.w[s] += 0.1 * (returns[t] - self.w[s])
    
    env = SimpleEnv(n_states=10)
    ppo = SimplePPO(env.n_states, 2, clip_epsilon=0.2, lr=0.01)
    
    rewards_history = []
    
    for episode in range(500):
        state = env.reset()
        states, actions, rewards, dones, old_probs = [], [], [], [], []
        total_reward = 0
        
        for _ in range(100):
            action = ppo.sample_action(state)
            old_probs.append(ppo.get_probs(state).copy())
            
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Compute advantages and update
        advantages = ppo.compute_advantages(rewards, states, dones)
        ppo.update(states, actions, advantages, old_probs, n_epochs=4)
        
        rewards_history.append(total_reward)
    
    print(f"PPO Average reward (last 50): {np.mean(rewards_history[-50:]):.3f}")


# =============================================================================
# Exercise 9: Implement Experience Replay Buffer
# =============================================================================

def exercise9_replay_buffer():
    """
    Implement prioritized experience replay buffer.
    
    Tasks:
    1. Implement circular buffer
    2. Implement uniform sampling
    3. Implement prioritized sampling
    4. Implement importance sampling weights
    """
    
    class ReplayBuffer:
        """Standard replay buffer."""
        
        def __init__(self, capacity: int):
            self.capacity = capacity
            # YOUR CODE HERE
            pass
        
        def push(self, state, action, reward, next_state, done):
            """Add transition to buffer."""
            # YOUR CODE HERE
            pass
        
        def sample(self, batch_size: int) -> Tuple:
            """Sample random batch."""
            # YOUR CODE HERE
            pass
        
        def __len__(self) -> int:
            # YOUR CODE HERE
            pass
    
    class PrioritizedReplayBuffer:
        """Prioritized experience replay."""
        
        def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
            self.capacity = capacity
            self.alpha = alpha  # Priority exponent
            self.beta = beta    # Importance sampling exponent
            # YOUR CODE HERE
            pass
        
        def push(self, state, action, reward, next_state, done, td_error: float):
            """Add transition with priority."""
            # YOUR CODE HERE
            pass
        
        def sample(self, batch_size: int) -> Tuple:
            """Sample batch with priorities."""
            # YOUR CODE HERE
            pass
        
        def update_priorities(self, indices: List[int], td_errors: np.ndarray):
            """Update priorities based on new TD errors."""
            # YOUR CODE HERE
            pass
    
    print("Test replay buffer implementations...")


def solution9_replay_buffer():
    """Solution for Exercise 9."""
    
    class ReplayBuffer:
        def __init__(self, capacity: int):
            self.capacity = capacity
            self.buffer = []
            self.position = 0
        
        def push(self, state, action, reward, next_state, done):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.capacity
        
        def sample(self, batch_size: int) -> Tuple:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
            
            states = np.array([t[0] for t in batch])
            actions = np.array([t[1] for t in batch])
            rewards = np.array([t[2] for t in batch])
            next_states = np.array([t[3] for t in batch])
            dones = np.array([t[4] for t in batch])
            
            return states, actions, rewards, next_states, dones
        
        def __len__(self) -> int:
            return len(self.buffer)
    
    class PrioritizedReplayBuffer:
        def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
            self.capacity = capacity
            self.alpha = alpha
            self.beta = beta
            
            self.buffer = []
            self.priorities = np.zeros(capacity)
            self.position = 0
            self.max_priority = 1.0
        
        def push(self, state, action, reward, next_state, done, td_error: float = None):
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            
            self.buffer[self.position] = (state, action, reward, next_state, done)
            
            # Set priority (use max for new transitions)
            priority = self.max_priority if td_error is None else (abs(td_error) + 1e-6)
            self.priorities[self.position] = priority ** self.alpha
            
            self.position = (self.position + 1) % self.capacity
        
        def sample(self, batch_size: int) -> Tuple:
            if len(self.buffer) < batch_size:
                return None
            
            # Compute sampling probabilities
            priorities = self.priorities[:len(self.buffer)]
            probs = priorities / np.sum(priorities)
            
            # Sample indices
            indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
            
            # Compute importance sampling weights
            N = len(self.buffer)
            weights = (N * probs[indices]) ** (-self.beta)
            weights /= np.max(weights)  # Normalize
            
            batch = [self.buffer[i] for i in indices]
            
            states = np.array([t[0] for t in batch])
            actions = np.array([t[1] for t in batch])
            rewards = np.array([t[2] for t in batch])
            next_states = np.array([t[3] for t in batch])
            dones = np.array([t[4] for t in batch])
            
            return states, actions, rewards, next_states, dones, weights, indices
        
        def update_priorities(self, indices: List[int], td_errors: np.ndarray):
            for idx, td_error in zip(indices, td_errors):
                priority = (abs(td_error) + 1e-6) ** self.alpha
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
        
        def __len__(self) -> int:
            return len(self.buffer)
    
    # Test
    buffer = ReplayBuffer(capacity=1000)
    
    for i in range(100):
        buffer.push(
            state=np.random.randn(4),
            action=np.random.randint(2),
            reward=np.random.randn(),
            next_state=np.random.randn(4),
            done=np.random.random() < 0.1
        )
    
    states, actions, rewards, next_states, dones = buffer.sample(32)
    print(f"Uniform buffer - sampled batch shape: {states.shape}")
    
    # Test prioritized buffer
    p_buffer = PrioritizedReplayBuffer(capacity=1000)
    
    for i in range(100):
        p_buffer.push(
            state=np.random.randn(4),
            action=np.random.randint(2),
            reward=np.random.randn(),
            next_state=np.random.randn(4),
            done=np.random.random() < 0.1,
            td_error=np.random.rand()
        )
    
    result = p_buffer.sample(32)
    states, actions, rewards, next_states, dones, weights, indices = result
    print(f"Prioritized buffer - sampled batch shape: {states.shape}")
    print(f"  Weights range: [{weights.min():.3f}, {weights.max():.3f}]")


# =============================================================================
# Exercise 10: Complete RL Training Pipeline
# =============================================================================

def exercise10_complete_pipeline():
    """
    Implement a complete RL training pipeline.
    
    Tasks:
    1. Environment wrapper
    2. Agent with exploration schedule
    3. Training loop with logging
    4. Evaluation procedure
    5. Model checkpointing
    """
    
    class RLTrainer:
        def __init__(
            self,
            env,
            agent,
            n_episodes: int = 1000,
            eval_frequency: int = 100,
            n_eval_episodes: int = 10
        ):
            self.env = env
            self.agent = agent
            self.n_episodes = n_episodes
            self.eval_frequency = eval_frequency
            self.n_eval_episodes = n_eval_episodes
            
            self.train_rewards = []
            self.eval_rewards = []
        
        def train_episode(self) -> float:
            """Run one training episode."""
            # YOUR CODE HERE
            pass
        
        def evaluate(self) -> float:
            """Evaluate agent."""
            # YOUR CODE HERE
            pass
        
        def train(self) -> Dict:
            """Full training loop."""
            # YOUR CODE HERE
            pass
    
    print("Test complete RL pipeline...")


def solution10_complete_pipeline():
    """Solution for Exercise 10."""
    
    class SimpleEnv:
        def __init__(self, n_states: int = 20):
            self.n_states = n_states
            self.goal = n_states - 1
            self.state = 0
        
        def reset(self) -> int:
            self.state = 0
            return self.state
        
        def step(self, action: int) -> Tuple[int, float, bool]:
            if action == 1:
                self.state = min(self.goal, self.state + 1)
            else:
                self.state = max(0, self.state - 1)
            
            done = self.state == self.goal
            reward = 10.0 if done else -0.1
            return self.state, reward, done
    
    class QLearningAgent:
        def __init__(
            self,
            n_states: int,
            n_actions: int,
            alpha: float = 0.1,
            gamma: float = 0.99,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.01,
            epsilon_decay: float = 0.995
        ):
            self.n_states = n_states
            self.n_actions = n_actions
            self.alpha = alpha
            self.gamma = gamma
            
            self.epsilon = epsilon_start
            self.epsilon_end = epsilon_end
            self.epsilon_decay = epsilon_decay
            
            self.Q = np.zeros((n_states, n_actions))
            self.training_steps = 0
        
        def select_action(self, state: int, training: bool = True) -> int:
            if training and np.random.random() < self.epsilon:
                return np.random.randint(self.n_actions)
            return np.argmax(self.Q[state])
        
        def update(self, state, action, reward, next_state, done):
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.Q[next_state])
            
            self.Q[state, action] += self.alpha * (target - self.Q[state, action])
            
            self.training_steps += 1
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        def save(self, path: str):
            np.save(path, {'Q': self.Q, 'epsilon': self.epsilon})
        
        def load(self, path: str):
            data = np.load(path, allow_pickle=True).item()
            self.Q = data['Q']
            self.epsilon = data['epsilon']
    
    class RLTrainer:
        def __init__(
            self,
            env,
            agent,
            n_episodes: int = 1000,
            eval_frequency: int = 100,
            n_eval_episodes: int = 10,
            max_steps: int = 200
        ):
            self.env = env
            self.agent = agent
            self.n_episodes = n_episodes
            self.eval_frequency = eval_frequency
            self.n_eval_episodes = n_eval_episodes
            self.max_steps = max_steps
            
            self.train_rewards = []
            self.eval_rewards = []
            self.best_eval_reward = float('-inf')
        
        def train_episode(self) -> float:
            state = self.env.reset()
            total_reward = 0
            
            for _ in range(self.max_steps):
                action = self.agent.select_action(state, training=True)
                next_state, reward, done = self.env.step(action)
                
                self.agent.update(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            return total_reward
        
        def evaluate(self) -> float:
            total_rewards = []
            
            for _ in range(self.n_eval_episodes):
                state = self.env.reset()
                episode_reward = 0
                
                for _ in range(self.max_steps):
                    action = self.agent.select_action(state, training=False)
                    next_state, reward, done = self.env.step(action)
                    
                    episode_reward += reward
                    state = next_state
                    
                    if done:
                        break
                
                total_rewards.append(episode_reward)
            
            return np.mean(total_rewards)
        
        def train(self) -> Dict:
            for episode in range(self.n_episodes):
                train_reward = self.train_episode()
                self.train_rewards.append(train_reward)
                
                # Evaluate periodically
                if (episode + 1) % self.eval_frequency == 0:
                    eval_reward = self.evaluate()
                    self.eval_rewards.append(eval_reward)
                    
                    print(f"Episode {episode + 1}: "
                          f"Train={np.mean(self.train_rewards[-100:]):.2f}, "
                          f"Eval={eval_reward:.2f}, "
                          f"ε={self.agent.epsilon:.3f}")
                    
                    # Save best model
                    if eval_reward > self.best_eval_reward:
                        self.best_eval_reward = eval_reward
            
            return {
                'train_rewards': self.train_rewards,
                'eval_rewards': self.eval_rewards,
                'best_eval_reward': self.best_eval_reward
            }
    
    # Run training
    env = SimpleEnv(n_states=20)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=2,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.99
    )
    
    trainer = RLTrainer(
        env=env,
        agent=agent,
        n_episodes=500,
        eval_frequency=100,
        n_eval_episodes=10
    )
    
    results = trainer.train()
    
    print(f"\nFinal Results:")
    print(f"  Best eval reward: {results['best_eval_reward']:.2f}")
    print(f"  Final train avg: {np.mean(results['train_rewards'][-100:]):.2f}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("REINFORCEMENT LEARNING: EXERCISES")
    print("=" * 70)
    
    np.random.seed(42)
    
    print("\n" + "=" * 70)
    print("Exercise 1: Value Iteration")
    print("=" * 70)
    solution1_value_iteration()
    
    print("\n" + "=" * 70)
    print("Exercise 2: Q-Learning with Exploration Decay")
    print("=" * 70)
    solution2_q_learning()
    
    print("\n" + "=" * 70)
    print("Exercise 3: SARSA(λ)")
    print("=" * 70)
    solution3_sarsa_lambda()
    
    print("\n" + "=" * 70)
    print("Exercise 4: REINFORCE with Baseline")
    print("=" * 70)
    solution4_reinforce_baseline()
    
    print("\n" + "=" * 70)
    print("Exercise 5: Actor-Critic")
    print("=" * 70)
    solution5_actor_critic()
    
    print("\n" + "=" * 70)
    print("Exercise 6: N-step TD")
    print("=" * 70)
    solution6_n_step_td()
    
    print("\n" + "=" * 70)
    print("Exercise 7: Double Q-Learning")
    print("=" * 70)
    solution7_double_q_learning()
    
    print("\n" + "=" * 70)
    print("Exercise 8: PPO (Simplified)")
    print("=" * 70)
    solution8_ppo()
    
    print("\n" + "=" * 70)
    print("Exercise 9: Replay Buffer")
    print("=" * 70)
    solution9_replay_buffer()
    
    print("\n" + "=" * 70)
    print("Exercise 10: Complete RL Pipeline")
    print("=" * 70)
    solution10_complete_pipeline()
    
    print("\n" + "=" * 70)
    print("All exercises completed!")
    print("=" * 70)
