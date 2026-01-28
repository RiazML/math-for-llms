"""
Reinforcement Learning: Implementation Examples
==============================================

From-scratch implementations of fundamental RL algorithms.
"""

import numpy as np
from typing import Tuple, List, Dict, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


# =============================================================================
# Example 1: Grid World Environment
# =============================================================================

class GridWorld:
    """
    Simple grid world environment.
    
    States: 2D grid positions
    Actions: Up, Down, Left, Right (0, 1, 2, 3)
    """
    
    def __init__(
        self,
        size: Tuple[int, int] = (5, 5),
        start: Tuple[int, int] = (0, 0),
        goal: Tuple[int, int] = (4, 4),
        obstacles: List[Tuple[int, int]] = None
    ):
        self.height, self.width = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles or []
        
        self.n_states = self.height * self.width
        self.n_actions = 4
        
        # Action effects: [dy, dx]
        self.action_effects = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        self.state = None
        self.reset()
    
    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.width + pos[1]
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        return (state // self.width, state % self.width)
    
    def reset(self) -> int:
        """Reset environment."""
        self.state = self.start
        return self._pos_to_state(self.state)
    
    def step(self, action: int) -> Tuple[int, float, bool]:
        """
        Take action in environment.
        
        Returns:
            next_state, reward, done
        """
        dy, dx = self.action_effects[action]
        new_y = self.state[0] + dy
        new_x = self.state[1] + dx
        
        # Check bounds
        if 0 <= new_y < self.height and 0 <= new_x < self.width:
            new_pos = (new_y, new_x)
            if new_pos not in self.obstacles:
                self.state = new_pos
        
        # Check if goal reached
        done = self.state == self.goal
        
        # Reward
        if done:
            reward = 1.0
        elif self.state in self.obstacles:
            reward = -1.0
        else:
            reward = -0.01  # Small step penalty
        
        return self._pos_to_state(self.state), reward, done
    
    def get_transition_prob(
        self, state: int, action: int
    ) -> List[Tuple[int, float, float]]:
        """
        Get transition probabilities for DP algorithms.
        
        Returns:
            List of (next_state, probability, reward)
        """
        pos = self._state_to_pos(state)
        
        if pos == self.goal:
            return [(state, 1.0, 0.0)]
        
        dy, dx = self.action_effects[action]
        new_y = pos[0] + dy
        new_x = pos[1] + dx
        
        # Check bounds and obstacles
        if 0 <= new_y < self.height and 0 <= new_x < self.width:
            new_pos = (new_y, new_x)
            if new_pos not in self.obstacles:
                next_state = self._pos_to_state(new_pos)
                reward = 1.0 if new_pos == self.goal else -0.01
                return [(next_state, 1.0, reward)]
        
        # Stay in place
        return [(state, 1.0, -0.01)]


def example_gridworld():
    """Demonstrate GridWorld environment."""
    print("=" * 70)
    print("Example 1: GridWorld Environment")
    print("=" * 70)
    
    env = GridWorld(
        size=(4, 4),
        start=(0, 0),
        goal=(3, 3),
        obstacles=[(1, 1), (2, 2)]
    )
    
    print(f"Grid size: {env.height}x{env.width}")
    print(f"Start: {env.start}, Goal: {env.goal}")
    print(f"Obstacles: {env.obstacles}")
    
    # Random episode
    state = env.reset()
    total_reward = 0
    steps = 0
    
    print("\nRandom episode:")
    while steps < 50:
        action = np.random.randint(4)
        next_state, reward, done = env.step(action)
        total_reward += reward
        steps += 1
        
        if done:
            print(f"  Goal reached in {steps} steps!")
            break
    
    print(f"  Total reward: {total_reward:.2f}")


# =============================================================================
# Example 2: Value Iteration
# =============================================================================

def value_iteration(
    env: GridWorld,
    gamma: float = 0.99,
    theta: float = 1e-6,
    max_iter: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Value Iteration algorithm.
    
    V_{k+1}(s) = max_a sum_{s'} P(s'|s,a)[R(s,a,s') + gamma * V_k(s')]
    
    Returns:
        V: Optimal value function
        policy: Optimal policy
    """
    V = np.zeros(env.n_states)
    
    for iteration in range(max_iter):
        delta = 0
        V_new = np.zeros_like(V)
        
        for s in range(env.n_states):
            # Skip goal state
            pos = env._state_to_pos(s)
            if pos == env.goal:
                continue
            
            # Compute Q-values for all actions
            q_values = np.zeros(env.n_actions)
            
            for a in range(env.n_actions):
                transitions = env.get_transition_prob(s, a)
                for next_s, prob, reward in transitions:
                    q_values[a] += prob * (reward + gamma * V[next_s])
            
            V_new[s] = np.max(q_values)
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        
        if delta < theta:
            print(f"Value iteration converged in {iteration + 1} iterations")
            break
    
    # Extract policy
    policy = np.zeros(env.n_states, dtype=int)
    
    for s in range(env.n_states):
        q_values = np.zeros(env.n_actions)
        
        for a in range(env.n_actions):
            transitions = env.get_transition_prob(s, a)
            for next_s, prob, reward in transitions:
                q_values[a] += prob * (reward + gamma * V[next_s])
        
        policy[s] = np.argmax(q_values)
    
    return V, policy


def example_value_iteration():
    """Demonstrate Value Iteration."""
    print("\n" + "=" * 70)
    print("Example 2: Value Iteration")
    print("=" * 70)
    
    env = GridWorld(size=(4, 4), start=(0, 0), goal=(3, 3))
    
    V, policy = value_iteration(env, gamma=0.99)
    
    # Display value function
    print("\nOptimal Value Function:")
    V_grid = V.reshape(env.height, env.width)
    for row in V_grid:
        print("  " + " ".join(f"{v:6.2f}" for v in row))
    
    # Display policy
    action_symbols = ['↑', '↓', '←', '→']
    print("\nOptimal Policy:")
    for i in range(env.height):
        row = ""
        for j in range(env.width):
            s = i * env.width + j
            if (i, j) == env.goal:
                row += "  G  "
            else:
                row += f"  {action_symbols[policy[s]]}  "
        print(row)


# =============================================================================
# Example 3: Policy Iteration
# =============================================================================

def policy_iteration(
    env: GridWorld,
    gamma: float = 0.99,
    theta: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Policy Iteration algorithm.
    
    1. Policy Evaluation: Compute V^π
    2. Policy Improvement: π' = greedy(V^π)
    3. Repeat until π' = π
    
    Returns:
        V: Optimal value function
        policy: Optimal policy
    """
    # Initialize random policy
    policy = np.random.randint(0, env.n_actions, size=env.n_states)
    
    iteration = 0
    while True:
        # Policy Evaluation
        V = np.zeros(env.n_states)
        
        while True:
            delta = 0
            for s in range(env.n_states):
                pos = env._state_to_pos(s)
                if pos == env.goal:
                    continue
                
                v_old = V[s]
                a = policy[s]
                
                V[s] = 0
                transitions = env.get_transition_prob(s, a)
                for next_s, prob, reward in transitions:
                    V[s] += prob * (reward + gamma * V[next_s])
                
                delta = max(delta, abs(V[s] - v_old))
            
            if delta < theta:
                break
        
        # Policy Improvement
        policy_stable = True
        
        for s in range(env.n_states):
            old_action = policy[s]
            
            q_values = np.zeros(env.n_actions)
            for a in range(env.n_actions):
                transitions = env.get_transition_prob(s, a)
                for next_s, prob, reward in transitions:
                    q_values[a] += prob * (reward + gamma * V[next_s])
            
            policy[s] = np.argmax(q_values)
            
            if old_action != policy[s]:
                policy_stable = False
        
        iteration += 1
        
        if policy_stable:
            print(f"Policy iteration converged in {iteration} iterations")
            break
    
    return V, policy


def example_policy_iteration():
    """Demonstrate Policy Iteration."""
    print("\n" + "=" * 70)
    print("Example 3: Policy Iteration")
    print("=" * 70)
    
    env = GridWorld(size=(4, 4), start=(0, 0), goal=(3, 3))
    
    V, policy = policy_iteration(env, gamma=0.99)
    
    print("\nOptimal Value Function:")
    V_grid = V.reshape(env.height, env.width)
    for row in V_grid:
        print("  " + " ".join(f"{v:6.2f}" for v in row))


# =============================================================================
# Example 4: Q-Learning
# =============================================================================

class QLearning:
    """
    Q-Learning algorithm (off-policy TD control).
    
    Q(s,a) ← Q(s,a) + α[R + γ max_a' Q(s',a') - Q(s,a)]
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.Q = np.zeros((n_states, n_actions))
    
    def select_action(self, state: int, training: bool = True) -> int:
        """ε-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ):
        """Update Q-value."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state])
        
        # TD update
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
    
    def train(
        self,
        env: GridWorld,
        n_episodes: int = 1000
    ) -> List[float]:
        """Train Q-learning agent."""
        rewards = []
        
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            
            for _ in range(500):  # Max steps per episode
                action = self.select_action(state)
                next_state, reward, done = env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            rewards.append(total_reward)
        
        return rewards
    
    def get_policy(self) -> np.ndarray:
        """Extract greedy policy from Q-values."""
        return np.argmax(self.Q, axis=1)


def example_q_learning():
    """Demonstrate Q-Learning."""
    print("\n" + "=" * 70)
    print("Example 4: Q-Learning")
    print("=" * 70)
    
    env = GridWorld(size=(4, 4), start=(0, 0), goal=(3, 3))
    
    agent = QLearning(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1
    )
    
    rewards = agent.train(env, n_episodes=500)
    
    print(f"\nAverage reward (last 50 episodes): {np.mean(rewards[-50:]):.3f}")
    
    # Show learned Q-values for start state
    print("\nQ-values at start state:")
    action_names = ['Up', 'Down', 'Left', 'Right']
    start_state = env._pos_to_state(env.start)
    for a, name in enumerate(action_names):
        print(f"  {name}: {agent.Q[start_state, a]:.3f}")
    
    # Evaluate
    env.reset()
    steps = 0
    state = env._pos_to_state(env.start)
    
    path = [env.start]
    while steps < 20:
        action = agent.select_action(state, training=False)
        next_state, _, done = env.step(action)
        path.append(env._state_to_pos(next_state))
        state = next_state
        steps += 1
        if done:
            break
    
    print(f"\nLearned path ({len(path)-1} steps): {' → '.join(str(p) for p in path)}")


# =============================================================================
# Example 5: SARSA
# =============================================================================

class SARSA:
    """
    SARSA algorithm (on-policy TD control).
    
    Q(s,a) ← Q(s,a) + α[R + γ Q(s',a') - Q(s,a)]
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.Q = np.zeros((n_states, n_actions))
    
    def select_action(self, state: int, training: bool = True) -> int:
        """ε-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.Q[state])
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int,
        done: bool
    ):
        """Update Q-value using SARSA rule."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.Q[next_state, next_action]
        
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
    
    def train(
        self,
        env: GridWorld,
        n_episodes: int = 1000
    ) -> List[float]:
        """Train SARSA agent."""
        rewards = []
        
        for episode in range(n_episodes):
            state = env.reset()
            action = self.select_action(state)
            total_reward = 0
            
            for _ in range(500):
                next_state, reward, done = env.step(action)
                next_action = self.select_action(next_state)
                
                self.update(state, action, reward, next_state, next_action, done)
                
                total_reward += reward
                state = next_state
                action = next_action
                
                if done:
                    break
            
            rewards.append(total_reward)
        
        return rewards


def example_sarsa():
    """Demonstrate SARSA."""
    print("\n" + "=" * 70)
    print("Example 5: SARSA")
    print("=" * 70)
    
    env = GridWorld(size=(4, 4), start=(0, 0), goal=(3, 3))
    
    agent = SARSA(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=0.1
    )
    
    rewards = agent.train(env, n_episodes=500)
    
    print(f"\nAverage reward (last 50 episodes): {np.mean(rewards[-50:]):.3f}")
    
    # Compare Q-Learning vs SARSA
    q_agent = QLearning(env.n_states, env.n_actions, alpha=0.1, gamma=0.99, epsilon=0.1)
    q_rewards = q_agent.train(env, n_episodes=500)
    
    print(f"Q-Learning average (last 50): {np.mean(q_rewards[-50:]):.3f}")
    print(f"SARSA average (last 50): {np.mean(rewards[-50:]):.3f}")


# =============================================================================
# Example 6: TD(λ) with Eligibility Traces
# =============================================================================

class TDLambda:
    """
    TD(λ) for value function estimation with eligibility traces.
    
    e_t(s) = γλ e_{t-1}(s) + 1(S_t = s)
    V(s) ← V(s) + α δ_t e_t(s)
    """
    
    def __init__(
        self,
        n_states: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        lambda_: float = 0.9
    ):
        self.n_states = n_states
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        
        self.V = np.zeros(n_states)
    
    def evaluate_policy(
        self,
        env: GridWorld,
        policy: np.ndarray,
        n_episodes: int = 100
    ) -> np.ndarray:
        """Evaluate policy using TD(λ)."""
        for episode in range(n_episodes):
            state = env.reset()
            
            # Reset eligibility traces
            e = np.zeros(self.n_states)
            
            for _ in range(500):
                action = policy[state]
                next_state, reward, done = env.step(action)
                
                # TD error
                if done:
                    delta = reward - self.V[state]
                else:
                    delta = reward + self.gamma * self.V[next_state] - self.V[state]
                
                # Update eligibility trace (accumulating)
                e[state] += 1
                
                # Update all state values
                self.V += self.alpha * delta * e
                
                # Decay eligibility traces
                e *= self.gamma * self.lambda_
                
                state = next_state
                
                if done:
                    break
        
        return self.V


def example_td_lambda():
    """Demonstrate TD(λ)."""
    print("\n" + "=" * 70)
    print("Example 6: TD(λ) with Eligibility Traces")
    print("=" * 70)
    
    env = GridWorld(size=(4, 4), start=(0, 0), goal=(3, 3))
    
    # Use optimal policy from value iteration
    _, optimal_policy = value_iteration(env, gamma=0.99)
    
    # Compare TD(0) vs TD(λ)
    td0 = TDLambda(env.n_states, alpha=0.1, gamma=0.99, lambda_=0.0)
    V_td0 = td0.evaluate_policy(env, optimal_policy, n_episodes=100)
    
    td_lambda = TDLambda(env.n_states, alpha=0.1, gamma=0.99, lambda_=0.9)
    V_td_lambda = td_lambda.evaluate_policy(env, optimal_policy, n_episodes=100)
    
    # True values from DP
    V_true, _ = value_iteration(env, gamma=0.99)
    
    print("\nMSE from true values:")
    print(f"  TD(0): {np.mean((V_td0 - V_true)**2):.6f}")
    print(f"  TD(0.9): {np.mean((V_td_lambda - V_true)**2):.6f}")


# =============================================================================
# Example 7: REINFORCE (Policy Gradient)
# =============================================================================

class REINFORCE:
    """
    REINFORCE algorithm (Monte Carlo Policy Gradient).
    
    θ ← θ + α ∑_t γ^t G_t ∇_θ log π_θ(a_t|s_t)
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.01,
        gamma: float = 0.99
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        
        # Policy parameters (logits)
        self.theta = np.zeros((n_states, n_actions))
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def get_action_probs(self, state: int) -> np.ndarray:
        """Get action probabilities for state."""
        return self._softmax(self.theta[state])
    
    def select_action(self, state: int) -> int:
        """Sample action from policy."""
        probs = self.get_action_probs(state)
        return np.random.choice(self.n_actions, p=probs)
    
    def update(
        self,
        states: List[int],
        actions: List[int],
        rewards: List[float]
    ):
        """Update policy using episode trajectory."""
        T = len(rewards)
        
        # Compute returns
        G = np.zeros(T)
        G[T-1] = rewards[T-1]
        for t in range(T-2, -1, -1):
            G[t] = rewards[t] + self.gamma * G[t+1]
        
        # Policy gradient update
        for t in range(T):
            state = states[t]
            action = actions[t]
            
            probs = self.get_action_probs(state)
            
            # Gradient of log π(a|s) = e_a - π
            grad = -probs
            grad[action] += 1
            
            # Update
            self.theta[state] += self.alpha * (self.gamma ** t) * G[t] * grad
    
    def train(
        self,
        env: GridWorld,
        n_episodes: int = 1000
    ) -> List[float]:
        """Train REINFORCE agent."""
        rewards_history = []
        
        for episode in range(n_episodes):
            state = env.reset()
            states, actions, rewards = [], [], []
            
            for _ in range(500):
                action = self.select_action(state)
                next_state, reward, done = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                
                state = next_state
                
                if done:
                    break
            
            self.update(states, actions, rewards)
            rewards_history.append(sum(rewards))
        
        return rewards_history


def example_reinforce():
    """Demonstrate REINFORCE."""
    print("\n" + "=" * 70)
    print("Example 7: REINFORCE (Policy Gradient)")
    print("=" * 70)
    
    env = GridWorld(size=(4, 4), start=(0, 0), goal=(3, 3))
    
    agent = REINFORCE(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.01,
        gamma=0.99
    )
    
    rewards = agent.train(env, n_episodes=1000)
    
    print(f"\nAverage reward (last 50 episodes): {np.mean(rewards[-50:]):.3f}")
    
    # Show learned policy probabilities at start
    print("\nPolicy at start state:")
    action_names = ['Up', 'Down', 'Left', 'Right']
    start_state = env._pos_to_state(env.start)
    probs = agent.get_action_probs(start_state)
    for a, name in enumerate(action_names):
        print(f"  {name}: {probs[a]:.3f}")


# =============================================================================
# Example 8: Actor-Critic
# =============================================================================

class ActorCritic:
    """
    Actor-Critic algorithm.
    
    Critic: V_w(s) - value function approximation
    Actor: π_θ(a|s) - policy
    
    Advantage: A(s,a) ≈ r + γV(s') - V(s)
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha_actor: float = 0.01,
        alpha_critic: float = 0.1,
        gamma: float = 0.99
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha_actor = alpha_actor
        self.alpha_critic = alpha_critic
        self.gamma = gamma
        
        # Actor parameters (policy logits)
        self.theta = np.zeros((n_states, n_actions))
        
        # Critic parameters (value function)
        self.w = np.zeros(n_states)
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    
    def get_action_probs(self, state: int) -> np.ndarray:
        return self._softmax(self.theta[state])
    
    def select_action(self, state: int) -> int:
        probs = self.get_action_probs(state)
        return np.random.choice(self.n_actions, p=probs)
    
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool
    ):
        """Update actor and critic."""
        # TD error (advantage estimate)
        if done:
            delta = reward - self.w[state]
        else:
            delta = reward + self.gamma * self.w[next_state] - self.w[state]
        
        # Critic update
        self.w[state] += self.alpha_critic * delta
        
        # Actor update
        probs = self.get_action_probs(state)
        grad = -probs
        grad[action] += 1
        self.theta[state] += self.alpha_actor * delta * grad
    
    def train(
        self,
        env: GridWorld,
        n_episodes: int = 1000
    ) -> List[float]:
        """Train Actor-Critic agent."""
        rewards_history = []
        
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            
            for _ in range(500):
                action = self.select_action(state)
                next_state, reward, done = env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            rewards_history.append(total_reward)
        
        return rewards_history


def example_actor_critic():
    """Demonstrate Actor-Critic."""
    print("\n" + "=" * 70)
    print("Example 8: Actor-Critic")
    print("=" * 70)
    
    env = GridWorld(size=(4, 4), start=(0, 0), goal=(3, 3))
    
    agent = ActorCritic(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha_actor=0.01,
        alpha_critic=0.1,
        gamma=0.99
    )
    
    rewards = agent.train(env, n_episodes=500)
    
    print(f"\nAverage reward (last 50 episodes): {np.mean(rewards[-50:]):.3f}")
    
    # Show learned value function
    print("\nLearned Value Function:")
    V_grid = agent.w.reshape(env.height, env.width)
    for row in V_grid:
        print("  " + " ".join(f"{v:6.2f}" for v in row))


# =============================================================================
# Example 9: DQN (Simplified)
# =============================================================================

class SimpleDQN:
    """
    Simplified Deep Q-Network with experience replay.
    
    Uses a linear function approximator for simplicity.
    Key DQN concepts: experience replay and target network.
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 32,
        alpha: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update: int = 100
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Q-network (simple 2-layer MLP)
        self.W1 = np.random.randn(state_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, n_actions) * 0.1
        self.b2 = np.zeros(n_actions)
        
        # Target network (copy of Q-network)
        self.W1_target = self.W1.copy()
        self.b1_target = self.b1.copy()
        self.W2_target = self.W2.copy()
        self.b2_target = self.b2.copy()
        
        # Experience replay buffer
        self.buffer = []
        self.buffer_size = buffer_size
        
        self.step_count = 0
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def _forward(
        self, state: np.ndarray, use_target: bool = False
    ) -> np.ndarray:
        """Forward pass through Q-network."""
        if use_target:
            W1, b1, W2, b2 = self.W1_target, self.b1_target, self.W2_target, self.b2_target
        else:
            W1, b1, W2, b2 = self.W1, self.b1, self.W2, self.b2
        
        h = self._relu(state @ W1 + b1)
        return h @ W2 + b2
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """ε-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        q_values = self._forward(state.reshape(1, -1))[0]
        return np.argmax(q_values)
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store transition in replay buffer."""
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))
    
    def update(self):
        """Update Q-network using sampled batch."""
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        # Compute targets using target network
        next_q = self._forward(next_states, use_target=True)
        targets = rewards + self.gamma * np.max(next_q, axis=1) * (1 - dones)
        
        # Forward pass
        h = self._relu(states @ self.W1 + self.b1)
        q_values = h @ self.W2 + self.b2
        
        # Compute gradients (only for selected actions)
        q_selected = q_values[np.arange(self.batch_size), actions]
        td_error = targets - q_selected
        
        # Gradient for W2, b2
        dq = np.zeros_like(q_values)
        dq[np.arange(self.batch_size), actions] = -td_error
        
        dW2 = h.T @ dq / self.batch_size
        db2 = np.mean(dq, axis=0)
        
        # Gradient for W1, b1
        dh = dq @ self.W2.T
        dh *= (h > 0)  # ReLU derivative
        
        dW1 = states.T @ dh / self.batch_size
        db1 = np.mean(dh, axis=0)
        
        # Update
        self.W2 -= self.alpha * dW2
        self.b2 -= self.alpha * db2
        self.W1 -= self.alpha * dW1
        self.b1 -= self.alpha * db1
        
        # Update target network periodically
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.W1_target = self.W1.copy()
            self.b1_target = self.b1.copy()
            self.W2_target = self.W2.copy()
            self.b2_target = self.b2.copy()
    
    def train(
        self,
        env: GridWorld,
        n_episodes: int = 500
    ) -> List[float]:
        """Train DQN agent."""
        rewards_history = []
        
        for episode in range(n_episodes):
            state_idx = env.reset()
            state = np.zeros(env.n_states)
            state[state_idx] = 1  # One-hot encoding
            
            total_reward = 0
            
            for _ in range(500):
                action = self.select_action(state)
                next_state_idx, reward, done = env.step(action)
                
                next_state = np.zeros(env.n_states)
                next_state[next_state_idx] = 1
                
                self.store_transition(state, action, reward, next_state, done)
                self.update()
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
            
            rewards_history.append(total_reward)
        
        return rewards_history


def example_dqn():
    """Demonstrate simplified DQN."""
    print("\n" + "=" * 70)
    print("Example 9: Simplified DQN")
    print("=" * 70)
    
    env = GridWorld(size=(4, 4), start=(0, 0), goal=(3, 3))
    
    agent = SimpleDQN(
        state_dim=env.n_states,  # One-hot encoded states
        n_actions=env.n_actions,
        hidden_dim=32,
        alpha=0.001,
        gamma=0.99,
        epsilon=0.1,
        buffer_size=5000,
        batch_size=32,
        target_update=50
    )
    
    rewards = agent.train(env, n_episodes=500)
    
    print(f"\nAverage reward (last 50 episodes): {np.mean(rewards[-50:]):.3f}")
    print(f"Buffer size: {len(agent.buffer)}")


# =============================================================================
# Example 10: Multi-Armed Bandit with UCB
# =============================================================================

class MultiArmedBandit:
    """
    Multi-armed bandit environment.
    
    Each arm has a fixed (unknown) probability of success.
    """
    
    def __init__(self, n_arms: int, probabilities: np.ndarray = None):
        self.n_arms = n_arms
        if probabilities is None:
            self.probabilities = np.random.random(n_arms)
        else:
            self.probabilities = probabilities
    
    def pull(self, arm: int) -> float:
        """Pull an arm and get reward."""
        if np.random.random() < self.probabilities[arm]:
            return 1.0
        return 0.0


class UCB:
    """
    Upper Confidence Bound algorithm.
    
    a_t = argmax_a [Q(a) + c * sqrt(ln(t) / N(a))]
    """
    
    def __init__(self, n_arms: int, c: float = 2.0):
        self.n_arms = n_arms
        self.c = c
        
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0
    
    def select_arm(self) -> int:
        """Select arm using UCB."""
        self.t += 1
        
        # Initialize: try each arm once
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        # UCB selection
        ucb_values = self.values + self.c * np.sqrt(np.log(self.t) / self.counts)
        return np.argmax(ucb_values)
    
    def update(self, arm: int, reward: float):
        """Update arm statistics."""
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class EpsilonGreedy:
    """ε-greedy bandit algorithm."""
    
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    
    def select_arm(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return np.argmax(self.values)
    
    def update(self, arm: int, reward: float):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


def example_bandits():
    """Demonstrate multi-armed bandits."""
    print("\n" + "=" * 70)
    print("Example 10: Multi-Armed Bandits (UCB vs ε-greedy)")
    print("=" * 70)
    
    np.random.seed(42)
    
    n_arms = 10
    true_probs = np.random.random(n_arms)
    
    print(f"\nTrue arm probabilities: {true_probs.round(3)}")
    print(f"Best arm: {np.argmax(true_probs)} (prob: {np.max(true_probs):.3f})")
    
    # Run experiment
    n_steps = 1000
    
    bandit = MultiArmedBandit(n_arms, true_probs)
    
    ucb = UCB(n_arms, c=2.0)
    eps_greedy = EpsilonGreedy(n_arms, epsilon=0.1)
    
    ucb_rewards = []
    eps_rewards = []
    
    for _ in range(n_steps):
        # UCB
        arm = ucb.select_arm()
        reward = bandit.pull(arm)
        ucb.update(arm, reward)
        ucb_rewards.append(reward)
        
        # ε-greedy
        arm = eps_greedy.select_arm()
        reward = bandit.pull(arm)
        eps_greedy.update(arm, reward)
        eps_rewards.append(reward)
    
    print(f"\nTotal rewards over {n_steps} steps:")
    print(f"  UCB: {sum(ucb_rewards):.0f}")
    print(f"  ε-greedy: {sum(eps_rewards):.0f}")
    
    print(f"\nEstimated arm values (UCB):")
    print(f"  {ucb.values.round(3)}")
    print(f"  Arm pulls: {ucb.counts}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("REINFORCEMENT LEARNING: IMPLEMENTATION EXAMPLES")
    print("=" * 70)
    
    np.random.seed(42)
    
    example_gridworld()
    example_value_iteration()
    example_policy_iteration()
    example_q_learning()
    example_sarsa()
    example_td_lambda()
    example_reinforce()
    example_actor_critic()
    example_dqn()
    example_bandits()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
