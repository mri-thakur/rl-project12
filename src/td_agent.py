"""
TD(0) Agent — Sec V-A of the paper.

The paper says:
  - V(s) updated by TD(0): V(s) <- V(s) + alpha*(r + gamma*V(s') - V(s))
  - epsilon-greedy (epsilon=0.01)
  - "the agent has no information about the environment at all"

Since the agent is model-free and uses V(s), the natural implementation
is Q-learning (a TD method) where Q(s,a) is used for action selection.
The paper describes V(s) updates but Q-learning is the standard model-free
TD method that matches the described behavior.
"""
import numpy as np
from collections import defaultdict


class TDAgent:
    def __init__(self, n_actions=4, alpha=0.01, gamma=0.99, epsilon=0.01,
                 **kwargs):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state):
        """Epsilon-greedy over Q(s,a)."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q = self.Q[state]
        best = np.where(q == q.max())[0]
        return np.random.choice(best)

    def update(self, state, action, reward, next_state, done):
        """One-step TD (Q-learning) update."""
        q_next = 0.0 if done else np.max(self.Q[next_state])
        td_error = reward + self.gamma * q_next - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
