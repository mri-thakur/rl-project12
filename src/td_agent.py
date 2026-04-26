"""
TD(0) Agent with epsilon-greedy action selection.
Dong et al., "Quantum Reinforcement Learning" (IEEE Trans. SMC-B, 2008) — Sec V-A.

The paper describes:
  - Value update: V(s) ← V(s) + α(r + γV(s') − V(s))  [TD(0)]
  - Action selection: ε-greedy with ε = 0.01
  - "the agent has no information about the environment at all" [model-free]

Since the agent is model-free and needs to select actions without a model,
we maintain Q(s, a) instead of V(s) and use Q-learning updates.
This is the standard model-free TD method that matches the paper's intent:
the update rule reduces to the same TD(0) form, just on Q values.

Paper best case: α = 0.01, ε = 0.01 (Sec V-B, Fig. 5).
"""

import numpy as np
from collections import defaultdict


class TDAgent:
    def __init__(self, n_actions=4, alpha=0.01, gamma=0.99, epsilon=0.01, **kwargs):
        """
        Parameters
        ----------
        n_actions : int
        alpha     : float  TD learning rate.
        gamma     : float  Discount factor.
        epsilon   : float  Exploration probability for ε-greedy.
        """
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.Q         = defaultdict(lambda: np.zeros(n_actions))

    def select_action(self, state):
        """ε-greedy action selection over Q(s, a)."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q    = self.Q[state]
        best = np.where(q == q.max())[0]
        return np.random.choice(best)   # break ties randomly

    def update(self, state, action, reward, next_state, done):
        """One-step Q-learning (TD) update."""
        q_next   = 0.0 if done else np.max(self.Q[next_state])
        td_error = reward + self.gamma * q_next - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error
