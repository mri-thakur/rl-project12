"""
QRL with Eligibility Traces — Innovation over base QRL.

Replaces TD(0) value update with TD(lambda) using eligibility traces.
Everything else (amplitude representation, collapse selection,
Grover-inspired rotation) stays identical to base QRL.

TD(0):  V(s) <- V(s) + alpha * delta
TD(λ):  V(s_i) <- V(s_i) + alpha * delta * e(s_i)  for ALL states

where:
  delta = r + gamma*V(s') - V(s)       (TD error)
  e(s) <- e(s) + 1                      (accumulating trace)
  e(s_i) <- gamma * lambda * e(s_i)     (decay all traces)

This propagates reward information backward through recently visited
states in a single episode, instead of waiting for V to propagate
one step at a time over many episodes.
"""

import numpy as np
from collections import defaultdict


class QRLTracesAgent:
    def __init__(self, n_actions=4, alpha=0.06, gamma=0.99, k=0.1,
                 lam=0.5):
        """
        Same as QRLAgent plus:
            lam: eligibility trace decay parameter (lambda)
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.k = k
        self.lam = lam
        self.V = defaultdict(float)
        self.amplitudes = {}
        self.theta = np.arcsin(1.0 / np.sqrt(n_actions))
        # Eligibility traces — reset each episode
        self.traces = defaultdict(float)

    def reset_traces(self):
        """Call at the start of each episode."""
        self.traces = defaultdict(float)

    def _get_amplitudes(self, state):
        if state not in self.amplitudes:
            self.amplitudes[state] = np.ones(self.n_actions) / np.sqrt(self.n_actions)
        return self.amplitudes[state]

    def select_action(self, state):
        amp = self._get_amplitudes(state)
        probs = amp ** 2
        probs /= probs.sum()
        return np.random.choice(self.n_actions, p=probs)

    def _rotate_amplitude(self, state, action, delta_phi):
        amp = self._get_amplitudes(state)
        c_a = np.clip(amp[action], -1.0, 1.0)
        phi = np.arcsin(c_a)
        cos_phi = np.cos(phi)
        new_phi = np.clip(phi + delta_phi, -np.pi/2 + 0.01, np.pi/2 - 0.01)
        amp[action] = np.sin(new_phi)
        if abs(cos_phi) > 1e-12:
            scale = np.cos(new_phi) / cos_phi
            for j in range(self.n_actions):
                if j != action:
                    amp[j] *= scale
        else:
            each = np.cos(new_phi) / np.sqrt(self.n_actions - 1)
            for j in range(self.n_actions):
                if j != action:
                    amp[j] = each
        norm = np.sqrt(np.sum(amp ** 2))
        if norm > 1e-12:
            amp[:] = amp / norm
        self.amplitudes[state] = amp

    def update(self, state, action, reward, next_state, done):
        """TD(lambda) value update + Grover amplitude rotation."""
        v_next = 0.0 if done else self.V[next_state]
        td_error = reward + self.gamma * v_next - self.V[state]

        # Accumulate trace for current state
        self.traces[state] += 1.0

        # Update V(s) for ALL traced states
        for s in list(self.traces.keys()):
            self.V[s] += self.alpha * td_error * self.traces[s]
            # Decay trace
            self.traces[s] *= self.gamma * self.lam
            # Remove negligible traces
            if self.traces[s] < 1e-6:
                del self.traces[s]

        # Grover amplitude rotation — same as base QRL
        signal = reward + v_next
        if signal > 0:
            delta_phi = self.k * signal * 2.0 * self.theta
            self._rotate_amplitude(state, action, delta_phi)
