"""
Quantum Reinforcement Learning (QRL) — Classical simulation
Dong et al., "Quantum Reinforcement Learning", IEEE Trans. SMC-B, 2008

Classical simulation of QRL (Sec IV-B2):
  "In this paper most work has been done to develop this kind of novel
   QRL algorithms by simulating on the traditional computer."

Key components:
  1. Amplitude representation: |a_s> = sum C_a |a>, |C_a|^2 = prob  (Eq. 18-24)
  2. Collapse action selection: sample from |C_a|^2                  (Def. 3)
  3. TD(0) value update: V(s) <- V(s) + alpha*(r+gamma*V(s')-V(s))  (Eq. 27)
  4. Grover-inspired amplitude rotation proportional to reward signal (Sec III-D)
     delta_phi = k * (r + V(s')) * 2 * theta
     where theta = arcsin(1/sqrt(N_a))                              (Eq. 37)
  5. Two registers per state — backup against collapse               (Sec III-E)
  6. Normalization: sum|C_a|^2 = 1                                   (Eq. 24)
"""

import numpy as np
from collections import defaultdict


class QRLAgent:
    def __init__(self, n_actions=4, alpha=0.06, gamma=0.99, k=0.01):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.k = k
        self.V = defaultdict(float)
        self.amplitudes = {}
        self.amplitudes_bk = {}
        self.theta = np.arcsin(1.0 / np.sqrt(n_actions))

    def _get_amplitudes(self, state):
        if state not in self.amplitudes:
            amp = np.ones(self.n_actions) / np.sqrt(self.n_actions)
            self.amplitudes[state] = amp.copy()
            self.amplitudes_bk[state] = amp.copy()
        return self.amplitudes[state]

    def select_action(self, state):
        """Collapse postulate (Def. 3): sample from |C_a|^2.
        Restore from backup register after collapse (Sec III-E)."""
        amp = self._get_amplitudes(state)
        probs = amp ** 2
        probs /= probs.sum()
        action = np.random.choice(self.n_actions, p=probs)
        self.amplitudes[state] = self.amplitudes_bk[state].copy()
        return action

    def _rotate(self, state, action, delta_phi):
        """Grover-inspired rotation in |a>, |a_perp> plane (Eq. 38-40)."""
        amp = self._get_amplitudes(state)
        c_a = np.clip(amp[action], -1.0, 1.0)
        phi = np.arcsin(c_a)
        cos_phi = np.cos(phi)

        new_phi = np.clip(phi + delta_phi,
                          -np.pi / 2 + 0.01, np.pi / 2 - 0.01)
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
            amp[:] /= norm
        self.amplitudes[state] = amp
        self.amplitudes_bk[state] = amp.copy()

    def update(self, state, action, reward, next_state, done):
        """QRL update (Fig. 3):
        1. TD(0) value update
        2. Amplitude rotation proportional to signal
        """
        v_next = 0.0 if done else self.V[next_state]
        td_error = reward + self.gamma * v_next - self.V[state]
        self.V[state] += self.alpha * td_error

        signal = reward + v_next
        if signal > 0:
            delta_phi = self.k * signal * 2.0 * self.theta
            self._rotate(state, action, delta_phi)
