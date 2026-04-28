"""
QRL with Eligibility Traces — TD(λ) Innovation
Dong et al. (2008) extension.

The base QRLAgent uses TD(0) for its value update:
    V(s) ← V(s) + α · δ

This agent replaces that with TD(λ) eligibility traces:
    e(s_t)  ← e(s_t) + 1                   (accumulate at current state)
    V(s_i)  ← V(s_i) + α · δ · e(s_i)     (update all traced states)
    e(s_i)  ← γ · λ · e(s_i)              (decay all traces)

where δ = r + γV(s') − V(s) is the TD error.

Key effect: credit is propagated backward through the episode's visited
states in a single update step, rather than waiting for V to propagate
one step per episode.  This gives faster convergence, especially early
in training when the agent rarely reaches the goal.

The Grover rotation (corrected, matching Eq. 40) is identical to QRLAgent.

Parameters
----------
lam : float
    Trace decay λ ∈ [0, 1].
    λ = 0  →  TD(0), equivalent to the base QRLAgent value update.
    λ = 1  →  full Monte-Carlo credit assignment.
    λ ≈ 0.5 typically gives the best tradeoff.
"""

import numpy as np
from collections import defaultdict


class QRLTracesAgent:
    def __init__(self, n_actions=4, alpha=0.06, gamma=0.99, k=1.0, lam=0.5):
        """
        Parameters
        ----------
        n_actions : int
        alpha     : float  TD learning rate.
        gamma     : float  Discount factor.
        k         : float  Grover signal-to-iterations scale (same as QRLAgent).
        lam       : float  Eligibility trace decay λ ∈ [0, 1].
        """
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.k         = k
        self.lam       = lam

        self.theta = np.arcsin(1.0 / np.sqrt(n_actions))
        self.L_max = max(1, int(np.pi / (4.0 * self.theta) - 0.5))

        self.V             = defaultdict(float)
        self.amplitudes    = {}
        self.amplitudes_bk = {}
        self.traces        = defaultdict(float)

    # ── Episode boundary ──────────────────────────────────────────────────────

    def reset_traces(self):
        """Must be called at the start of every episode."""
        self.traces = defaultdict(float)

    # ── Amplitude helpers ─────────────────────────────────────────────────────

    def _init_state(self, state):
        amp = np.ones(self.n_actions) / np.sqrt(self.n_actions)
        self.amplitudes[state]    = amp.copy()
        self.amplitudes_bk[state] = amp.copy()

    def _get_amp(self, state):
        if state not in self.amplitudes:
            self._init_state(state)
        return self.amplitudes[state]

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state):
        """Collapse postulate: sample from |C_a|², restore backup register."""
        amp   = self._get_amp(state)
        probs = amp ** 2
        probs /= probs.sum()
        action = np.random.choice(self.n_actions, p=probs)
        self.amplitudes[state] = self.amplitudes_bk[state].copy()
        return action

    # ── Grover rotation (identical to QRLAgent) ───────────────────────────────

    def _grover_rotate(self, state, action, L):
        """
        L Grover iterations in the (|a⟩, |a⊥⟩) plane (Eq. 35-40).
        Rotation direction increases φ (reinforces the action).
        """
        if L <= 0:
            return

        amp    = self._get_amp(state)
        c_a    = amp[action]
        c_perp = np.sqrt(max(0.0, 1.0 - c_a ** 2))

        phi_current = np.arcsin(np.clip(c_a, -1.0, 1.0))
        phi_target  = min(phi_current + 2.0 * self.theta * L, np.pi / 2.0)

        angle        = phi_target - phi_current
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        new_c_a    = cos_a * c_a    + sin_a * c_perp
        new_c_perp = cos_a * c_perp - sin_a * c_a

        amp[action] = new_c_a

        if c_perp > 1e-12:
            scale = new_c_perp / c_perp
            for j in range(self.n_actions):
                if j != action:
                    amp[j] *= scale
        else:
            each = new_c_perp / np.sqrt(max(self.n_actions - 1, 1))
            for j in range(self.n_actions):
                if j != action:
                    amp[j] = each

        norm = np.linalg.norm(amp)
        if norm > 1e-12:
            amp /= norm

        self.amplitudes[state]    = amp
        self.amplitudes_bk[state] = amp.copy()

    # ── Learning update ───────────────────────────────────────────────────────

    def update(self, state, action, reward, next_state, done):
        """
        TD(λ) value update + Grover amplitude rotation.
        Call reset_traces() at the start of each episode.
        """
        v_next   = 0.0 if done else self.V[next_state]
        td_error = reward + self.gamma * v_next - self.V[state]

        # Accumulate trace for the current state (replacing traces variant)
        self.traces[state] = min(self.traces.get(state, 0.0) + 1.0, 5.0)

        # Update V for all traced states; decay traces
        for s in list(self.traces.keys()):
            self.V[s] += self.alpha * td_error * self.traces[s]
            self.traces[s] *= self.gamma * self.lam
            if self.traces[s] < 1e-6:
                del self.traces[s]

        # Grover rotation — only for positive reinforcement signal
        signal = reward + v_next
        if signal > 0:
            L = min(int(self.k * signal), self.L_max)
            self._grover_rotate(state, action, L)
