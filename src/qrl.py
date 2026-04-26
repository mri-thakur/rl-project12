"""
Quantum Reinforcement Learning (QRL) — Classical Simulation
Dong et al., "Quantum Reinforcement Learning", IEEE Trans. SMC-B, 2008

GROVER ROTATION (Sec III-D, Eq. 36-40)
========================================
Action amplitudes are represented as an angle φ in the (|a⟩, |a⊥⟩) plane:

    c_a   = sin(φ)    amplitude of the chosen eigen-action
    c_⊥   = cos(φ)    norm of the orthogonal complement

Uniform initialisation (Eq. 28-29):
    φ₀ = θ = arcsin(1/√N_a)                                   (Eq. 37)

Each Grover iteration UGrov = U_{a0} · U_a INCREASES φ by 2θ (Fig. 1).
After L iterations, the rotation is applied as:

    new_c_a  = cos(2Lθ)·c_a + sin(2Lθ)·c_⊥    [= sin(φ + 2Lθ)]
    new_c_⊥  = cos(2Lθ)·c_⊥ − sin(2Lθ)·c_a    [= cos(φ + 2Lθ)]

Starting from uniform (φ = θ), after L iterations:
    c_a = sin((2L+1)θ)                                         (Eq. 40)

Verified: for N_a=4, θ=π/6, L=1 → c_a = sin(π/2) = 1.0 ✓

Number of iterations (Sec III-D):
    L = min( int(k · (r + V(s'))),  L_max )
    L_max = int(π / (4θ) − 0.5)    [prevents over-rotation past peak]

With k=1.0 and goal reward 100: L = min(int(1.0×100), L_max) = L_max for N_a=4.

Value update:  TD(0)  V(s) ← V(s) + α(r + γV(s') − V(s))    (Eq. 27)
Action select: collapse — sample from |C_a|²                  (Def. 3)
Backup register: restored after each collapse                  (Sec III-E)
"""

import numpy as np
from collections import defaultdict


class QRLAgent:
    def __init__(self, n_actions=4, alpha=0.06, gamma=0.99, k=1.0):
        """
        Parameters
        ----------
        n_actions : int
            Number of eigen-actions (4 for gridworld).
        alpha : float
            TD(0) learning rate.
        gamma : float
            Discount factor.
        k : float
            Maps signal (r + V(s')) to Grover iteration count:
            L = int(k · signal).  With k=1.0 and goal reward 100,
            L=L_max (clamped) for N_a=4.
        """
        self.n_actions = n_actions
        self.alpha     = alpha
        self.gamma     = gamma
        self.k         = k

        # θ = arcsin(1/√N_a)  — Eq. 37
        self.theta = np.arcsin(1.0 / np.sqrt(n_actions))
        # L_max = int(π/(4θ) − 0.5)  — prevents over-rotation past amplitude peak
        self.L_max = max(1, int(np.pi / (4.0 * self.theta) - 0.5))

        self.V             = defaultdict(float)
        self.amplitudes    = {}   # live register (collapses on observation)
        self.amplitudes_bk = {}   # backup register (Sec III-E)

    # ── Amplitude helpers ─────────────────────────────────────────────────────

    def _init_state(self, state):
        """Uniform superposition: every eigen-action equally likely (Eq. 28-29)."""
        amp = np.ones(self.n_actions) / np.sqrt(self.n_actions)
        self.amplitudes[state]    = amp.copy()
        self.amplitudes_bk[state] = amp.copy()

    def _get_amp(self, state):
        if state not in self.amplitudes:
            self._init_state(state)
        return self.amplitudes[state]

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(self, state):
        """
        Collapse postulate (Def. 3): sample an eigen-action from |C_a|².
        The backup register is restored immediately so the amplitude
        distribution is not destroyed by the collapse (Sec III-E).
        """
        amp   = self._get_amp(state)
        probs = amp ** 2
        probs /= probs.sum()                          # numerical safety
        action = np.random.choice(self.n_actions, p=probs)
        self.amplitudes[state] = self.amplitudes_bk[state].copy()
        return action

    # ── Grover rotation ───────────────────────────────────────────────────────

    def _grover_rotate(self, state, action, L):
        """
        Apply L Grover iterations for `action` in `state` (Fig. 1, Eq. 35-40).

        Decomposition:
            c_a   = amp[action]
            c_⊥   = √(1 − c_a²)    (norm of the N_a−1 other amplitudes)

        Rotation by +2Lθ (sign increases φ, reinforcing the action):
            new_c_a  = cos(2Lθ)·c_a + sin(2Lθ)·c_⊥
            new_c_⊥  = cos(2Lθ)·c_⊥ − sin(2Lθ)·c_a

        All other amplitudes are scaled proportionally to preserve ‖amp‖=1.
        """
        if L <= 0:
            return

        amp    = self._get_amp(state)
        c_a    = amp[action]
        c_perp = np.sqrt(max(0.0, 1.0 - c_a ** 2))

        # Clamp phi so we never over-rotate past the peak (pi/2).
        # phi is the current angle; target is phi + 2*L*theta, capped at pi/2.
        phi_current  = np.arcsin(np.clip(c_a, -1.0, 1.0))
        phi_target   = min(phi_current + 2.0 * self.theta * L, np.pi / 2.0)
        angle        = phi_target - phi_current   # actual rotation applied
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        new_c_a    = cos_a * c_a    + sin_a * c_perp
        new_c_perp = cos_a * c_perp - sin_a * c_a

        amp[action] = new_c_a

        # Scale the N_a−1 orthogonal amplitudes uniformly
        if c_perp > 1e-12:
            scale = new_c_perp / c_perp
            for j in range(self.n_actions):
                if j != action:
                    amp[j] *= scale
        else:
            # Degenerate: all amplitude was on this action — distribute evenly
            each = new_c_perp / np.sqrt(max(self.n_actions - 1, 1))
            for j in range(self.n_actions):
                if j != action:
                    amp[j] = each

        # Renormalise for numerical stability
        norm = np.linalg.norm(amp)
        if norm > 1e-12:
            amp /= norm

        self.amplitudes[state]    = amp
        self.amplitudes_bk[state] = amp.copy()

    # ── Learning update ───────────────────────────────────────────────────────

    def update(self, state, action, reward, next_state, done):
        """
        Single-step QRL update (Fig. 3):
          (a) TD(0) state-value update    — Eq. 27
          (b) Grover probability-amplitude update — Sec III-D
        """
        v_next = 0.0 if done else self.V[next_state]

        # (a) TD(0)
        td_error = reward + self.gamma * v_next - self.V[state]
        self.V[state] += self.alpha * td_error

        # (b) Grover update — only for positive reinforcement signal
        signal = reward + v_next
        if signal > 0:
            L = min(int(self.k * signal), self.L_max)
            self._grover_rotate(state, action, L)
