"""
20x20 Gridworld — Sec V, Fig. 4
Dong et al., "Quantum Reinforcement Learning" (2008)

- 20x20 grid (0~19), Start S=(1,1), Goal G=(18,18)
- 4 eigen actions: up, down, left, right
- r=+100 at goal, r=-1 per step
- gamma=0.99, V(s)=0 init
- Optimal path = 36 steps (Sec V-B)
- "actions that would lead into a blocked cell are not executed"
"""
import numpy as np


class GridWorld:
    def __init__(self, size=20, start=(1, 1), goal=(18, 18)):
        self.size = size
        self.start = start
        self.goal = goal
        self.n_actions = 4
        self.action_deltas = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.blocked = self._build_walls()
        self.state = self.start

    def _build_walls(self):
        blocked = set()
        # Border walls
        for i in range(self.size):
            blocked.add((0, i))
            blocked.add((self.size - 1, i))
            blocked.add((i, 0))
            blocked.add((i, self.size - 1))

        # Internal walls — maze structure for optimal path = 36
        # Horizontal walls
        for c in range(2, 10):
            blocked.add((4, c))
        for c in range(5, 15):
            blocked.add((8, c))
        for c in range(1, 8):
            blocked.add((12, c))
        for c in range(10, 18):
            blocked.add((12, c))
        for c in range(3, 12):
            blocked.add((16, c))

        # Vertical walls
        for r in range(1, 6):
            blocked.add((r, 10))
        for r in range(6, 12):
            blocked.add((r, 14))
        for r in range(13, 18):
            blocked.add((r, 6))
        for r in range(9, 16):
            blocked.add((r, 3))

        # Extra to get optimal = 36
        blocked.add((8, 4))
        blocked.add((15, 12))

        return blocked

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        dr, dc = self.action_deltas[action]
        nr, nc = self.state[0] + dr, self.state[1] + dc
        ns = (nr, nc)
        if (nr < 0 or nr >= self.size or nc < 0 or nc >= self.size
                or ns in self.blocked):
            ns = self.state  # stay in place
        self.state = ns
        if self.state == self.goal:
            return self.state, 100.0, True
        return self.state, -1.0, False

    def get_all_states(self):
        return [(r, c) for r in range(self.size) for c in range(self.size)
                if (r, c) not in self.blocked]
