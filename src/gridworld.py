"""
20x20 Gridworld — Sec V, Fig. 4
Dong et al., "Quantum Reinforcement Learning" (IEEE Trans. SMC-B, 2008)

Coordinates: (row, col), row 0 = top, col 0 = left.
  S = (1,  1)  — start (top-left interior)
  G = (18, 18) — goal  (bottom-right interior)

Actions: 0=up, 1=down, 2=left, 3=right
Rewards: +100 at goal, -1 every other step.
Blocked actions leave the agent in place (paper Sec V-A).

Wall layout is pixel-sampled at 300 DPI directly from Fig. 4 of the paper,
then validated: BFS shortest path S→G = 36 steps, matching the paper's
stated result "costs 36 steps to the goal state G" (Sec V-B).
"""

import numpy as np
from collections import deque


# ── Internal walls (pixel-sampled from Fig. 4) ───────────────────────────────
# Border walls (row 0, row 19, col 0, col 19) are added programmatically.
_INTERNAL_WALLS = frozenset([
    # Row 1: vertical divider at col 7
    (1, 7),
    # Row 2: L-block top-left cols 1-4, divider col 7
    (2, 1), (2, 2), (2, 3), (2, 4), (2, 7),
    # Row 3: divider col 7
    (3, 7),
    # Row 4: divider col 7, horizontal block cols 11-14
    (4, 7), (4, 11), (4, 12), (4, 13), (4, 14),
    # Row 5
    (5, 7), (5, 11), (5, 12), (5, 13), (5, 14),
    # Row 6
    (6, 7), (6, 11), (6, 12), (6, 13), (6, 14),
    # Row 7: stub cols 5-7 joins divider; right block continues
    (7, 5), (7, 6), (7, 7), (7, 11), (7, 12), (7, 13), (7, 14),
    # Rows 8-9: open corridors
    # Row 10: long horizontal wall cols 2-9
    (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9),
    # Row 11: open
    # Rows 12-13: vertical wall col 15
    (12, 15), (13, 15),
    # Row 14: col 2, horizontal block cols 8-12, col 15
    (14, 2), (14, 8), (14, 9), (14, 10), (14, 11), (14, 12), (14, 15),
    # Row 15: col 2, col 15
    (15, 2), (15, 15),
    # Row 16: col 2, horizontal cols 3-7, col 15
    (16, 2), (16, 3), (16, 4), (16, 5), (16, 6), (16, 7), (16, 15),
    # Rows 17-18: col 15 continues
    (17, 15), (18, 15),
])


def _build_walls():
    walls = set()
    for i in range(20):
        walls.add((0,  i)); walls.add((19, i))
        walls.add((i,  0)); walls.add((i,  19))
    walls.update(_INTERNAL_WALLS)
    return frozenset(walls)


WALLS = _build_walls()
START = (1, 1)
GOAL  = (18, 18)

_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


class GridWorld:
    """20×20 gridworld matching Fig. 4 of Dong et al. 2008."""

    def __init__(self):
        self.size      = 20
        self.n_actions = 4
        self.start     = START
        self.goal      = GOAL
        self.blocked   = WALLS
        self.action_deltas = _DELTAS
        self.state     = START

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        dr, dc = _DELTAS[action]
        ns = (self.state[0] + dr, self.state[1] + dc)
        if ns in self.blocked:
            ns = self.state          # blocked: stay in place
        self.state = ns
        if self.state == self.goal:
            return self.state, 100.0, True
        return self.state, -1.0, False

    def get_all_states(self):
        return [(r, c)
                for r in range(self.size)
                for c in range(self.size)
                if (r, c) not in self.blocked]

    def render_ascii(self):
        lines = []
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                if   (r, c) == self.state:   row += "A"
                elif (r, c) == self.goal:     row += "G"
                elif (r, c) == self.start:    row += "S"
                elif (r, c) in self.blocked:  row += "█"
                else:                         row += "."
            lines.append(row)
        return "\n".join(lines)

    def bfs_shortest_path(self):
        """Returns the length of the shortest path S→G (should be 36)."""
        q   = deque([(self.start, 0)])
        vis = {self.start}
        while q:
            (r, c), d = q.popleft()
            if (r, c) == self.goal:
                return d
            for dr, dc in _DELTAS.values():
                ns = (r + dr, c + dc)
                if ns not in self.blocked and ns not in vis:
                    vis.add(ns)
                    q.append((ns, d + 1))
        return None  # unreachable
