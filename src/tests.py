"""
Test suite for QRL project.
Merges test_grover.py + test_env.py + test_quick.py into one file.

Run from the src/ directory:
    python tests.py

Tests
-----
T01  Gridworld BFS shortest path == 36 (matches paper Sec V-B)
T02  Goal is reachable from start
T03  Step into wall stays in place
T04  Goal step returns reward=+100 and done=True
T05  Non-goal step returns reward=-1 and done=False
T06  QRL uniform init: ‖amp‖² = 1
T07  Eq. 40: c_a saturates at 1.0 after L=1 rotation; clamped for L≥2 [L=1,2,3]
T08  Normalisation preserved after rotation  [L=1,2,3,5]
T09  L=0 leaves amplitudes unchanged
T10  Action probability increases after one positive rotation
T11  select_action always returns a valid index
T12  Backup register restored after collapse
T13  L_max cap: c_a > 0 after L_max iterations (no over-rotation)
T14  QRLAgent.update runs without error
T15  QRLTracesAgent: reset_traces clears traces dict
T16  QRLTracesAgent: traces accumulate across steps then decay
T17  TDAgent: Q values update after a positive reward step
T18  TDAgent: epsilon-greedy explores (non-greedy actions selected)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from collections import deque

from gridworld   import GridWorld, WALLS, START, GOAL
from qrl         import QRLAgent
from qrl_traces  import QRLTracesAgent
from td_agent    import TDAgent


# ── Helpers ───────────────────────────────────────────────────────────────────

def check(condition, label, detail=""):
    status = "PASS" if condition else "FAIL"
    msg = f"  [{status}] {label}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    return condition


def close(a, b, tol=1e-7):
    return abs(a - b) < tol


# ── Gridworld tests ───────────────────────────────────────────────────────────

def test_gridworld():
    print("\n── Gridworld ──────────────────────────────────────────────────")
    results = []
    env = GridWorld()

    # T01: BFS shortest path == 36
    dist = env.bfs_shortest_path()
    results.append(check(dist == 36, "T01  BFS shortest path = 36",
                         f"got {dist}"))

    # T02: goal reachable
    results.append(check(dist is not None, "T02  Goal is reachable from start"))

    # T03: stepping into a wall keeps agent in place
    env.reset()
    # From (1,1), going up hits row 0 (border wall)
    env.state = (1, 1)
    ns, _, _ = env.step(0)   # action 0 = up
    results.append(check(ns == (1, 1), "T03  Step into wall: agent stays in place",
                         f"ns={ns}"))

    # T04: reaching goal gives r=+100, done=True
    env.state = (17, 18)     # one step below goal
    ns, r, done = env.step(1)   # action 1 = down → (18,18)
    results.append(check(ns == GOAL and r == 100.0 and done,
                         "T04  Goal step: r=+100, done=True",
                         f"ns={ns} r={r} done={done}"))

    # T05: normal step gives r=-1, done=False
    env.reset()
    env.state = (1, 1)
    # Right is open from (1,1)
    ns, r, done = env.step(3)   # action 3 = right
    results.append(check(r == -1.0 and not done,
                         "T05  Normal step: r=-1, done=False",
                         f"r={r} done={done}"))

    return results


# ── QRL Grover rotation tests ─────────────────────────────────────────────────

def test_qrl():
    print("\n── QRL Agent ──────────────────────────────────────────────────")
    results = []
    n  = 4
    s  = "test_state"

    # T06: uniform init normalisation
    ag = QRLAgent(n_actions=n)
    amp = ag._get_amp(s)
    results.append(check(close(np.sum(amp**2), 1.0),
                         "T06  Uniform init: ‖amp‖² = 1",
                         f"got {np.sum(amp**2):.10f}"))

    # T07: Eq. 40 — c_a = sin((2L+1)θ) after L iters from uniform
    # NOTE: With clamping at π/2, c_a saturates at 1.0 after the first rotation
    # reaches the peak. For L=1: sin(3θ)=1.0. For L>1: clamping prevents further
    # rotation, so c_a stays at 1.0.
    theta = np.arcsin(1.0 / np.sqrt(n))
    
    ag2 = QRLAgent(n_actions=n)
    ag2._init_state(s)
    ag2._grover_rotate(s, 0, 1)
    got = ag2.amplitudes[s][0]
    exp = np.sin((2 * 1 + 1) * theta)  # L=1 should give sin(3θ) = 1.0
    results.append(check(close(got, exp),
                         f"T07  Eq.40 L=1: c_a = sin(3θ)",
                         f"got {got:.8f} expected {exp:.8f}"))
    
    # For L=2, clamping at π/2 prevents further rotation (already at peak)
    ag3 = QRLAgent(n_actions=n)
    ag3._init_state(s)
    ag3._grover_rotate(s, 0, 2)
    got2 = ag3.amplitudes[s][0]
    # After L=1, we're at π/2, so additional rotation is clamped.
    # This prevents the theoretical sin(5θ), keeping c_a ≈ 1.0
    results.append(check(close(got2, 1.0, tol=1e-6),
                         f"T07  Eq.40 L=2: c_a clamped at π/2 (≈1.0)",
                         f"got {got2:.8f}"))
    
    # For L=3, same clamping applies
    ag4 = QRLAgent(n_actions=n)
    ag4._init_state(s)
    ag4._grover_rotate(s, 0, 3)
    got3 = ag4.amplitudes[s][0]
    results.append(check(close(got3, 1.0, tol=1e-6),
                         f"T07  Eq.40 L=3: c_a clamped at π/2 (≈1.0)",
                         f"got {got3:.8f}"))

    # T08: normalisation preserved after rotation
    for L in [1, 2, 3, 5]:
        ag3 = QRLAgent(n_actions=n)
        ag3._init_state(s)
        ag3._grover_rotate(s, 1, L)
        norm_sq = np.sum(ag3.amplitudes[s]**2)
        results.append(check(close(norm_sq, 1.0, tol=1e-10),
                             f"T08  ‖amp‖² = 1 after L={L}",
                             f"got {norm_sq:.12f}"))

    # T09: L=0 leaves amplitudes unchanged
    ag4 = QRLAgent(n_actions=n)
    ag4._init_state(s)
    before = ag4.amplitudes[s].copy()
    ag4._grover_rotate(s, 0, 0)
    diff = np.max(np.abs(before - ag4.amplitudes[s]))
    results.append(check(close(diff, 0.0, tol=1e-15),
                         "T09  L=0: amplitudes unchanged",
                         f"max diff={diff:.2e}"))

    # T10: action probability increases after one rotation
    ag5 = QRLAgent(n_actions=n)
    ag5._init_state(s)
    p_before = ag5.amplitudes[s][0] ** 2
    ag5._grover_rotate(s, 0, 1)
    p_after = ag5.amplitudes[s][0] ** 2
    results.append(check(p_after > p_before,
                         "T10  Prob[action] increases after L=1 rotation",
                         f"{p_before:.4f} → {p_after:.4f}"))

    # T11: select_action returns valid index
    ag6 = QRLAgent(n_actions=n)
    seen = set(ag6.select_action(s) for _ in range(500))
    results.append(check(seen.issubset(set(range(n))),
                         "T11  select_action ⊂ valid indices",
                         f"seen={sorted(seen)}"))

    # T12: backup register restored after collapse
    ag7 = QRLAgent(n_actions=n)
    ag7._init_state(s)
    amp_pre = ag7.amplitudes[s].copy()
    _       = ag7.select_action(s)
    results.append(check(np.allclose(amp_pre, ag7.amplitudes[s]),
                         "T12  Backup register restored after collapse"))

    # T13: L_max cap — no over-rotation
    ag8 = QRLAgent(n_actions=n)
    ag8._init_state(s)
    ag8._grover_rotate(s, 0, ag8.L_max)
    c_a = ag8.amplitudes[s][0]
    results.append(check(c_a > 0,
                         f"T13  L_max={ag8.L_max}: c_a={c_a:.6f} > 0 (no over-rotation)"))

    # T14: update runs without error on a full step
    ag9 = QRLAgent(n_actions=n)
    try:
        ag9.update((1, 1), 0, -1.0, (1, 2), False)
        ag9.update((1, 2), 1, 100.0, (18, 18), True)
        ok = True
    except Exception as e:
        ok = False
    results.append(check(ok, "T14  QRLAgent.update runs without error"))

    return results


# ── QRL Traces tests ──────────────────────────────────────────────────────────

def test_qrl_traces():
    print("\n── QRL Traces Agent ───────────────────────────────────────────")
    results = []

    ag = QRLTracesAgent(lam=0.5)
    s  = "ts"

    # T15: reset_traces clears the dict
    ag.update(s, 0, -1.0, "ts2", False)
    ag.reset_traces()
    results.append(check(len(ag.traces) == 0,
                         "T15  reset_traces clears trace dict",
                         f"len={len(ag.traces)}"))

    # T16: traces accumulate then decay
    ag2 = QRLTracesAgent(lam=0.9)
    ag2.reset_traces()
    ag2.update("a", 0, -1.0, "b", False)
    after_1 = ag2.traces.get("a", 0.0)
    ag2.update("b", 1, -1.0, "c", False)
    after_2 = ag2.traces.get("a", 0.0)
    results.append(check(after_1 > 0 and after_2 < after_1,
                         "T16  Traces accumulate then decay",
                         f"after step1={after_1:.4f} after step2={after_2:.4f}"))

    return results


# ── TD Agent tests ────────────────────────────────────────────────────────────

def test_td():
    print("\n── TD Agent ───────────────────────────────────────────────────")
    results = []

    # T17: Q values update after a reward
    ag = TDAgent(alpha=0.1, gamma=0.99, epsilon=0.0)
    q_before = ag.Q[(1, 1)][3]
    ag.update((1, 1), 3, 100.0, (1, 2), True)
    q_after = ag.Q[(1, 1)][3]
    results.append(check(q_after > q_before,
                         "T17  Q value increases after positive reward",
                         f"{q_before:.4f} → {q_after:.4f}"))

    # T18: epsilon-greedy explores with epsilon=1.0
    ag2 = TDAgent(n_actions=4, epsilon=1.0)
    ag2.Q[(0, 0)][0] = 999.0   # make action 0 always greedy-best
    seen = set(ag2.select_action((0, 0)) for _ in range(200))
    results.append(check(len(seen) > 1,
                         "T18  ε-greedy explores (ε=1.0 sees multiple actions)",
                         f"seen={sorted(seen)}"))

    return results


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all():
    all_results = []
    all_results += test_gridworld()
    all_results += test_qrl()
    all_results += test_qrl_traces()
    all_results += test_td()

    total  = len(all_results)
    passed = sum(all_results)
    print(f"\n{'═'*52}")
    print(f"  {passed}/{total} tests passed")
    print(f"{'═'*52}")
    return passed == total


if __name__ == "__main__":
    print("QRL Test Suite")
    print("═" * 52)
    success = run_all()
    sys.exit(0 if success else 1)
