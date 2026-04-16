import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from gridworld import GridWorld
from td_agent import TDAgent
import numpy as np

# Quick check: TD should start high (~thousands) and be slow to converge
for a in [0.01, 0.02, 0.03]:
    env = GridWorld()
    ag = TDAgent(alpha=a, gamma=0.99, epsilon=0.01)
    s = []
    for ep in range(200):
        state = env.reset()
        steps = 0
        for _ in range(7000):
            act = ag.select_action(state)
            ns, r, done = env.step(act)
            ag.update(state, act, r, ns, done)
            state = ns
            steps += 1
            if done: break
        s.append(steps)
    print(f"TD a={a}: first5={np.mean(s[:5]):.0f} ep50={np.mean(s[45:55]):.0f} ep150={np.mean(s[145:155]):.0f}")
