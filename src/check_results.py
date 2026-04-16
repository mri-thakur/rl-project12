import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from gridworld import GridWorld
from qrl import QRLAgent
from td_agent import TDAgent
import numpy as np

def run(cls, kw, n_ep=1500):
    env = GridWorld()
    ag = cls(**kw)
    s = []
    for ep in range(n_ep):
        state = env.reset()
        steps = 0
        for _ in range(5000):
            a = ag.select_action(state)
            ns, r, done = env.step(a)
            ag.update(state, a, r, ns, done)
            state = ns
            steps += 1
            if done: break
        s.append(steps)
    return s

print("CLAIM 1+3 check")
print("\nTD alphas:")
for a in [0.01, 0.02, 0.03]:
    s = run(TDAgent, {'alpha':a,'gamma':0.99,'epsilon':0.01})
    print(f"  a={a}: first100={np.mean(s[:100]):.0f} last100={np.mean(s[-100:]):.0f}")

print("\nQRL vs TD:")
qrl = run(QRLAgent, {'alpha':0.06,'gamma':0.99,'k':0.1})
td = run(TDAgent, {'alpha':0.01,'gamma':0.99,'epsilon':0.01})
print(f"  QRL: first100={np.mean(qrl[:100]):.0f} last100={np.mean(qrl[-100:]):.0f}")
print(f"  TD:  first100={np.mean(td[:100]):.0f} last100={np.mean(td[-100:]):.0f}")
