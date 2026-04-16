"""
Replication of Dong et al., "Quantum Reinforcement Learning" (2008)

Fig 5: QRL vs TD + Theoretical (3 panels)
Fig 6: QRL alpha = 0.01 ~ 0.11 (3x3 grid)
Fig 7: TD alpha = 0.01, 0.02, 0.03
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from gridworld import GridWorld
from qrl import QRLAgent
from td_agent import TDAgent

N_EPISODES = 10000
MAX_STEPS = 3000
K_VAL = 0.02


def run(cls, kw, n_ep=N_EPISODES, max_steps=MAX_STEPS):
    env = GridWorld()
    ag = cls(**kw)
    steps = []
    for _ in range(n_ep):
        state = env.reset(); s = 0
        for _ in range(max_steps):
            a = ag.select_action(state)
            ns, r, done = env.step(a)
            ag.update(state, a, r, ns, done)
            state = ns; s += 1
            if done: break
        steps.append(s)
    return steps


class QRLTheo:
    """QRL Theoretical — 1 Grover iteration = perfect for 4 actions.
    theta=pi/6, one iteration rotates by 2*theta=pi/3.
    From equal superposition phi=pi/6 -> pi/6+pi/3=pi/2 -> prob=1.0
    """
    def __init__(self, n_actions=4, alpha=0.06, gamma=0.99):
        self.na = n_actions; self.alpha = alpha; self.gamma = gamma
        self.V = {}; self.amp = {}
        self.theta = np.arcsin(1.0 / np.sqrt(n_actions))

    def _a(self, s):
        if s not in self.amp:
            self.amp[s] = np.ones(self.na) / np.sqrt(self.na)
        return self.amp[s]

    def select_action(self, s):
        a = self._a(s); p = a**2; p /= p.sum()
        return np.random.choice(self.na, p=p)

    def update(self, s, a, r, ns, done):
        vn = 0.0 if done else self.V.get(ns, 0.0)
        self.V[s] = self.V.get(s, 0.0) + self.alpha * (r + self.gamma * vn - self.V.get(s, 0.0))
        if r + vn > 0:
            amp = self._a(s)
            c = np.clip(amp[a], -1.0, 1.0)
            phi = np.arcsin(c); cp = np.cos(phi)
            np2 = np.clip(phi + 2.0 * self.theta, -np.pi/2+0.01, np.pi/2-0.01)
            amp[a] = np.sin(np2)
            if abs(cp) > 1e-12:
                sc = np.cos(np2) / cp
                for j in range(self.na):
                    if j != a: amp[j] *= sc
            else:
                e = np.cos(np2) / np.sqrt(self.na - 1)
                for j in range(self.na):
                    if j != a: amp[j] = e
            n = np.sqrt(np.sum(amp**2))
            if n > 1e-12: amp[:] /= n


def exp1():
    """Fig 5: TD blue | QRL black | QRL Theoretical green"""
    print("=== Fig 5 ===")
    print("  TD..."); td = run(TDAgent, {'alpha':0.01,'gamma':0.99,'epsilon':0.01})
    print("  QRL..."); qrl = run(QRLAgent, {'alpha':0.06,'gamma':0.99,'k':K_VAL})
    print("  Theo..."); theo = run(QRLTheo, {'alpha':0.06,'gamma':0.99}, n_ep=25, max_steps=7000)

    fig, (a1,a2,a3) = plt.subplots(1,3,figsize=(18,5))
    a1.plot(td,'b',lw=0.3); a1.set_title('Performance of TD (epsilon = 0.01, alpha = 0.01)')
    a1.set_xlabel('Episodes'); a1.set_ylabel('Steps per episode'); a1.set_ylim(0,7000); a1.set_xlim(0,10000)
    a2.plot(qrl,'k',lw=0.3); a2.set_title('Performance of QRL (alpha = 0.06)')
    a2.set_xlabel('Episodes'); a2.set_ylabel('Steps per episode'); a2.set_ylim(0,7000); a2.set_xlim(0,10000)
    a3.plot(theo,'g',lw=0.8); a3.set_title('Performance of QRL (Theoretical results)')
    a3.set_xlabel('Episodes'); a3.set_ylabel('Steps per episode'); a3.set_ylim(0,7000); a3.set_xlim(0,25)
    fig.tight_layout(); fig.savefig('results/fig5_qrl_vs_td.png',dpi=150); plt.close(fig)
    print("  -> fig5\n")


def exp2():
    """Fig 6: QRL alphas 3x3"""
    print("=== Fig 6 ===")
    alphas = [0.01,0.02,0.03,0.05,0.06,0.07,0.08,0.10,0.11]
    fig, axes = plt.subplots(3,3,figsize=(15,12)); af = axes.flatten()
    for i,a in enumerate(alphas):
        print(f"  QRL a={a}...")
        s = run(QRLAgent, {'alpha':a,'gamma':0.99,'k':K_VAL})
        af[i].plot(s,'b',lw=0.3)
        af[i].set_title(f'Performance of QRL (alpha = {a})',fontsize=9)
        af[i].set_xlabel('Episodes',fontsize=8); af[i].set_ylabel('Steps per episode',fontsize=8)
        af[i].set_ylim(0,6000); af[i].set_xlim(0,10000); af[i].tick_params(labelsize=7)
    fig.suptitle('Figure 6: Comparison of QRL algorithms with different learning rates (alpha = 0.01 ~ 0.11)',fontsize=13)
    fig.tight_layout(); fig.savefig('results/fig6_qrl_alphas.png',dpi=150); plt.close(fig)
    print("  -> fig6\n")


def exp3():
    """Fig 7: TD alphas"""
    print("=== Fig 7 ===")
    alphas = [0.01,0.02,0.03]; colors = ['k','b','r']
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    for i,(a,c) in enumerate(zip(alphas,colors)):
        print(f"  TD a={a}...")
        s = run(TDAgent, {'alpha':a,'gamma':0.99,'epsilon':0.01})
        axes[i].plot(s,c,lw=0.3)
        axes[i].set_title(f'Performance of TD (epsilon = 0.01, alpha = {a})')
        axes[i].set_xlabel('Episodes'); axes[i].set_ylabel('Steps per episode')
        axes[i].set_xlim(0,10000)
    fig.suptitle('Figure 7: Comparison of TD(0) algorithms with different learning rates (alpha = 0.01, 0.02, 0.03)',fontsize=13)
    fig.tight_layout(); fig.savefig('results/fig7_td_alphas.png',dpi=150); plt.close(fig)
    print("  -> fig7\n")


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    exp1(); exp2(); exp3()
    print("Done.")
