"""
Innovation experiment: QRL + Eligibility Traces vs base QRL vs TD.

Compares:
1. QRL (base) — paper's method
2. QRL + TD(lambda) — our innovation
3. TD(0) Q-learning — baseline

Tests multiple lambda values to find the best.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from gridworld import GridWorld
from qrl import QRLAgent
from qrl_traces import QRLTracesAgent
from td_agent import TDAgent

N_EPISODES = 5000
MAX_STEPS = 5000


def run_single(cls, kwargs, n_ep=N_EPISODES):
    env = GridWorld()
    ag = cls(**kwargs)
    steps = []
    for ep in range(n_ep):
        if hasattr(ag, 'reset_traces'):
            ag.reset_traces()
        state = env.reset()
        s = 0
        for _ in range(MAX_STEPS):
            a = ag.select_action(state)
            ns, r, done = env.step(a)
            ag.update(state, a, r, ns, done)
            state = ns
            s += 1
            if done: break
        steps.append(s)
    return steps


def smooth(d, w=50):
    return np.convolve(d, np.ones(w)/w, mode='valid')


def exp_comparison():
    """QRL vs QRL+traces vs TD."""
    print("=== Innovation: QRL vs QRL+Traces vs TD ===")

    print("  TD...")
    td = run_single(TDAgent, {'alpha':0.01,'gamma':0.99,'epsilon':0.01})
    print("  QRL base...")
    qrl = run_single(QRLAgent, {'alpha':0.06,'gamma':0.99,'k':0.1})
    print("  QRL+traces λ=0.5...")
    qrl_t5 = run_single(QRLTracesAgent, {'alpha':0.06,'gamma':0.99,'k':0.1,'lam':0.5})
    print("  QRL+traces λ=0.7...")
    qrl_t7 = run_single(QRLTracesAgent, {'alpha':0.06,'gamma':0.99,'k':0.1,'lam':0.7})
    print("  QRL+traces λ=0.9...")
    qrl_t9 = run_single(QRLTracesAgent, {'alpha':0.06,'gamma':0.99,'k':0.1,'lam':0.9})

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(smooth(td), label='TD(0) Q-learning (α=0.01)', lw=1.5, alpha=0.8)
    ax.plot(smooth(qrl), label='QRL base (α=0.06)', lw=1.5, alpha=0.8)
    ax.plot(smooth(qrl_t5), label='QRL + traces λ=0.5', lw=1.5, alpha=0.8)
    ax.plot(smooth(qrl_t7), label='QRL + traces λ=0.7', lw=1.5, alpha=0.8)
    ax.plot(smooth(qrl_t9), label='QRL + traces λ=0.9', lw=1.5, alpha=0.8)
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Steps per Episode', fontsize=12)
    ax.set_title('Innovation: QRL with Eligibility Traces', fontsize=14)
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('results/innovation_comparison.png', dpi=150)
    plt.close(fig)
    print("  -> results/innovation_comparison.png\n")


def exp_lambda_sweep():
    """Sweep lambda values for QRL+traces."""
    print("=== Lambda sweep ===")
    lambdas = [0.0, 0.3, 0.5, 0.7, 0.9]
    fig, ax = plt.subplots(figsize=(10, 6))
    for lam in lambdas:
        print(f"  λ={lam}...")
        s = run_single(QRLTracesAgent, {'alpha':0.06,'gamma':0.99,'k':0.1,'lam':lam})
        ax.plot(smooth(s), label=f'λ={lam}', lw=1.2)
    ax.set_xlabel('Episode'); ax.set_ylabel('Steps per Episode')
    ax.set_title('QRL + Traces: Effect of λ'); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('results/innovation_lambda_sweep.png', dpi=150)
    plt.close(fig)
    print("  -> results/innovation_lambda_sweep.png\n")


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    exp_comparison()
    exp_lambda_sweep()
    print("Innovation experiments done.")
