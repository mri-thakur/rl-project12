"""
Innovation experiment: QRL + Eligibility Traces (TD(λ)).

Compares four algorithms on the 20×20 gridworld from Dong et al. (2008):
  1. TD(0) Q-learning       — baseline from paper
  2. QRL base               — paper's algorithm (corrected Grover)
  3. QRL + TD(λ=0.3)        — innovation: QRL with eligibility traces
  4. QRL + TD(λ=0.5)        — same, moderate λ
  5. QRL + TD(λ=0.7)        — same, higher λ
  6. QRL + TD(λ=0.9)        — same, near-MC credit assignment

Also produces a λ-sweep panel plot (raw steps, no smoothing) to show
how different trace decay rates affect the learning curve shape.

Run from anywhere:
    python src/experiment_innovation.py
    cd src && python experiment_innovation.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gridworld   import GridWorld
from qrl         import QRLAgent
from qrl_traces  import QRLTracesAgent
from td_agent    import TDAgent

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results")

N_EPISODES = 5_000
MAX_STEPS  = 5_000
K_VAL      = 0.01


# ── Shared runner ─────────────────────────────────────────────────────────────

def run(cls, kwargs, n_ep=N_EPISODES):
    env   = GridWorld()
    agent = cls(**kwargs)
    steps = []
    for _ in range(n_ep):
        if hasattr(agent, "reset_traces"):
            agent.reset_traces()
        state = env.reset()
        count = 0
        for _ in range(MAX_STEPS):
            action          = agent.select_action(state)
            ns, r, done     = env.step(action)
            agent.update(state, action, r, ns, done)
            state           = ns
            count          += 1
            if done:
                break
        steps.append(count)
    return steps


def smooth(data, window=100):
    return np.convolve(data, np.ones(window) / window, mode="valid")


# ── Comparison plot ───────────────────────────────────────────────────────────

def comparison():
    """Smoothed comparison: all algorithms on one axes."""
    print("=== Innovation: comparison plot ===")
    runs = [
        ("TD(0)  α=0.01 ε=0.01",
         TDAgent,        {"alpha": 0.01, "gamma": 0.99, "epsilon": 0.01}),
        ("QRL base  α=0.06",
         QRLAgent,       {"alpha": 0.06, "gamma": 0.99, "k": K_VAL}),
        ("QRL + TD(λ=0.3)",
         QRLTracesAgent, {"alpha": 0.06, "gamma": 0.99, "k": K_VAL, "lam": 0.3}),
        ("QRL + TD(λ=0.5)",
         QRLTracesAgent, {"alpha": 0.06, "gamma": 0.99, "k": K_VAL, "lam": 0.5}),
        ("QRL + TD(λ=0.7)",
         QRLTracesAgent, {"alpha": 0.06, "gamma": 0.99, "k": K_VAL, "lam": 0.7}),
        ("QRL + TD(λ=0.9)",
         QRLTracesAgent, {"alpha": 0.06, "gamma": 0.99, "k": K_VAL, "lam": 0.9}),
    ]

    styles = [
        dict(lw=1.8, linestyle="-"),
        dict(lw=1.8, linestyle="-"),
        dict(lw=1.5, linestyle="--"),
        dict(lw=1.5, linestyle="--"),
        dict(lw=1.5, linestyle="--"),
        dict(lw=1.5, linestyle="--"),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    for (label, cls, kwargs), style in zip(runs, styles):
        print(f"  {label} …")
        s = run(cls, kwargs)
        ax.plot(smooth(s), label=label, **style)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Steps per Episode  (smoothed, window=100)", fontsize=12)
    ax.set_title("Innovation: QRL with Eligibility Traces TD(λ)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "innovation_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}\n")


# ── Lambda sweep panel ────────────────────────────────────────────────────────

def lambda_sweep():
    """Raw per-episode steps for each λ value, side-by-side panels."""
    print("=== Innovation: λ sweep ===")
    lambdas = [0.0, 0.3, 0.5, 0.7, 0.9]
    fig, axes = plt.subplots(1, len(lambdas), figsize=(18, 4), sharey=True)

    for ax, lam in zip(axes, lambdas):
        print(f"  QRL+Traces λ={lam} …")
        s = run(QRLTracesAgent,
                {"alpha": 0.06, "gamma": 0.99, "k": K_VAL, "lam": lam})
        ax.plot(s, lw=0.4)
        ax.set_title(f"QRL + Traces  λ={lam}", fontsize=10)
        ax.set_xlabel("Episode", fontsize=9)
        ax.set_ylim(0, 5_500)

    axes[0].set_ylabel("Steps per Episode", fontsize=9)
    fig.suptitle("QRL + Eligibility Traces: Effect of λ", fontsize=13)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "innovation_lambda_sweep.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  → {out}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    comparison()
    lambda_sweep()
    print("Innovation experiments done.")
