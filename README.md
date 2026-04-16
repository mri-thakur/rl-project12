# Quantum Reinforcement Learning — Paper Reproduction

Reproduction of the results from:

> D. Dong, C. Chen, H. Li, and T.-J. Tarn, "Quantum Reinforcement Learning,"
> *IEEE Transactions on Systems, Man, and Cybernetics — Part B: Cybernetics*, vol. 38, no. 5, pp. 1207–1220, 2008.
> [arXiv:0810.3828](https://arxiv.org/abs/0810.3828)

## Overview

This project implements the Quantum Reinforcement Learning (QRL) algorithm proposed by Dong et al. and reproduces the three main experimental figures from the paper (Figures 5, 6, and 7). The QRL algorithm combines quantum computation concepts — state superposition, collapse-based action selection, and Grover-inspired amplitude amplification — with classical TD(0) reinforcement learning in a 20×20 gridworld environment.

## Reproduced Figures

| Figure | Description |
|--------|-------------|
| **Fig 5** | QRL vs TD(0) performance comparison + QRL theoretical results on a quantum computer |
| **Fig 6** | QRL with different learning rates (α = 0.01 ~ 0.11), 3×3 subplot grid |
| **Fig 7** | TD(0) with different learning rates (α = 0.01, 0.02, 0.03) |

## Environment Setup

**20×20 Gridworld** (Sec V, Fig. 4 of the paper):
- Grid size: 20×20 (indices 0–19), with border walls
- Start: S = (1, 1), Goal: G = (18, 18)
- Reward: r = +100 at goal, r = −1 per step
- Discount factor: γ = 0.99
- Optimal path length: 36 steps
- Internal walls create corridors and dead-ends matching the paper's maze structure

## Algorithms

### QRL Agent (`src/qrl.py`)
Classical simulation of the QRL framework (Sec III–IV of the paper):
- **Representation**: Action amplitudes |C_a|² as probabilities (Eq. 18–24)
- **Action selection**: Collapse postulate — sample from |C_a|² (Def. 3, Eq. 26)
- **Value update**: TD(0) — V(s) ← V(s) + α(r + γV(s') − V(s)) (Eq. 27)
- **Amplitude update**: Grover-inspired rotation proportional to reward signal (Sec III-D, Eq. 37–40)
  - θ = arcsin(1/√N_a), rotation Δφ = k·(r + V(s'))·2θ
- **Backup register**: Two copies of amplitudes per state to prevent collapse memory loss (Sec III-E)
- **Normalization**: Σ|C_a|² = 1 after each update (Eq. 24)

### TD(0) Agent (`src/td_agent.py`)
Standard Q-learning with ε-greedy action selection (Sec V-A):
- ε-greedy policy with ε = 0.01
- One-step TD update for Q(s, a)

### QRL Theoretical (`src/experiment.py :: QRLTheo`)
Simulates the expected performance on a real quantum computer (Sec IV-B1):
- With 4 actions, θ = π/6
- One Grover iteration rotates by 2θ = π/3
- From equal superposition (φ = π/6), one iteration gives φ = π/2 → probability 1.0
- Result: near-instant convergence within ~5 episodes

## Paper Claims Reproduced

1. **QRL converges faster than TD**: QRL (α = 0.06) converges within ~2000 episodes while TD (α = 0.01) takes the full 10000 episodes (Fig 5)
2. **QRL is robust to learning rate**: QRL works well for 0.02 ≤ α ≤ 0.10, converging faster at higher α (Fig 6). At α = 0.01 it learns too slowly; at α = 0.11 it becomes unstable
3. **TD is sensitive to learning rate**: TD converges at α = 0.01 but struggles at α = 0.02 and α = 0.03 (Fig 7)
4. **Theoretical QRL converges near-instantly**: With true quantum Grover iterations, the agent converges in ~5 episodes (Fig 5, right panel)

## Project Structure

```
├── src/
│   ├── gridworld.py              # 20×20 gridworld environment
│   ├── qrl.py                    # QRL agent (classical simulation)
│   ├── td_agent.py               # TD(0) / Q-learning agent
│   ├── experiment.py             # Main experiment — reproduces Figs 5, 6, 7
│   ├── qrl_traces.py             # Innovation: QRL + eligibility traces
│   ├── experiment_innovation.py  # Innovation experiments
│   ├── check_results.py          # Quick validation script
│   ├── test_env.py               # Environment path verification
│   ├── test_quick.py             # Quick agent sanity checks
│   └── find_walls.py             # Wall configuration search utility
├── results/
│   ├── fig5_qrl_vs_td.png        # Figure 5 reproduction
│   ├── fig6_qrl_alphas.png       # Figure 6 reproduction
│   └── fig7_td_alphas.png        # Figure 7 reproduction
└── 2_Quantum_Reinforcement_Learning.pdf  # Original paper
```

## How to Run

**Requirements**: Python 3, NumPy, Matplotlib

```bash
python src/experiment.py
```

Results are saved to the `results/` directory. The full run takes approximately 30–60 minutes depending on hardware.

## Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Grid size | 20×20 | Sec V-A |
| Start / Goal | (1,1) / (18,18) | Sec V-A |
| Reward (goal / step) | +100 / −1 | Sec V-A |
| γ (discount) | 0.99 | Sec V-A |
| ε (TD exploration) | 0.01 | Sec V-A |
| α (TD, Fig 5) | 0.01 | Sec V-B |
| α (QRL, Fig 5) | 0.06 | Sec V-B |
| k (amplitude rotation) | 0.02 | Experiential (Sec III-D) |
| Episodes per run | 10000 | Sec V |
| Optimal path | 36 steps | Sec V-B |

## Innovation Extension

Beyond the paper reproduction, `src/qrl_traces.py` and `src/experiment_innovation.py` implement an extension: **QRL with eligibility traces (TD(λ))**. This replaces the TD(0) value update with TD(λ), propagating reward information backward through recently visited states within a single episode. Run with:

```bash
python src/experiment_innovation.py
```
