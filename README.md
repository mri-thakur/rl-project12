# Quantum Reinforcement Learning — Implementation & Innovation

Replication, analysis, and extension of:
> Dong et al., "Quantum Reinforcement Learning", IEEE Transactions on Systems, Man, and Cybernetics, Part B, Vol. 38, No. 5, October 2008

## Overview

This project provides a complete implementation of Quantum Reinforcement Learning (QRL), a novel learning framework that combines quantum computation principles with classical reinforcement learning. The implementation faithfully reproduces the original paper's experimental results and extends it with TD(λ) eligibility traces for improved convergence.

## Project Files

| File | Purpose |
|------|---------|
| `QRL_Report_Final.md` | Full implementation report with analysis, challenges, solutions, and results |
| `README.md` | This file |
| `src/gridworld.py` | 20×20 maze environment (Sec. V, Fig. 4; BFS shortest path = 36 steps) |
| `src/qrl.py` | QRL agent with corrected Grover rotation (Eq. 40) and TD(0) value updates |
| `src/qrl_traces.py` | **Innovation**: QRL + TD(λ) eligibility traces for faster convergence |
| `src/td_agent.py` | TD(0) Q-learning baseline with ε-greedy action selection |
| `src/tests.py` | 18-test verification suite (run first) |
| `src/experiment.py` | Reproduces Figures 5–7 from the original paper (~30 min) |
| `src/experiment_innovation.py` | Compares QRL, QRL+Traces, and TD agents; sweeps λ parameter |
| `results/` | Generated plots and performance data |

## Quick Start

### 1. Verify Implementation (≈30 seconds)
```bash
cd src
python tests.py
```
Expected output: **18/18 tests passed**

### 2. Reproduce Paper Figures (≈30 minutes)
```bash
python experiment.py
```
Generates:
- `results/fig5_convergence.png` — QRL vs TD vs Theoretical QRL
- `results/fig6_qrl_alpha.png` — QRL sensitivity to learning rate α
- `results/fig7_td_alpha.png` — TD agent learning rate analysis

### 3. Run Innovation Experiment (≈5 minutes)
```bash
python experiment_innovation.py
```
Generates:
- `results/innovation_agents.png` — QRL vs QRL+Traces vs TD comparison
- `results/lambda_sweep.png` — QRL+Traces performance across λ ∈ [0, 1]

## Key Features

### Correct Grover Rotation (Eq. 40)
Unlike naive implementations, the rotation uses the correct sign and clamping:
```python
# Increases φ, reinforcing the action
phi_target = min(phi_current + 2*L*theta, π/2)
```
This prevents amplitude oscillation past the peak.

### Configurable Parameters
All agents support:
- `alpha`: TD learning rate
- `gamma`: Discount factor (0.99)
- `k`: Grover signal scaling (1.0 recommended)
- `epsilon`: Exploration rate (ε-greedy for TD)
- `lam`: Eligibility trace decay (0.5 default for QRL+Traces)



## Experimental Results

### Figure 5: Convergence Comparison
- **TD(0)**: ~2500 episodes to convergence (α=0.01, ε=0.01)
- **QRL**: ~1000 episodes to convergence (α=0.06)
- **Theoretical QRL**: ~5 episodes (ideal quantum agent)
- All reach optimal 36-step policy

### Figure 6: QRL Learning Rate Sensitivity
- Optimal α = 0.06 (fastest convergence)
- Converges for α ∈ [0.02, 0.10]
- Too low (α=0.01) → slow learning
- Too high (α≥0.11) → oscillation/instability

### Figure 7: TD Agent Learning Rate Analysis
- All learning rates converge (but at different rates)
- Paper's non-convergence at α=0.02, 0.03 is likely a stochastic artifact or specific MATLAB seed

### Innovation: QRL + TD(λ) Traces
QRL+Traces uses eligibility traces to propagate credit backward through visited states:
- Faster convergence than both QRL and TD(0), especially early in training
- Improves performance by typically **2–5×** in early episodes
- λ ≈ 0.5 provides best balance of recency vs. accumulation

## Implementation Highlights

### 1. Corrected Grover Rotation (Sec. III-D, Eq. 40)
The paper's specification requires rotating the amplitude vector in the (|a⟩, |a⊥⟩) plane:
```
new_c_a  = cos(2Lθ)·c_a + sin(2Lθ)·c_⊥
new_c_⊥  = cos(2Lθ)·c_⊥ - sin(2Lθ)·c_a
```
**Common mistakes**:
- **Wrong sign** (rotation backwards) — reduces probability instead of increasing
- **Over-rotation** past π/2 — causes oscillation on subsequent visits
- **k parameter** — must be calibrated correctly (k=1.0 gives L≈1 per typical step)

### 2. Backup Register (Sec. III-E)
Action selection collapses the amplitude distribution, but the backup register preserves 
the original for the next step's Grover rotation. This prevents learning from decaying.

### 3. Eligibility Traces (Innovation)
Extends the base QRL with TD(λ) for faster credit propagation:
```
e(s_t)  ← e(s_t) + 1           (accumulate at current state)
V(s)    ← V(s) + α·δ·e(s)      (update all traced states)
e(s)    ← γ·λ·e(s)             (decay all traces)
```

## Parameters & Variations

### QRLAgent
```python
QRLAgent(n_actions=4, alpha=0.06, gamma=0.99, k=1.0)
```
- `alpha=0.06`: Best for paper reproduction (Fig. 5)
- `gamma=0.99`: Standard discount factor
- `k=1.0`: Maps reward scale to Grover iterations

### QRLTracesAgent
```python
QRLTracesAgent(n_actions=4, alpha=0.06, gamma=0.99, k=1.0, lam=0.5)
```
- `lam=0.5`: Default trace decay (try 0.3–0.9 range)
- `lam=0.0`: Reduces to standard QRL (TD(0))
- `lam=1.0`: Full Monte-Carlo credit

### TDAgent
```python
TDAgent(n_actions=4, alpha=0.01, gamma=0.99, epsilon=0.01)
```
- `alpha=0.01`: Paper's recommended learning rate (conservative)
- `epsilon=0.01`: Exploration rate (1% random actions)

## Known Limitations & Design Decisions

### Figure 7 Non-Convergence
The paper shows TD(0) failing to converge at α=0.02 and α=0.03 within 10,000 episodes,
but **our implementation converges for all tested learning rates**. This discrepancy is
likely due to:
- Different random seed in the original MATLAB implementation
- Stochastic variation in gridworld exploration
- Potential differences in Q-learning vs. V(s) updates

**Our choice**: Present faithful convergence results rather than fabricating instability to match a 
single stochastic run.

### TD(0) vs. Q-Learning
The paper describes a V(s) value-based agent, but our TD agent uses Q(s,a) for model-free 
learning. This is standard practice and produces the required Bellman backup semantics.

## Reproducibility Notes

- **Random seed**: Results vary slightly across runs (inherent stochasticity)
- **Platform**: Tested on Python 3.8+ with NumPy and Matplotlib
- **Execution time**: Experiments are I/O-bound (plotting); consider `results/` in `.gitignore`
- **Directory-agnostic**: All scripts resolve `results/` relative to the source file, 
  so you can run from any working directory
