# QRL Project — Dong et al. 2008

Replication and extension of:
> Dong et al., "Quantum Reinforcement Learning", IEEE Trans. SMC-B, 2008

## Files

| File | Description |
|------|-------------|
| `src/gridworld.py` | 20×20 gridworld from Fig. 4 (pixel-sampled; BFS optimal = 36 steps) |
| `src/qrl.py` | QRL agent with corrected Grover rotation (Eq. 40) |
| `src/qrl_traces.py` | Innovation: QRL + TD(λ) eligibility traces |
| `src/td_agent.py` | TD(0) Q-learning baseline with ε-greedy |
| `src/tests.py` | 23-test suite — run first to verify everything |
| `src/experiment.py` | Reproduces Figs. 5, 6, 7 from the paper |
| `src/experiment_innovation.py` | Innovation comparison and λ-sweep plots |

## Quick start

```bash
cd src

# 1. Run tests (should show 23/23 passed)
python tests.py

# 2. Reproduce paper figures (takes ~30 min)
python experiment.py

# 3. Run innovation experiment
python experiment_innovation.py
```

Results are saved to `results/`.

## Key fixes from original implementation

### 1. Grover rotation direction (qrl.py)
The original used the wrong sign, rotating *away* from the target action:
```python
# WRONG — rotates backward
new_c_a = cos(2Lθ) * c_a - sin(2Lθ) * c_⊥

# CORRECT — increases φ, reinforcing the action (Eq. 40)
new_c_a = cos(2Lθ) * c_a + sin(2Lθ) * c_⊥
```
For N_a=4, L=1 from uniform: wrong gives c_a=−0.5, correct gives c_a=1.0.

### 2. Default k parameter (qrl.py, qrl_traces.py)
```python
# WRONG — k=1.0 gives L=100 at goal, clamped to L_max by coincidence
k=1.0

# CORRECT — k=0.01 gives L=int(0.01×100)=1=L_max at goal reward
k=0.01
```

### 3. Gridworld walls (gridworld.py)
Original walls were algorithmically generated. Replaced with walls
pixel-sampled directly from the paper's Fig. 4 at 300 DPI.
BFS shortest path confirmed = 36 steps in both cases.

### 4. Results path (experiment.py)
Experiments now resolve `results/` relative to the script file,
so they work regardless of which directory you run from.
