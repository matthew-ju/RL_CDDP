# Hostage Gridworld: CDDP Baseline Analysis (OmniSafe Edition)

This repository contains the **Baseline Analysis** for the Context-Dependent Deontological PPO (CDDP) research project. It demonstrates the fundamental limitations of standard Reinforcement Learning approaches in ethical dilemmas.

## 🎯 The Ethical Dilemma

We use a custom **Hostage Gridworld** environment where an agent must save a hostage. The agent faces a choice between a "Safe Path" (Ethical) and a "Fast Path" (Unethical/Dangerous).

### Two Scenarios
1.  **Low Stakes (Timer = 50s)**: Plenty of time. The agent *should* take the long, safe path around civilians.
2.  **High Stakes (Timer = 5s)**: A bomb is ticking. The agent *must* take the fast path (trampling civilians) to save the hostage, or everyone dies.

## 📊 The "Gap" in Current RL

We compared two state-of-the-art baselines using the **OmniSafe** framework:

| Scenario | PPO (Utilitarian) | Constrained PPO-Lag (Deontological) | Analysis |
| :--- | :--- | :--- | :--- |
| **Low Stakes** | **Saved** (with Violations) | **Saved** (0 Violations) | PPO is **Reckless** (tramples unnecessarily). Constrained is **Safe**. |
| **High Stakes** | **Saved** (with Violations) | **FAILED** (Bomb Exploded) | Constrained is **Rigid** (refuses to violate constraints to prevent catastrophe). PPO is **Flexible** (but reckless). |

**Conclusion:** Neither agent is ideal. We need an agent that is **Safe in Low Stakes** but **Flexible in High Stakes**. This motivates the **CDDP** algorithm.

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+ installed. See [GETTING_STARTED.md](GETTING_STARTED.md) for detailed installation instructions, including necessary patches for `safety-gymnasium`.

### 2. Run the Full Experiment
The `run_full_experiment.py` script automates the entire pipeline:
1.  **Trains PPO** (Unconstrained) - 500k steps.
2.  **Trains PPO-Lag** (Constrained) - 500k steps.
3.  **Evaluates** both agents on Low and High Stakes scenarios.
4.  **Generates** plots and a summary table.

```bash
cd omnisafe_SRL
python3 run_full_experiment.py --grid 1 --steps 500000
```

### 3. View Results
After the script finishes (~15-20 mins), check the output files:
- `summary_comparison_grid1.png`: Visual comparison of paths and metrics.
- `viz_*.png`: Trajectory visualizations for each agent/scenario.
- Terminal Output: Detailed metrics table.

## 📂 Repository Structure

- `hostage_gridworld.py`: Custom Gymnasium environment (The "World").
- `hostage_gridworld_adapter.py`: Bridge between our Env and OmniSafe (The "Interface").
- `run_full_experiment.py`: Main automation script.
- `train_omnisafe_ppo.py`: Training script for Utilitarian agent.
- `train_omnisafe_constrained.py`: Training script for Deontological agent.
- `evaluate_baselines_omnisafe.py`: Evaluation and visualization logic.

## 📚 Documentation
- [GETTING_STARTED.md](GETTING_STARTED.md): Installation and Setup.
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md): Detailed explanation of training parameters and decisions.
- [OMNISAFE_INTEGRATION.md](OMNISAFE_INTEGRATION.md): Technical details on bridging Custom Envs with OmniSafe.
