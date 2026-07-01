# Training Guide: Mastering Baseline Agents

This guide explains the specific configuration choices required to successfully train baseline agents in the Hostage Gridworld environment using OmniSafe.

## 🏗 Architecture Overview

We use two distinct algorithms to establish the behavioral extremes:
1.  **PPO (Proximal Policy Optimization)**: Represents the **Utilitarian** agent. It optimizes for total reward, ignoring constraints unless heavily penalized.
2.  **PPO-Lag (PPO with Lagrangian Relaxation)**: Represents the **Deontological** agent. It optimizes for reward *subject to* strict safety constraints (Civilian Violations = 0).

## 🔑 Critical Configuration Decisions

### 1. Hard Constraints (`hard_constraint=True`)
For the Constrained Agent, we enforce `hard_constraint=True` in the environment.
- **Why?** Initially, we treated civilians as penalties (Soft Constraint). The agent learned to "eat the cost" for higher reward.
- **Fix:** We made civilians act like physical walls during training. This forces the agent to learn the detour path physically, ingraining the "Deontological" behavior.

### 2. Zero Entropy (`entropy_coef=0.0`)
We set `entropy_coef: 0.0` for the Constrained Agent in `train_omnisafe_constrained.py`.
- **Why?** The combination of high penalties and a constrained optimization objective caused the policy variance (`PolicyStd`) to explode to ~1900.0. The agent became purely random to maximize entropy.
- **Fix:** Disabling entropy regularization forced the agent to exploit the reward signal (distance to goal) immediately.

### 3. Quadrant Discretization (The "Variance Hack")
The environment is discrete (Up, Down, Left, Right), but OmniSafe assumes continuous control (Mujoco).
- **Problem:** With a standard 1D continuous action, high variance caused the agent to get stuck outputting extreme values (always Action 0 or Action 3).
- **Solution:** We use a 2D continuous action space `Box(shape=(2,), low=-1, high=1)`.
  - We map the **signs** of the two outputs to the 4 quadrants:
    - `(+, +)` → **Up** (0)
    - `(+, -)` → **Down** (1)
    - `(-, +)` → **Left** (2)
    - `(-, -)` → **Right** (3)
  - **Benefit:** This is robust to variance explosion. Even if the policy outputs `[1000, -1000]`, the sign is preserved, ensuring valid action selection.

### 4. Raw Observations (`obs_normalize=False`)
We disabled observation normalization in both training scripts.
- **Why?** OmniSafe normalizes observations by default. However, our evaluation environment provided raw coordinates (0-4). This mismatch caused the trained agent to fail during evaluation.
- **Fix:** `obs_normalize=False` ensures training and evaluation see the exact same state space.

## 📜 Scripts Explained

### `run_full_experiment.py`
The master script. It:
1.  Calls `train_omnisafe_ppo.py`.
2.  Calls `train_omnisafe_constrained.py`.
3.  Loads the best models.
4.  Runs `evaluate_baselines_omnisafe.py`.

### `train_omnisafe_ppo.py`
Trains the Utilitarian baseline.
- **Algo**: PPO
- **Env**: `HostageGridworld-v0` (Low Stakes, No Constraints)
- **Steps**: 500,000

### `train_omnisafe_constrained.py`
Trains the Deontological baseline.
- **Algo**: PPO-Lag
- **Env**: `HostageGridworld-Constrained-v0` (Hard Constraints)
- **Parameters**: `cost_limit=0.0`, `lagrangian_multiplier_init=0.001`, `lambda_lr=0.01`
- **Steps**: 500,000

## 🔧 Troubleshooting Common Failures

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **Agent Spins/Stationary** | "Variance Explosion" | Check `entropy_coef=0.0` and Quadrant Discretization. |
| **Agent Fails Eval** | Obs Mismatch | Ensure `obs_normalize=False`. |
| **Agent Ignores Walls** | Weak Penalties | Check `WALL_COLLISION_PENALTY` in `hostage_gridworld.py`. |
