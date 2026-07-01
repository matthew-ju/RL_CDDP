# OmniSafe Integration: Adapter Technical Details

This document explains the technical implementation of the `HostageGridworldAdapter` and how it bridges our custom Discrete Gridworld with the Continuous-native OmniSafe framework.

## 🌉 The HostageGridworldAdapter

OmniSafe expects environments to follow the Gymnasium Robotics API (continuous actions, specific observation dicts, etc.). Our `HostageGridworld` is a simple discrete environment.

### 1. Action Space Bridge

The core challenge was converting the discrete action space (Up, Down, Left, Right) to a continuous space for PPO/PPO-Lag.

- **Original Env**: `Discrete(4)`
- **OmniSafe Expectation**: `Box(shape=(k,), low=-1, high=1)`
- **Solution**: We use a **Quadrant Mapping**. The agent outputs a `2D` vector `[x, y]`. We interpret the **sign** of each component to select a discrete action.

```python
# In hostage_gridworld_adapter.py
def step(self, action):
    # Retrieve raw continuous action from agent (Box 2D)
    val = action 
    
    # Map Quadrants to Discrete Action
    if val[0] >= 0 and val[1] >= 0:
        idx = 0 # Up
    elif val[0] >= 0 and val[1] < 0:
        idx = 1 # Down
    elif val[0] < 0 and val[1] >= 0:
        idx = 2 # Left
    else:
        idx = 3 # Right (-, -)
        
    return self._env.step(idx)
```
**Why?** This prevents "Variance Explosion". Even if the policy variance is high (e.g., outputs `[100, -500]`), the quadrant mapping remains stable, ensuring all 4 actions are reachable.

### 2. Observation Space Bridge

Our environment returns a flat vector of shape `(9,)` representing the agent's position, hostage position, and civilian positions.

- **Original Env**: `Box(low=0, high=5, shape=(9,), dtype=float32)`
- **OmniSafe**: Handles this natively, but critically, we **MUST DISABLE** normalization.
- **Why?** OmniSafe's default normalization (running mean/std) distorted the grid coordinates (0, 1, 2, 3, 4) into arbitrary floats, causing the agent to fail during evaluation where raw coordinates are used. `obs_normalize=False` fixes this.

### 3. Cost Signal

OmniSafe requires a specific `cost` signal in the `info` dictionary for constrained optimization.

- **Original Env**: Returns `reward`.
- **Adapter**: injects `cost` into `info`.

```python
# In hostage_gridworld_adapter.py
def step(self, idx):
    obs, reward, terminated, truncated, info = self._env.step(idx)
    # Ensure cost is present for OmniSafe
    if 'cost' not in info:
        info['cost'] = 0.0 # Default if not provided
    return obs, reward, terminated, truncated, info
```

## 🛠 File Structure

- **`hostage_gridworld.py`**: The "Physics Engine". Handles grid logic, collisions, rewards.
- **`hostage_gridworld_adapter.py`**: The "Translator". Converts actions/observations for OmniSafe.
- **`train_omnisafe_constrained.py`**: The "Trainer". Configures the PPO-Lag algorithm with correct hyperparameters (`cost_limit`, `lagrangian_multiplier`).

## ⚠️ Known Issues & Fixes

1.  **Evaluator Crash**: The evaluation script `evaluate_baselines_omnisafe.py` crashed initially because it tried to send discrete actions.
    - **Fix**: Updated evaluation script to use the exact same Quadrant Mapping logic as the training adapter.

2.  **Stationary Agent**: The Constraints were too soft (penalty-based).
    - **Fix**: Switched to `hard_constraint=True` (Civilians = Walls) during training.
