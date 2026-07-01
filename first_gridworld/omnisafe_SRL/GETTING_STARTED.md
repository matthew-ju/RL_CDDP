# Getting Started with Hostage Gridworld (OmniSafe Edition)

This guide will walk you through setting up the environment, installing dependencies (including critical patches for Python 3.12 compatibility), and running your first experiment.

## 1. Prerequisites

- **Python 3.10+**: Developed and tested on Python 3.12.
- **Mujoco**: Required for OmniSafe physics simulation (even though our Gridworld is 2D, OmniSafe depends on it).

## 2. Installation

We use a local, patched version of `safety-gymnasium` to fix compatibility issues with Python 3.12 (specifically `dataclasses.field` mutable defaults).

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/CDDP_RL.git
cd CDDP_RL/omnisafe_SRL
```

### Step 2: Install Base Dependencies
```bash
# Install Gymnasium and Mujoco bindings manually to ensure version compatibility
pip install gymnasium==0.29.1
pip install "gymnasium[mujoco]"
pip install "mujoco>=3.0.0"
```

### Step 3: Install Patched Safety Gymnasium (Local)
This repository includes a patched version of `safety-gymnasium` in the `safety-gymnasium/` directory. Install it in editable mode:
```bash
cd safety-gymnasium
pip install -e .
cd ..
```

### Step 4: Install OmniSafe
Now install OmniSafe itself. Note that we force it to respect the installed `safety-gymnasium`.
```bash
pip install omnisafe
```

### Step 5: Verify Installation
Run the integration test script to ensure everything is working:
```bash
python3 test_omnisafe_integration.py
```
If you see "SUCCESS: Environment check passed!", you are ready to go.

## 3. Running Your First Experiment

The project includes an automated script `run_full_experiment.py` that handles:
1.  **Training PPO** (Unconstrained agent)
2.  **Training PPO-Lag** (Constrained agent)
3.  **Evaluating** both agents
4.  **Generating Plots**

To run the standard "Grid 1" (Simple Detour) experiment:

```bash
python3 run_full_experiment.py --grid 1 --steps 500000
```

### What Happens Next?
- **Training**: You will see training logs for PPO, followed by PPO-Lag. This takes ~15-20 minutes on a modern CPU.
- **Evaluation**: The script will load the best models and run them in Low Stakes (50 steps) and High Stakes (5 steps) scenarios.
- **Results**: Check the generated `.png` files for visualizations.

## 4. Troubleshooting

### "AttributeError: module 'safety_gymnasium' has no attribute 'Builder'"
This usually means you are using the PyPI version of `safety-gymnasium`, not our local patched version. Uninstall and reinstall from local:
```bash
pip uninstall safety-gymnasium
cd safety-gymnasium && pip install -e . && cd ..
```

### "ValueError: mutable default <class> for field"
This means you are running Python 3.11+ without the patch. Ensure you installed the local `safety-gymnasium` which contains the `@dataclass` fixes.

### "AttributeError: 'HostageGridworld' object has no attribute 'model'"
This occurs if OmniSafe tries to access Mujoco-specific attributes. We use a custom `HostageGridworldAdapter` to bridge this gap. Ensure your training script uses the adapter correctly.
