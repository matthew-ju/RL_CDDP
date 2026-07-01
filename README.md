# CDDP Research Project

This repository contains the ongoing research for the **Context-Dependent Deontological PPO (CDDP)** algorithm, aiming to improve safety-constrained RL in complex environments. 

## 📂 Project Structure

- **[`docs/`](docs/)**: Skeleton backbone framework and theoretical formalizations for CDDP.
- **[`first_gridworld/`](first_gridworld/)**: Initial discrete gridworld experiments containing baseline analysis and OmniSafe integrations. (Good preliminary results)
- **[`second_gridworld/`](second_gridworld/)**: Continuous gridworld experiments (currently facing challenges/bad results). **Collaborators should start here.**
- **[`mujoco/`](mujoco/)**: 3D physics simulator integration for CDDP. (Crucial for scalability and evaluation in top-tier conferences like NeurIPS, ICML, ICLR).
- **[`results/`](results/)**: High-level aggregated results, key findings, and comparison charts from earlier stages.

## 🚀 Getting Started

To set up the repository for development, it is recommended to create a new virtual environment:

```bash
# Clone the repo
git clone https://github.com/your-username/CDDP_RL.git
cd CDDP_RL

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

### Continuing the work

If you are collaborating on the current issues, please navigate to the `second_gridworld/src` directory. You can run the sandbox script to see the continuous CDDP behavior:

```bash
cd second_gridworld
python src/train_sandbox_cddp.py --episodes 4 --pretrain-episodes 2 --warmup-steps 0
```
