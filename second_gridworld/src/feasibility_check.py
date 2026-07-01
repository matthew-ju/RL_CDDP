"""
Feasibility check: can a greedy max-thrust CBF-safe controller solve 420-step
tight-clock episodes? If not, RL cannot be expected to do better.
"""
import sys
import math
import numpy as np
import torch

sys.path.insert(0, '.')
from src.analytical_sandbox import DifferentiablePointMassEnv

DEVICE = 'cpu'
N_TRIALS = 200
MAX_STEPS_TIGHT = 420
MAX_STEPS_LOOSE = 2500
GOAL_TOL = 0.4
DT = 0.02
V_MAX = 4.0
U_MAX = 2.0
GAMMA_CBF = 0.7

OBS_C = torch.tensor([
    [5.0, 5.0], [1.5, 5.0], [8.0, 2.0], [6.5, 8.5], [4.0, 2.0],
], dtype=torch.float32)
OBS_R = torch.tensor([
    math.sqrt(0.5), math.sqrt(0.2), math.sqrt(0.2), math.sqrt(0.2), math.sqrt(0.2),
], dtype=torch.float32)


def make_env(goal):
    return DifferentiablePointMassEnv(
        goal=goal,
        obstacle_c=OBS_C,
        obstacle_r=OBS_R,
        dt=DT,
        goal_tol=GOAL_TOL,
        v_max=V_MAX,
        obstacle_amplitude=0.0,
        obstacle_omega=None,
        device=DEVICE,
    )


def run_greedy_trial(max_steps, seed):
    rng = torch.Generator()
    rng.manual_seed(seed)

    # Sample start from [0,1]^2, goal from [8,10]^2 (training distribution)
    start_pos = torch.rand(2, generator=rng) * 1.0
    goal_pos = 8.0 + torch.rand(2, generator=rng) * 2.0

    env = make_env(goal_pos)
    init_state = torch.cat([start_pos, torch.zeros(2), torch.ones(1)]).unsqueeze(0)
    env.reset(init_state, max_steps=max_steps)

    min_dist = float('inf')
    for step in range(max_steps):
        state = env.state  # (1, 5)
        pos = state[0, :2]
        dist = (pos - goal_pos).norm().item()
        min_dist = min(min_dist, dist)
        if dist < GOAL_TOL:
            return True, step, min_dist

        direction = goal_pos - pos
        u_nom = (direction / direction.norm()).unsqueeze(0) * U_MAX
        stakes = torch.zeros(1)
        u_safe = env.project_action_cbf(state, u_nom, stakes, gamma=GAMMA_CBF)
        env.step(u_safe)

    return False, max_steps, min_dist


tight_successes = 0
loose_successes = 0
tight_steps = []
loose_steps = []
tight_min_dists = []
loose_min_dists = []

for seed in range(N_TRIALS):
    r, s, d = run_greedy_trial(MAX_STEPS_TIGHT, seed)
    if r:
        tight_successes += 1
        tight_steps.append(s)
    else:
        tight_min_dists.append(d)

    r, s, d = run_greedy_trial(MAX_STEPS_LOOSE, seed)
    if r:
        loose_successes += 1
        loose_steps.append(s)
    else:
        loose_min_dists.append(d)

print("=== FEASIBILITY REPORT ===")
print(f"Tight clock ({MAX_STEPS_TIGHT} steps): {tight_successes}/{N_TRIALS} = {100*tight_successes/N_TRIALS:.1f}%")
if tight_steps:
    print(f"  Avg steps when success: {np.mean(tight_steps):.1f}, min: {np.min(tight_steps)}")
if tight_min_dists:
    print(f"  Avg min_dist when fail:  {np.mean(tight_min_dists):.2f}, max: {np.max(tight_min_dists):.2f}")
print(f"Loose clock ({MAX_STEPS_LOOSE} steps): {loose_successes}/{N_TRIALS} = {100*loose_successes/N_TRIALS:.1f}%")
if loose_steps:
    print(f"  Avg steps when success: {np.mean(loose_steps):.1f}, min: {np.min(loose_steps):.1f}")
if loose_min_dists:
    print(f"  Avg min_dist when fail:  {np.mean(loose_min_dists):.2f}")

print()
tight_frac = tight_successes / N_TRIALS
print("VERDICT:")
if tight_frac < 0.30:
    print(f"  INFEASIBLE ({tight_frac*100:.1f}% < 30% threshold)")
    print("  A greedy max-thrust CBF-safe controller cannot reliably solve 420-step episodes.")
    print("  RL policy cannot be expected to do better.")
    print("  ACTION: Remove tight-clock. Train on long-clock only (Phase 1A).")
else:
    print(f"  FEASIBLE ({tight_frac*100:.1f}% >= 30% threshold)")
    print("  Greedy controller solves tight-clock. RL has room to improve.")
    print("  ACTION: Fix curriculum and clock conditioning (Phase 1B).")
