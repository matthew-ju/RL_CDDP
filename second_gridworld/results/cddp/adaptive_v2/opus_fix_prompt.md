```
# CDDP Fix: Directed goal-seeking exploration at high stakes

# The actor has learned implicit obstacle aversion from Q-function training on
# CBF-projected data. Even with CBF fully relaxed (stakes≈1), the actor avoids
# obstacles because the Q-function has never seen near-obstacle transitions succeed.
# Fix: blend the actor's nominal action with a goal-directed vector during
# high-stakes utilitarian episodes, generating the near-obstacle training data
# the Q-function needs.

# Step 1: Read the training script to verify current state
Read src/train_sandbox_cddp.py

# Step 2: Add hyperparameter constants near the other CDDP hyperparameters (after P_REF line)
# Find the P_REF = 0.35 / ALLOW_HARD_CBF_BYPASS block and add two new constants after it:

Edit src/train_sandbox_cddp.py
OLD:
ALLOW_HARD_CBF_BYPASS = False
NEW:
ALLOW_HARD_CBF_BYPASS = False

# Goal-seeking exploration: when stakes are high on utilitarian episodes,
# blend the nominal action toward the goal to generate near-obstacle data
# that the Q-function has never seen. This lets the actor discover that
# shortcut paths succeed when the CBF is relaxed.
GOAL_SEEK_BLEND = 0.3
GOAL_SEEK_STAKES_THRESHOLD = 0.7

# Step 3: Add the goal-seeking blend in the training loop, between the OU noise
# application and the u_nom zeroing. This goes inside the `with torch.no_grad():`
# block, after the OU noise else-branch and before `u_nom[~active] = 0.0`.

Edit src/train_sandbox_cddp.py
OLD:
                    u_nom[noise_mask] = torch.clamp(u_nom[noise_mask] + noise[noise_mask], -U_MAX, U_MAX)
            u_nom[~active] = 0.0
NEW:
                    u_nom[noise_mask] = torch.clamp(u_nom[noise_mask] + noise[noise_mask], -U_MAX, U_MAX)

            if announced_cddp_phase and GOAL_SEEK_BLEND > 0:
                high_stakes_util = util_mask & (stakes > GOAL_SEEK_STAKES_THRESHOLD)
                if high_stakes_util.any():
                    hs_pos = s_before[high_stakes_util, :2]
                    hs_goals = current_goals[high_stakes_util]
                    goal_dir = hs_goals - hs_pos
                    goal_dir = F.normalize(goal_dir, dim=-1) * U_MAX
                    u_nom[high_stakes_util] = (
                        (1.0 - GOAL_SEEK_BLEND) * u_nom[high_stakes_util]
                        + GOAL_SEEK_BLEND * goal_dir
                    )

            u_nom[~active] = 0.0

# Step 4: Delete old checkpoints so training starts fresh (run_name is adaptive_v2)
Bash: rm -rf results/cddp/adaptive_v2/checkpoints

# Step 5: Launch training
Bash: bash run_v2.sh adaptive_v2 --timer-curriculum compression --seed-episodes 3000 --pretrain-episodes 10000 --episodes 60000

# Monitor: ExcessViol_odd should become non-zero within first 5K CDDP episodes;
#   tight-clock util success should climb toward 40%+ by ep 30K;
#   Z_deon mean_P_cat should stay in [0.3, 0.7]
# Early warning nick: if overall Succ drops below 50% for >2K episodes, reduce
#   GOAL_SEEK_BLEND to 0.2; if ExcessViol_odd exceeds 30 on loose-clock episodes,
#   the Q-function is over-valuing near-obstacle paths globally
```