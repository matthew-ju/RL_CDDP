```
# CDDP Fix: Train/eval clock encoding mismatch + Z_deon absolute time blindness
#
# ROOT CAUSE: state[4] (clock feature) is encoded differently in training vs eval.
#   Training: state[4] = remaining_steps / max_episode_steps → normalized [0,1] per episode
#   Eval:     state[4] = remaining_steps / 1000.0 → absolute time / fixed constant
# This causes the actor and Z_deon to receive completely OOD inputs during eval
# (e.g. state[4]=2.5 for loose clocks, vs training range [0,1]).
#
# Additionally, the per-episode normalization means Z_deon cannot distinguish
# tight-clock from loose-clock episodes (both start at state[4]=1.0), making
# the stakes/adaptive mechanism fundamentally unable to differentiate.
#
# FIX: Use state[4] = remaining_steps / LONG_CLOCK_STEPS (=2500) everywhere.
# Tight clock (420) starts at 0.168; loose clock (2500) starts at 1.0.
# Z_deon gets absolute time information; eval matches training.
#
# CONSTRAINTS: Do NOT touch GAMMA_CBF, weighted-BCE Z_deon loss, or terminal rewards.

# Step 1: Read the files we need to modify
Read src/train_sandbox_cddp.py
Read src/helper_plotter.py

# Step 2: Fix training clock encoding in train_sandbox_cddp.py

# 2a. Fix episode launch: change /1000.0 to /LONG_CLOCK_STEPS and remove the
#     override that sets state[4]=1.0. Lines 1528-1531.
#
#     BEFORE:
#         starts[:, 4] = clocks.to(device=DEVICE, dtype=starts.dtype) / 1000.0
#
#         # time_remaining_normalized = 1.0 at episode start (full budget)
#         starts[:, 4] = 1.0
#
#     AFTER:
#         starts[:, 4] = clocks.to(device=DEVICE, dtype=starts.dtype) / float(LONG_CLOCK_STEPS)

Edit src/train_sandbox_cddp.py:
  old: |
        starts[:, 4] = clocks.to(device=DEVICE, dtype=starts.dtype) / 1000.0

        # time_remaining_normalized = 1.0 at episode start (full budget)
        starts[:, 4] = 1.0
  new: |
        starts[:, 4] = clocks.to(device=DEVICE, dtype=starts.dtype) / float(LONG_CLOCK_STEPS)

# 2b. Fix per-step clock update. Lines 1746-1750.
#
#     BEFORE (normalizes by per-episode max_episode_steps):
#         # Normalize clock to [0, 1]: 1.0=full budget, 0.0=expired
#         next_state[active, 4] = (
#             active_steps_remaining.to(dtype=next_state.dtype)
#             / _ems_active.to(dtype=next_state.dtype).clamp(min=1.0)
#         )
#
#     AFTER (normalizes by fixed LONG_CLOCK_STEPS):
#         next_state[active, 4] = (
#             active_steps_remaining.to(dtype=next_state.dtype)
#             / float(LONG_CLOCK_STEPS)
#         )

Edit src/train_sandbox_cddp.py:
  old: |
        # Normalize clock to [0, 1]: 1.0=full budget, 0.0=expired
        next_state[active, 4] = (
            active_steps_remaining.to(dtype=next_state.dtype)
            / _ems_active.to(dtype=next_state.dtype).clamp(min=1.0)
        )
  new: |
        next_state[active, 4] = (
            active_steps_remaining.to(dtype=next_state.dtype)
            / float(LONG_CLOCK_STEPS)
        )

# Step 3: Fix eval clock encoding in helper_plotter.py

# 3a. Fix initial state clock. Line 366.
#     BEFORE: start_batch[:, 4] = float(clock_max_steps) / 1000.0
#     AFTER:  start_batch[:, 4] = float(clock_max_steps) / float(cddp.LONG_CLOCK_STEPS)

Edit src/helper_plotter.py:
  old: "start_batch[:, 4] = float(clock_max_steps) / 1000.0"
  new: "start_batch[:, 4] = float(clock_max_steps) / float(cddp.LONG_CLOCK_STEPS)"

# 3b. Fix per-step clock update. Line 405.
#     BEFORE: next_state[:, 4] = float(remaining_steps) / 1000.0
#     AFTER:  next_state[:, 4] = float(remaining_steps) / float(cddp.LONG_CLOCK_STEPS)

Edit src/helper_plotter.py:
  old: "next_state[:, 4] = float(remaining_steps) / 1000.0"
  new: "next_state[:, 4] = float(remaining_steps) / float(cddp.LONG_CLOCK_STEPS)"

# Step 4: Launch training
bash run_v2.sh adaptive_v2 --timer-curriculum compression --seed-episodes 3000 --pretrain-episodes 10000 --episodes 60000

# Monitor: Z_deon frac>0.8 should climb above 0.30 by ep 40K (tight-clock states get confident high predictions). AvgS_odd should track clock budget: higher on tight episodes, lower on loose.
# Early warning nick: If BestUtilSucc drops below 60% by ep 25K, the actor may be confused by the new clock range. If AvgS_odd saturates at 1.0 before ep 30K, Z_deon is over-predicting failure everywhere.
```