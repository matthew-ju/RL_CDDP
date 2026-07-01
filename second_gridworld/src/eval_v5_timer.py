"""
eval_v5_timer.py
================
Evaluate the best adaptive_v5 checkpoint across tight and loose clock budgets.
Reports success rate and violations per timer bucket.

Usage:
    python3 src/eval_v5_timer.py --run-name adaptive_v5 --n-eval 200
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/cddp_rl_matplotlib")

import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

import src.train_adaptive_v5 as v5

DEVICE = v5.DEVICE


def run_eval(agent: v5.CDDPAgent, env: v5.DifferentiablePointMassEnv,
             n_episodes: int, clock_steps: int,
             force_stakes: float | None = None) -> dict:
    successes, violations = [], []

    completed = 0
    NUM_ENVS = v5.NUM_ENVS

    # initialise vectorised envs
    init_state = torch.zeros(NUM_ENVS, 5, dtype=torch.float32, device=DEVICE)
    env.reset(init_state, max_steps=clock_steps)
    goals = v5.GOAL.to(DEVICE).unsqueeze(0).expand(NUM_ENVS, -1).clone()
    env.set_goal(goals)

    active = torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE)
    per_env_steps = torch.zeros(NUM_ENVS, dtype=torch.long, device=DEVICE)
    ep_success = torch.zeros(NUM_ENVS, dtype=torch.bool, device=DEVICE)
    ep_violations = torch.zeros(NUM_ENVS, dtype=torch.long, device=DEVICE)
    launched = 0

    def launch_slots(slots):
        nonlocal launched
        n = min(int(slots.numel()), n_episodes - launched)
        if n <= 0:
            return
        slots = slots[:n]
        starts, episode_goals = v5.sample_episode_context(int(slots.numel()))
        # fixed clock
        starts[:, 4] = clock_steps / float(v5.LONG_CLOCK_STEPS)
        env._state[slots] = starts
        if env.elapsed_steps is not None:
            env.elapsed_steps[slots] = 0.0
        goals[slots] = episode_goals
        env.set_goal(goals)
        active[slots] = True
        per_env_steps[slots] = 0
        ep_success[slots] = False
        ep_violations[slots] = 0
        launched += int(slots.numel())

    launch_slots(torch.arange(NUM_ENVS, device=DEVICE))

    while completed < n_episodes:
        if not active.any():
            free = torch.where(~active)[0]
            launch_slots(free)
            if not active.any():
                break

        s = env.state.detach().clone()

        # stakes for active envs
        stakes = torch.zeros(NUM_ENVS, dtype=torch.float32, device=DEVICE)
        if force_stakes is not None:
            stakes[active] = force_stakes
        else:
            with torch.no_grad():
                if active.any():
                    stakes[active] = agent.stakes_from_deon(s[active], detach=True)

        # actor → safe action
        with torch.no_grad():
            s_aug = v5.aug_s(s, stakes)
            u_nom = torch.zeros(NUM_ENVS, 2, device=DEVICE)
            if active.any():
                u_nom[active] = agent.actor(s_aug[active])
            u_safe = torch.zeros_like(u_nom)
            if active.any():
                u_safe[active] = agent.safe_action(s[active], u_nom[active], stakes[active])

        # step — returns dict
        out = env.step(u_safe)
        unsafe_all = out["unsafe"].detach()  # [NUM_ENVS] bool

        per_env_steps[active] += 1

        # track violations only for active envs
        if active.any():
            ep_violations[active] += unsafe_all[active].long()

        # termination: goal reached or timeout
        reached = out["goal_reached"].detach() & active
        timeout = active & (per_env_steps >= clock_steps) & ~reached
        done = reached | timeout

        ep_success[reached] = True

        # mark done
        done_mask = active & done
        if done_mask.any():
            done_idx = torch.where(done_mask)[0]
            for idx in done_idx:
                successes.append(bool(ep_success[idx].item()))
                violations.append(int(ep_violations[idx].item()))
                active[idx] = False
                completed += 1

        # relaunch finished slots
        free = torch.where(~active)[0]
        if free.numel() > 0 and launched < n_episodes:
            launch_slots(free)

    return {
        "n": len(successes),
        "success_rate": sum(successes) / max(1, len(successes)),
        "avg_violations": sum(violations) / max(1, len(violations)),
        "pct_any_violation": sum(v > 0 for v in violations) / max(1, len(violations)),
        "total_violations": sum(violations),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="adaptive_v5")
    parser.add_argument("--n-eval", type=int, default=200,
                        help="Episodes per clock bucket")
    parser.add_argument("--timers", nargs="+", type=int,
                        default=[420, 600, 800, 1200, 2500],
                        help="Clock budgets to evaluate (env steps)")
    parser.add_argument("--force-stakes", type=float, default=None,
                        help="Override Z_deon: set stakes to this value for all steps (e.g. 1.0 or 0.0)")
    args = parser.parse_args()

    # set up results dir
    v5.configure_results_dir(args.run_name)
    ckpt_dir = v5.CHECKPOINT_DIR

    env = v5.make_env()
    agent = v5.CDDPAgent(env)

    # load best checkpoint
    best_actor = os.path.join(ckpt_dir, "cddp_actor.pt")
    best_zdeon = os.path.join(ckpt_dir, "cddp_z_deon.pt")
    if not os.path.exists(best_actor):
        print(f"No best checkpoint found at {best_actor}")
        sys.exit(1)
    agent.actor.load_state_dict(torch.load(best_actor, map_location=DEVICE))
    agent.z_deon.load_state_dict(torch.load(best_zdeon, map_location=DEVICE))
    agent.actor.eval()
    agent.z_deon.eval()
    print(f"Loaded best weights from {ckpt_dir}")
    force_stakes = args.force_stakes
    stakes_label = f"forced={force_stakes:.1f}" if force_stakes is not None else "Z_deon"
    print(f"Evaluating {args.n_eval} episodes per timer on {DEVICE}  [stakes: {stakes_label}]\n")
    n_eval = args.n_eval

    print(f"{'Timer':>8} | {'Success':>8} | {'AvgViol':>8} | {'%AnyViol':>9} | {'N':>5}")
    print("-" * 55)
    for clock in sorted(args.timers):
        result = run_eval(agent, env, n_eval, clock, force_stakes=force_stakes)
        print(f"{clock:>8} | {result['success_rate']:>8.1%} | "
              f"{result['avg_violations']:>8.1f} | "
              f"{result['pct_any_violation']:>9.1%} | "
              f"{result['n']:>5}")

    print()


if __name__ == "__main__":
    main()
