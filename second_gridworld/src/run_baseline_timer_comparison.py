"""Batched timer-sweep comparison for CDDP and PPO-family baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/private", "tmp", "cddp_rl_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

import src.train_baselines_ppo as baseline
import src.train_sandbox_cddp as cddp


MODELS = ["cddp", "ppo", "strict", "lagrangian", "cpo"]
LABELS = {
    "cddp": "CDDP",
    "ppo": "PPO",
    "strict": "Strict PPO",
    "lagrangian": "PPO-Lagrangian",
    "cpo": "CPO",
}
DEFAULT_TIMERS = list(range(340, 601, 10)) + list(range(650, 3001, 50))
SUMMARY_METRICS = [
    "success_rate",
    "total_violations",
    "core_violation_steps",
    "violation_obstacle_entries",
    "core_obstacle_entries",
    "final_distance_to_goal",
    "min_distance_to_goal",
    "path_efficiency",
    "steps_after_first_entering_goal_zone",
    "steps_dist_lt_0_5",
    "steps_dist_lt_1_0",
    "avg_stakes",
    "max_stakes",
    "frac_stakes_gt_0_8",
    "excess_violations",
    "has_excess",
]


def checkpoint_paths() -> dict[str, dict[str, str]]:
    paths = {
        "cddp": {
            "actor": os.path.join(cddp.CHECKPOINT_DIR, "cddp_actor_best.pt"),
            "z_free": os.path.join(cddp.CHECKPOINT_DIR, "cddp_z_free_best.pt"),
            "z_deon": os.path.join(cddp.CHECKPOINT_DIR, "cddp_z_deon_best.pt"),
        }
    }
    for algo in baseline.BASELINE_ALGOS:
        paths[algo] = {"actor_critic": baseline.checkpoint_path_for_algo(algo)}
    return paths


def load_agent(model: str) -> tuple[Any, Any]:
    env = cddp.make_env()
    paths = checkpoint_paths()[model]
    if model == "cddp":
        agent = cddp.CDDPAgent(env)
        agent.actor.load_state_dict(torch.load(paths["actor"], map_location=cddp.DEVICE))
        agent.z_free.load_state_dict(torch.load(paths["z_free"], map_location=cddp.DEVICE))
        agent.z_deon.load_state_dict(torch.load(paths["z_deon"], map_location=cddp.DEVICE))
        agent.actor.eval()
        agent.z_free.eval()
        agent.z_deon.eval()
        return agent, env

    agent = baseline.ActorCritic(cddp.MODEL_HIDDEN_DIM, cddp.MODEL_HIDDEN_LAYERS).to(cddp.DEVICE)
    agent.load_state_dict(torch.load(paths["actor_critic"], map_location=cddp.DEVICE))
    agent.eval()
    return agent, env


def require_checkpoints(models: list[str]) -> None:
    missing = [
        path
        for model in models
        for path in checkpoint_paths()[model].values()
        if not os.path.exists(path)
    ]
    if missing:
        raise FileNotFoundError("Missing checkpoint(s):\n" + "\n".join(missing))


def make_contexts(seed_count: int, episodes_per_seed: int, base_seed: int) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for seed_idx in range(seed_count):
        sample_seed = base_seed + seed_idx
        starts, goals = cddp.sample_episode_context(episodes_per_seed, seed=sample_seed)
        for episode in range(episodes_per_seed):
            contexts.append(
                {
                    "context_id": f"s{seed_idx:03d}_e{episode:03d}",
                    "seed": sample_seed,
                    "episode": episode,
                    "start_state": [float(v) for v in starts[episode].detach().cpu().tolist()],
                    "goal": [float(v) for v in goals[episode].detach().cpu().tolist()],
                }
            )
    return contexts


def violation_counts(pos: torch.Tensor, step: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    centers = cddp.obstacle_centers_at(
        timestep=torch.full((pos.shape[0],), float(step), dtype=pos.dtype, device=pos.device),
        batch_size=pos.shape[0],
        device=pos.device,
        dtype=pos.dtype,
    )
    distances = torch.norm(pos.unsqueeze(1) - centers, dim=2)
    layer3 = distances < cddp.LAYER_3_R.to(device=pos.device, dtype=pos.dtype).unsqueeze(0)
    core = distances < cddp.OBS_R.to(device=pos.device, dtype=pos.dtype).unsqueeze(0)
    return layer3.any(dim=1), layer3.sum(dim=1), core.any(dim=1), core.sum(dim=1)


@torch.no_grad()
def rollout_batch(model: str, agent: Any, env: Any, contexts: list[dict[str, Any]], timer: int) -> list[dict[str, Any]]:
    n = len(contexts)
    starts = torch.tensor([ctx["start_state"] for ctx in contexts], dtype=torch.float32, device=cddp.DEVICE)
    goals = torch.tensor([ctx["goal"] for ctx in contexts], dtype=torch.float32, device=cddp.DEVICE)
    starts[:, 4] = float(timer) / 1000.0
    env.reset(starts, max_steps=timer)
    env.set_goal(goals)
    if env.elapsed_steps is not None:
        env.elapsed_steps[:] = 0.0

    pos0 = starts[:, :2].detach().clone()
    prev_path_pos = pos0.clone()
    straight = torch.norm(pos0 - goals, dim=1)
    min_dist = straight.clone()
    final_dist = straight.clone()
    path_len = torch.zeros(n, dtype=torch.float32, device=cddp.DEVICE)
    success = torch.zeros(n, dtype=torch.bool, device=cddp.DEVICE)
    done = torch.zeros(n, dtype=torch.bool, device=cddp.DEVICE)
    total_steps = torch.zeros(n, dtype=torch.long, device=cddp.DEVICE)
    first_zone = torch.full((n,), -1, dtype=torch.long, device=cddp.DEVICE)
    first_tol = torch.full((n,), -1, dtype=torch.long, device=cddp.DEVICE)
    steps_lt_05 = torch.zeros(n, dtype=torch.long, device=cddp.DEVICE)
    steps_lt_10 = torch.zeros(n, dtype=torch.long, device=cddp.DEVICE)
    viol_steps = torch.zeros(n, dtype=torch.long, device=cddp.DEVICE)
    viol_entries = torch.zeros(n, dtype=torch.long, device=cddp.DEVICE)
    core_steps = torch.zeros(n, dtype=torch.long, device=cddp.DEVICE)
    core_entries = torch.zeros(n, dtype=torch.long, device=cddp.DEVICE)
    total_reward = torch.zeros(n, dtype=torch.float32, device=cddp.DEVICE)
    stakes_sum = torch.zeros(n, dtype=torch.float32, device=cddp.DEVICE)
    stakes_max = torch.zeros(n, dtype=torch.float32, device=cddp.DEVICE)
    stakes_gt_08 = torch.zeros(n, dtype=torch.long, device=cddp.DEVICE)
    stakes_count = torch.zeros(n, dtype=torch.long, device=cddp.DEVICE)
    low = cddp.GOAL_LOW.to(device=cddp.DEVICE)
    high = cddp.GOAL_HIGH.to(device=cddp.DEVICE)

    for step in range(1, timer + 1):
        active = ~done
        if not bool(active.any().detach().cpu().item()):
            break

        state = env.state.detach().clone()
        prev_pos = state[:, :2].clone()
        if model == "cddp":
            u_nom = agent.actor(state)
            stakes = agent.stakes_from_deon(state, detach=True).reshape(-1)
            action = agent.safe_action(state, u_nom, stakes=stakes)
        else:
            nominal = agent.act_mean(state)
            stakes = torch.zeros(n, dtype=state.dtype, device=state.device)
            if model == "strict":
                action = env.project_action_cbf(
                    state,
                    nominal,
                    stakes=stakes,
                    gamma=cddp.GAMMA_CBF,
                    tau=cddp.TAU_CDDP,
                    alpha_prime=cddp.ALPHA_CDDP,
                )
            else:
                action = nominal.clamp(-cddp.U_MAX, cddp.U_MAX)
        action = torch.where(active.unsqueeze(1), action, torch.zeros_like(action))

        out = env.step(action)
        next_state = out["state"].detach().clone()
        next_state = torch.where(done.unsqueeze(1), state, next_state)
        next_state[:, 4] = float(max(0, timer - step)) / 1000.0
        env._state = next_state
        if env.elapsed_steps is not None:
            env.elapsed_steps[:] = torch.where(
                done,
                total_steps.to(dtype=env.elapsed_steps.dtype),
                torch.full_like(env.elapsed_steps, float(step)),
            )

        reached = cddp.success_mask(next_state[:, :2], goals, out["goal_reached"].detach()) & active
        timeout = (step >= timer) & active & ~reached
        reward = cddp.compute_reward(
            next_state[:, :2],
            prev_pos,
            goals,
            reached,
            out["unsafe"].detach() & active,
            timeout,
            timestep=torch.full((n,), float(step), dtype=torch.float32, device=cddp.DEVICE),
        )
        total_reward += torch.where(active, reward.to(torch.float32), torch.zeros_like(total_reward))

        pos = next_state[:, :2]
        dist = torch.norm(pos - goals, dim=1)
        path_len += torch.where(active, torch.norm(pos - prev_path_pos, dim=1), torch.zeros_like(path_len))
        prev_path_pos = torch.where(active.unsqueeze(1), pos, prev_path_pos)
        min_dist = torch.where(active, torch.minimum(min_dist, dist), min_dist)
        final_dist = torch.where(active, dist, final_dist)

        in_zone = ((pos >= low) & (pos <= high)).all(dim=1) & active
        in_tol = (dist < cddp.GOAL_TOL) & active
        first_zone = torch.where((first_zone < 0) & in_zone, torch.full_like(first_zone, step), first_zone)
        first_tol = torch.where((first_tol < 0) & in_tol, torch.full_like(first_tol, step), first_tol)
        steps_lt_05 += ((dist < 0.5) & active).long()
        steps_lt_10 += ((dist < 1.0) & active).long()

        layer_hit, layer_entry, core_hit, core_entry = violation_counts(pos, step)
        viol_steps += (layer_hit & active).long()
        viol_entries += torch.where(active, layer_entry.long(), torch.zeros_like(viol_entries))
        core_steps += (core_hit & active).long()
        core_entries += torch.where(active, core_entry.long(), torch.zeros_like(core_entries))

        stakes_sum += torch.where(active, stakes, torch.zeros_like(stakes_sum))
        stakes_max = torch.where(active, torch.maximum(stakes_max, stakes), stakes_max)
        stakes_gt_08 += ((stakes > 0.8) & active).long()
        stakes_count += active.long()
        total_steps = torch.where(active, torch.full_like(total_steps, step), total_steps)
        success |= reached
        done |= reached | timeout

    cpu = {
        "success": success.cpu().tolist(),
        "steps": total_steps.cpu().tolist(),
        "reward": total_reward.cpu().tolist(),
        "viol": viol_steps.cpu().tolist(),
        "viol_entries": viol_entries.cpu().tolist(),
        "core": core_steps.cpu().tolist(),
        "core_entries": core_entries.cpu().tolist(),
        "final_dist": final_dist.cpu().tolist(),
        "min_dist": min_dist.cpu().tolist(),
        "first_zone": first_zone.cpu().tolist(),
        "first_tol": first_tol.cpu().tolist(),
        "lt05": steps_lt_05.cpu().tolist(),
        "lt10": steps_lt_10.cpu().tolist(),
        "path": path_len.cpu().tolist(),
        "straight": straight.cpu().tolist(),
        "stakes_sum": stakes_sum.cpu().tolist(),
        "stakes_max": stakes_max.cpu().tolist(),
        "stakes_gt": stakes_gt_08.cpu().tolist(),
        "stakes_count": stakes_count.cpu().tolist(),
    }

    records: list[dict[str, Any]] = []
    for i, ctx in enumerate(contexts):
        first_zone_i = int(cpu["first_zone"][i])
        first_tol_i = int(cpu["first_tol"][i])
        total_steps_i = int(cpu["steps"][i])
        path_i = float(cpu["path"][i])
        straight_i = float(cpu["straight"][i])
        count_i = max(1, int(cpu["stakes_count"][i]))
        records.append(
            {
                "model": model,
                "model_label": LABELS[model],
                "timer": int(timer),
                "seed": int(ctx["seed"]),
                "sample_seed": int(ctx["seed"]),
                "episode": int(ctx["episode"]),
                "context_id": ctx["context_id"],
                "success": bool(cpu["success"][i]),
                "success_rate": 1.0 if cpu["success"][i] else 0.0,
                "total_steps": total_steps_i,
                "total_violations": int(cpu["viol"][i]),
                "violation_steps": int(cpu["viol"][i]),
                "violation_obstacle_entries": int(cpu["viol_entries"][i]),
                "core_violation_steps": int(cpu["core"][i]),
                "core_obstacle_entries": int(cpu["core_entries"][i]),
                "reference_violations": None,
                "excess_defined": False,
                "excess_violations": None,
                "has_excess": None,
                "final_distance_to_goal": float(cpu["final_dist"][i]),
                "min_distance_to_goal": float(cpu["min_dist"][i]),
                "first_step_inside_goal_zone": None if first_zone_i < 0 else first_zone_i,
                "first_step_inside_goal_tol": None if first_tol_i < 0 else first_tol_i,
                "steps_after_first_entering_goal_zone": None if first_zone_i < 0 else total_steps_i - first_zone_i,
                "steps_dist_lt_0_5": int(cpu["lt05"][i]),
                "steps_dist_lt_1_0": int(cpu["lt10"][i]),
                "path_length": path_i,
                "straight_line_distance": straight_i,
                "path_efficiency": straight_i / max(path_i, 1e-8),
                "avg_stakes": float(cpu["stakes_sum"][i]) / count_i,
                "max_stakes": float(cpu["stakes_max"][i]),
                "frac_stakes_gt_0_8": float(cpu["stakes_gt"][i]) / count_i,
                "start_state": ctx["start_state"],
                "goal": ctx["goal"],
                "start_x": float(ctx["start_state"][0]),
                "start_y": float(ctx["start_state"][1]),
                "goal_x": float(ctx["goal"][0]),
                "goal_y": float(ctx["goal"][1]),
                "total_reward": float(cpu["reward"][i]),
            }
        )
    return records


def add_excess(records: list[dict[str, Any]]) -> None:
    by_context: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_context[(row["context_id"], int(row["timer"]))].append(row)
    refs = {
        key: min(int(row["total_violations"]) for row in rows if row["success"])
        for key, rows in by_context.items()
        if any(row["success"] for row in rows)
    }
    for row in records:
        ref = refs.get((row["context_id"], int(row["timer"])))
        row["reference_violations"] = ref
        row["excess_defined"] = bool(row["success"] and ref is not None)
        if row["excess_defined"]:
            excess = max(0, int(row["total_violations"]) - int(ref))
            row["excess_violations"] = excess
            row["has_excess"] = 1.0 if excess > 0 else 0.0


def mean_std(values: list[Any]) -> tuple[float | None, float | None]:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None, None
    mean = sum(clean) / len(clean)
    return mean, math.sqrt(sum((v - mean) ** 2 for v in clean) / len(clean))


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[(row["model"], int(row["timer"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, timer), rows in sorted(groups.items(), key=lambda item: (MODELS.index(item[0][0]), item[0][1])):
        summary: dict[str, Any] = {"model": model, "model_label": LABELS[model], "timer": timer}
        for metric in SUMMARY_METRICS:
            mean, std = mean_std([row.get(metric) for row in rows])
            summary[f"mean_{metric}"] = mean
            summary[f"std_{metric}"] = std
        output.append(summary)
    return output


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def plot_summary(path: str, rows: list[dict[str, Any]], models: list[str]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["model"]].append(row)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    specs = [
        ("mean_success_rate", "Success Rate", (0.0, 1.05)),
        ("mean_total_violations", "Layer-3 Violation Steps", None),
        ("mean_core_violation_steps", "Core Violation Steps", None),
        ("mean_final_distance_to_goal", "Final Distance to Goal", None),
        ("mean_path_efficiency", "Path Efficiency", None),
        ("mean_has_excess", "P(Excess Violations)", (0.0, 1.05)),
    ]
    for ax, (metric, title, ylim) in zip(axes.flatten(), specs):
        for model in models:
            model_rows = grouped[model]
            ax.plot(
                [row["timer"] for row in model_rows],
                [row.get(metric) for row in model_rows],
                marker="o",
                markersize=2.5,
                linewidth=1.4,
                label=LABELS[model],
            )
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        if ylim is not None:
            ax.set_ylim(*ylim)
    for ax in axes[-1, :]:
        ax.set_xlabel("Timer (environment steps)")
    axes[0, 0].set_ylabel("Mean")
    axes[1, 0].set_ylabel("Mean")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models), frameon=False)
    fig.suptitle("Baseline Timer Sweep Comparison")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_timers(raw: str | None) -> list[int]:
    if raw is None:
        return DEFAULT_TIMERS
    return sorted({int(item.strip()) for item in raw.split(",") if item.strip()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CDDP, CPO, PPO, Lagrangian, and Strict baselines.")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--timers", default=None)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--episodes-per-seed", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=cddp.EVAL_SEED)
    parser.add_argument("--output-prefix", default="baseline_timer_sweep")
    args = parser.parse_args()

    models = [model.strip().lower() for model in args.models.split(",") if model.strip()]
    invalid = [model for model in models if model not in MODELS]
    if invalid:
        raise ValueError(f"Unsupported model(s): {invalid}")
    timers = parse_timers(args.timers)
    require_checkpoints(models)

    out_dir = os.path.join(cddp.RESULTS_ROOT, "baseline_comparison")
    os.makedirs(out_dir, exist_ok=True)
    contexts = make_contexts(args.seeds, args.episodes_per_seed, args.base_seed)
    contexts_path = os.path.join(out_dir, f"{args.output_prefix}_contexts.json")
    with open(contexts_path, "w", encoding="utf-8") as f:
        json.dump(contexts, f, indent=2)

    agents = {model: load_agent(model) for model in models}
    records: list[dict[str, Any]] = []
    total = len(models) * len(timers) * len(contexts)
    done = 0
    for model in models:
        agent, env = agents[model]
        for timer in timers:
            records.extend(rollout_batch(model, agent, env, contexts, timer))
            done += len(contexts)
            print(f"[{done}/{total}] {LABELS[model]} timer={timer}", flush=True)

    add_excess(records)
    summary = summarize(records)
    records_path = os.path.join(out_dir, f"{args.output_prefix}_records.csv")
    summary_path = os.path.join(out_dir, f"{args.output_prefix}_summary.csv")
    json_path = os.path.join(out_dir, f"{args.output_prefix}.json")
    plot_path = os.path.join(out_dir, "baseline_timer_comparison_2x3.png")

    write_csv(records_path, records)
    write_csv(summary_path, summary)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "models": models,
                    "model_labels": LABELS,
                    "checkpoint_paths": {model: checkpoint_paths()[model] for model in models},
                    "timers": timers,
                    "seeds": int(args.seeds),
                    "episodes_per_seed": int(args.episodes_per_seed),
                    "total_contexts": len(contexts),
                    "contexts_path": contexts_path,
                    "base_seed": int(args.base_seed),
                    "diagnostics_version": 3,
                    "timer_unit": "environment steps",
                    "violation_zone": "layer3",
                    "environment": cddp.task_config(),
                },
                "results": {model: [row for row in summary if row["model"] == model] for model in models},
                "records": records,
            },
            f,
            indent=2,
        )
    plot_summary(plot_path, summary, models)
    print(f"Saved records CSV: {records_path}")
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
