#!/usr/bin/env python3
"""
Compares a CDDP run against all baselines.
Focuses on the key adaptive metrics: violations and success by clock type.
Runs regardless of whether the adaptive verdict is good or bad.
"""
import json, argparse
import numpy as np
from pathlib import Path


def load_split(path):
    """Load an eval log and split episodes into tight (≤600 steps) and loose (>600)."""
    if not path.exists():
        return None, None
    with open(path) as f:
        raw = json.load(f)
    eps = raw if isinstance(raw, list) else raw.get("episodes", [])
    tight, loose = [], []
    for ep in eps:
        ms = ep.get("max_steps") or ep.get("episode_max_steps")
        if ms is None:
            continue
        viols = ep.get("total_violations", 0)
        if isinstance(viols, list):
            viols = len([v for v in viols if v])
        rec = {
            "success":    ep.get("success", False),
            "violations": float(viols) if viols is not None else 0.0,
        }
        (tight if float(ms) <= 600 else loose).append(rec)
    return tight, loose


def stat(eps):
    if not eps:
        return {"success": None, "avg_viol": None, "n": 0}
    return {
        "success":  float(np.mean([e["success"] for e in eps])),
        "avg_viol": float(np.mean([e["violations"] for e in eps])),
        "n":        len(eps),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="adaptive_v1",
                    help="CDDP run name to compare as the focal model")
    args = ap.parse_args()

    MODELS = [
        (f"CDDP {args.run_name}",  Path(f"results/cddp/{args.run_name}/cddp_eval_log.json")),
        ("CDDP gamma07_fix",       Path("results/cddp/gamma07_fix/cddp_eval_log.json")),
        ("PPO-Lagrangian",         Path("results/lagrangian/lagrangian_eval_log.json")),
        ("PPO",                    Path("results/ppo/ppo_eval_log.json")),
        ("CPO",                    Path("results/cpo/cpo_eval_log.json")),
        ("Strict",                 Path("results/strict/strict_eval_log.json")),
    ]

    rows = []
    missing = []
    for name, path in MODELS:
        tight_eps, loose_eps = load_split(path)
        if tight_eps is None:
            missing.append(name)
            continue
        t = stat(tight_eps)
        l = stat(loose_eps)
        ratio = None
        if t["avg_viol"] is not None and l["avg_viol"] is not None:
            ratio = t["avg_viol"] / (l["avg_viol"] + 0.5)
        rows.append({"name": name, "tight": t, "loose": l, "ratio": ratio})

    print("\n" + "="*82)
    print("  FRAMEWORK COMPARISON — Tight vs Loose Clock")
    print("  Ideal CDDP: high tight success + violations | high loose success + ~0 violations")
    print("="*82)

    hdr = (f"{'Framework':<25} {'Tight Succ':>10} {'Tight Viol':>11} "
           f"{'Loose Succ':>10} {'Loose Viol':>11} {'Viol Ratio':>11}")
    print(hdr)
    print("-"*82)

    def fmt_pct(v):   return f"{v*100:.0f}%" if v is not None else "N/A"
    def fmt_f(v):     return f"{v:.1f}"      if v is not None else "N/A"
    def fmt_ratio(v): return f"{v:.1f}x"     if v is not None else "N/A"

    for r in rows:
        t, l = r["tight"], r["loose"]
        marker = " ◄" if args.run_name in r["name"] else ""
        print(f"  {r['name']:<23} {fmt_pct(t['success']):>10} {fmt_f(t['avg_viol']):>11} "
              f"{fmt_pct(l['success']):>10} {fmt_f(l['avg_viol']):>11} "
              f"{fmt_ratio(r['ratio']):>11}{marker}")

    print("="*82)
    print("  Viol Ratio = tight_avg_viol / (loose_avg_viol + 0.5)")
    print("  Target for CDDP adaptive: Ratio ≥ 4.0x (clear adaptive signature)")
    if missing:
        print(f"  Missing eval logs for: {', '.join(missing)}")
    print("="*82 + "\n")

    # Save comparison JSON
    out = Path(f"results/cddp/{args.run_name}/baseline_comparison.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"  Saved: {out}\n")


if __name__ == "__main__":
    main()
