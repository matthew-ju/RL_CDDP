# CDDP MuJoCo — Time-Pressure Crowd

**Scaling the Constrained Deontic Decision Process to Continuous Control**

This directory graduates the CDDP framework from the 2D point-mass gridworld to a custom MuJoCo environment. The core dilemma — *when should a safety-aware agent intentionally violate constraints to avoid an even worse catastrophe?* — is now instantiated in continuous physics.

---

## Environment: Time-Pressure Crowd

The agent (a 2D planar Point robot) spawns near `(0, 0)` and must reach a circular goal zone at `(8, 0)`.

```
Start (0,0) ──[CROWD]──[CROWD]──[CROWD]──> Goal (8,0)   ← Direct path (~8 m)
     │                                                    ↗
     └──────────── Arc detour (y>0, ~12 m, obstacle-free) ─┘
```

| Signal | Value |
|---|---|
| Goal capture | **+500** |
| Timer expiry (no goal) | **-500** catastrophe |
| Step inside any obstacle | **-1** per step |
| Dense living cost | **-0.05** per step |
| Dense approach bonus | **+0.2 × Δ-dist** |

**State vector** (15-D): `[px, py, vx, vy | Δobs₀..₄ (2D each) | Δgoal (2D) | time_remaining_norm]`

---

## Algorithms

| Script | Algorithm | Expected Behaviour |
|---|---|---|
| `train_cddp.py` | **CDDP-DDPG** (ours) | Zero violations with long timer; adaptively violates when short |
| `train_ddpg.py` | Vanilla DDPG | Always charges through the crowd (reward hacking) |
| `train_ppo.py` | Vanilla PPO | Same reward-hacking pattern as DDPG |
| `train_strict_cbf.py` | DDPG + Strict CBF | Never violates; catastrophes at short timers |
| `train_ppo_lagrangian.py` | PPO-Lagrangian | Trial-and-error penalty; no structural guarantee |

---

## Quick Start

### 1. Install dependencies
```bash
pip install mujoco gymnasium torch numpy matplotlib
```

### 2. Smoke test (fast, CPU)
```bash
cd CDDP_RL/mujoco
python src/train_all.py --episodes 100 --max-steps 200 --device cpu
python src/evaluate_timer_sweep.py --n-eval 3 --timers 100 200
```

### 3. Full training run
```bash
python src/train_all.py --episodes 5000 --max-steps 500
```

### 4. Evaluate with timer sweep
```bash
python src/evaluate_timer_sweep.py \
    --timers 100 200 300 500 750 1000 1500 2000 \
    --n-eval 20
```

---

## CDDP Architecture

```
s  →  Z_deon(s)  →  P_cat(s) = sigmoid(Z_deon)
                   →  S(s) = clip(P_cat / P_ref, 0, 1)
                   
s  →  Actor(s)   →  u_nom
                   →  CBF_project(u_nom, S(s))  →  u_safe  →  Env
                   
u_safe, r  →  Z_free(s, a)  →  Q-loss  →  update Actor
```

**Key invariant:** Z_deon is trained **only** on strict (S=0) rollouts so it never learns that unsafe shortcuts are safe. The utilitarian critic and actor use the dynamic CBF boundary modulated by S(s).

### Stakes functional
```
S(s) = clip( sigmoid(Z_deon(s)) / P_ref , 0, 1 )    [P_ref = 0.2]
```

### Softplus boundary expansion
```
d_soft(S) = 10 · (1/α') · log(1 + exp(α'·(S − τ)))    [τ=0.3, α'=10]
```

When `S < τ`: `d_soft ≈ 0` → hard CBF enforcement.  
When `S > τ`: `d_soft > 0` → boundary expands, agent may cross obstacle.

---

## Output Files

```
results/
├── cddp/
│   ├── checkpoints/final/  actor.pt, z_free.pt, z_deon.pt
│   └── train_log.csv
├── ddpg/ ppo/ strict_cbf/ ppo_lag/   (same structure)
├── timer_sweep_raw.json        per-episode raw data
├── timer_sweep_summary.csv     aggregated metrics
├── timer_sweep_summary.json
├── pareto_frontier.json
└── plots/
    ├── success_rate.png
    ├── violations.png
    ├── excess_violations.png
    ├── prob_excess.png
    ├── catastrophe_rate.png
    └── pareto_frontier.png
```

---

## Evaluation Metrics

| Metric | Definition |
|---|---|
| **Success Rate** | Fraction of episodes reaching the goal |
| **Total Violations** | Mean obstacle-penetration steps per episode |
| **Excess Violations** | Violations above threshold (default: 0) |
| **P(Excess > 0)** | Fraction of episodes with any excess violation |
| **Pareto Frontier** | (Mean Violations, Mean Goal Distance) per algorithm |

CDDP uniquely shows: near-zero violations at long timers, then a sharp increase in violations at short timers — but accompanied by a sharp *increase* in success rate (it voluntarily cuts through the crowd to capture the goal before timeout).
