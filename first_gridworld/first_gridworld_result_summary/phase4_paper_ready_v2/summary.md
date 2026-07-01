# Phase 4 v2 — Paper-Ready Evaluation Summary
## Configuration
- **Seeds:** 15  |  **Episodes/seed:** 200
- **Episode seeds:** fixed per (seed, grid), shared across all T and agents
- **Oracle:** PlannerOracleAgent (BFS deontic + Dijkstra utilitarian)
- **Baseline:** PPO-Baseline-Finetuned (G1 + G2)


## Grid 1 (T=3–12)
**L_safe drift across T:** 0.000 steps ✅

### Tightest timer (T=3)
| Model | Success% | Viols | Excess | P(excess>0)% |
|-------|----------|-------|--------|-------------|
| PPO-Baseline-Finetuned | 94.4 | 0.542 | 0.007 | 0.6 |
| Deontic-Oracle | 61.5 | 0.017 | 0.006 | 0.6 |
| CDDP-Oracle | 94.3 | 0.553 | 0.007 | 0.6 |
| CDDP-Prospective | 94.4 | 0.542 | 0.007 | 0.6 |

## Grid 2 (T=5–20)
**L_safe drift across T:** 0.000 steps ✅

### Tightest timer (T=5)
| Model | Success% | Viols | Excess | P(excess>0)% |
|-------|----------|-------|--------|-------------|
| PPO-Baseline-Finetuned | 91.3 | 0.564 | 0.117 | 11.6 |
| Deontic-Oracle | 63.8 | 0.052 | 0.010 | 1.0 |
| CDDP-Oracle | 89.8 | 0.479 | 0.013 | 1.3 |
| CDDP-Prospective | 91.3 | 0.564 | 0.117 | 11.6 |
