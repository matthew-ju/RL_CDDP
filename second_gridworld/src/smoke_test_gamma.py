"""
Smoke test: verify strict CBF navigation is feasible with GAMMA_CBF=0.7.
Run: python3 src/smoke_test_gamma.py
"""
import torch, math, sys, os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cddp_mpl')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.train_sandbox_cddp as cddp

print('GAMMA_CBF =', cddp.GAMMA_CBF)
device = cddp.DEVICE
env = cddp.make_env()

torch.manual_seed(42)
N = 64

start_states = torch.zeros(N, 5, device=device)
start_states[:, 0] = torch.rand(N, device=device) * 1.0
start_states[:, 1] = torch.rand(N, device=device) * 1.0
start_states[:, 4] = 1.0

env.reset(start_states, max_steps=2500)
goals = cddp.GOAL.to(device).unsqueeze(0).expand(N, -1).clone()
env.set_goal(goals)

active = torch.ones(N, dtype=torch.bool, device=device)
successes_mask = torch.zeros(N, dtype=torch.bool, device=device)
stakes = torch.zeros(N, device=device)
min_dists = torch.full((N,), float('inf'), device=device)

for step in range(2500):
    if not active.any():
        break
    s = env.state.detach().clone()
    pos = s[:, :2]
    to_goal = goals - pos
    dist = torch.norm(to_goal, dim=1, keepdim=True).clamp(min=1e-6)
    u_nom = (to_goal / dist) * cddp.U_MAX
    u_nom = u_nom.clamp(-cddp.U_MAX, cddp.U_MAX)
    u_safe = env.project_action_cbf(s, u_nom, stakes=stakes, gamma=cddp.GAMMA_CBF,
                                     tau=cddp.TAU_CDDP, alpha_prime=cddp.ALPHA_CDDP)
    u_safe[~active] = 0.0
    out = env.step(u_safe)
    next_s = out['state']
    d = torch.norm(next_s[:, :2] - goals, dim=1)
    min_dists = torch.min(min_dists, d)
    reached = out['goal_reached']
    successes_mask |= (reached & active)
    active &= ~reached

successes = successes_mask.sum().item()
print('Strict CBF smoke test (N=%d, timer=2500, proportional controller):' % N)
print('  Successes: %d/%d (%.1f%%)' % (successes, N, 100*successes/N))
print('  Min dist to goal across all envs: %.3f' % min_dists.min().item())
print('  Mean min dist: %.3f' % min_dists.mean().item())
if successes > 0:
    print('  PASS: strict CBF is navigable — Z_deon will get meaningful supervision.')
else:
    print('  FAIL: strict CBF still impassable — need further gamma reduction.')
