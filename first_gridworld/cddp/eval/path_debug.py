"""
Path-level debugging for CDDP agent behavior analysis.

Provides:
    1. render_grid_with_path — ASCII grid visualization with arrows
    2. trace_episode — per-step diagnostic trace
    3. print_timer_sweep_debug — multi-timer diagnostic sweep

Usage:
    python -m cddp.eval.path_debug --grid 1 --checkpoint <path>
"""

import sys
import os
import argparse
import math

_omnisafe_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'omnisafe_SRL')
sys.path.insert(0, os.path.abspath(_omnisafe_dir))

from cddp.env.hostage_env import InstrumentedHostageEnv
from cddp.shield.action_shield import ActionShield
from cddp.stakes.analytical import AnalyticalStakesDmin, AnalyticalStakesLsafe, AnalyticalFeasibilityTrigger
from cddp.agents.ppo_actor import PPOActor
from cddp.agents.cddp_agent import CDDPAgent

ACTION_ARROWS = {0: '↑', 1: '↓', 2: '←', 3: '→'}
ACTION_NAMES = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}


def render_grid_with_path(env, trajectory, show_step_idx=False):
    """Render ASCII grid with the agent's path marked.

    Args:
        env: InstrumentedHostageEnv (used for grid layout info)
        trajectory: list of step dicts from trace_episode
        show_step_idx: if True, show step index instead of arrows

    Returns:
        str: printable ASCII grid
    """
    grid_size = env.grid_size
    civilians = set(env.civilians)
    walls = env.walls
    hostage = env.hostage_pos
    trampled = env.trampled_civilians if hasattr(env, 'trampled_civilians') else set()

    # Build position→arrow map from trajectory
    pos_to_arrow = {}
    pos_to_step = {}
    for step in trajectory:
        pos = step['position']
        action = step['action']
        t = step['step']
        pos_to_arrow[pos] = ACTION_ARROWS.get(action, '?')
        pos_to_step[pos] = t

    # Start and end positions
    start_pos = trajectory[0]['position'] if trajectory else None
    end_pos = trajectory[-1]['next_position'] if trajectory else None

    # Column header
    lines = ['    ' + '   '.join(str(c) for c in range(grid_size))]

    for r in range(grid_size):
        row = f'{r}  '
        for c in range(grid_size):
            pos = (r, c)
            if pos == hostage:
                cell = ' H '
            elif pos in walls:
                cell = ' # '
            elif pos in trampled:
                cell = ' X '
            elif pos in civilians:
                cell = ' C '
            elif pos in pos_to_arrow:
                if show_step_idx:
                    cell = f'{pos_to_step[pos]:2d} '
                else:
                    cell = f' {pos_to_arrow[pos]} '
            elif pos == end_pos and pos not in pos_to_arrow:
                cell = ' E '
            else:
                cell = ' . '
            row += cell + ' '
        lines.append(row)

    return '\n'.join(lines)


def trace_episode(agent, layout_id, bomb_timer, deterministic=True, max_steps=50):
    """Run one episode and collect a detailed per-step trace.

    Args:
        agent: CDDPAgent instance
        layout_id: grid layout
        bomb_timer: bomb timer (None for no bomb)
        deterministic: deterministic policy
        max_steps: safety limit

    Returns:
        dict with keys:
            'trajectory': list of per-step dicts
            'summary': dict with episode-level stats
            'grid_render': str (ASCII grid with path)
    """
    env = InstrumentedHostageEnv.make(layout_id=layout_id, bomb_timer=bomb_timer)
    obs, info = env.reset()
    agent.reset_mode_tracking()

    trajectory = []
    done = False
    step = 0
    prev_mode = None

    while not done and step < max_steps:
        pos = tuple(env.agent_pos)
        T_rem_before = env.get_T_rem()
        D_min = env.get_D_min(pos)
        L_safe = env.get_L_safe(pos)
        safe_actions = env.get_safe_actions(pos)
        in_fb = env.is_in_failure_band(pos)

        # Select action
        action, act_info = agent.select_action(obs, env, deterministic=deterministic)

        stakes = act_info['stakes']
        mode = act_info['mode']
        violation = env.get_constraint_cost(action, pos)
        switch = mode != prev_mode if prev_mode is not None else False
        prev_mode = mode

        # Step environment
        next_obs, reward, terminated, truncated, new_info = env.step(action)
        done_flag = terminated or truncated
        next_pos = tuple(env.agent_pos)
        T_rem_after = env.get_T_rem()

        step_record = {
            'step': step,
            'position': pos,
            'action': action,
            'action_name': ACTION_NAMES[action],
            'next_position': next_pos,
            'reward': reward,
            'violation': violation,
            'T_rem_before': T_rem_before,
            'T_rem_after': T_rem_after,
            'D_min': D_min,
            'L_safe': L_safe,
            'stakes': stakes,
            'mode': mode,
            'safe_actions': safe_actions,
            'in_failure_band': in_fb,
            'switch_event': switch,
            'action_was_shielded': act_info.get('action_was_shielded', False),
            'ranked_safe_actions': act_info.get('ranked_safe_actions'),
            'deontic_tiebreak_reason': act_info.get('deontic_tiebreak_reason'),
            'terminated': terminated,
            'truncated': truncated,
        }
        trajectory.append(step_record)

        obs = next_obs
        info = new_info
        done = done_flag
        step += 1

    # Episode summary
    saved = tuple(env.agent_pos) == env.hostage_pos
    total_violations = len(env.trampled_civilians)
    deontic_steps = sum(1 for s in trajectory if s['mode'] == 'deontic')
    util_steps = sum(1 for s in trajectory if s['mode'] == 'utilitarian')
    switches = sum(1 for s in trajectory if s['switch_event'])
    shielded = sum(1 for s in trajectory if s['action_was_shielded'])

    summary = {
        'saved': saved,
        'total_violations': total_violations,
        'total_steps': step,
        'deontic_steps': deontic_steps,
        'utilitarian_steps': util_steps,
        'mode_switches': switches,
        'shield_overrides': shielded,
        'bomb_timer': bomb_timer,
        'layout_id': layout_id,
    }

    grid_render = render_grid_with_path(env, trajectory)

    return {
        'trajectory': trajectory,
        'summary': summary,
        'grid_render': grid_render,
    }


def format_compact_step_table(trajectory, tau=0.5):
    """Format trajectory as a compact table string."""
    lines = []
    hdr = (f"{'t':>2} {'pos':>7} {'act':>5} {'T_rem':>5} {'D_min':>5} {'L_safe':>6} "
           f"{'S':>4} {'cond':>5} {'mode':>4} {'shld':>4} {'viol':>4} {'switch':>6}")
    lines.append(hdr)
    lines.append('─' * len(hdr))

    for s in trajectory:
        t_str = f"{s['T_rem_before']:.0f}" if not math.isinf(s['T_rem_before']) else '∞'
        l_str = f"{s['L_safe']:.0f}" if not math.isinf(s['L_safe']) else '∞'
        cond_str = 'S<τ' if s['stakes'] < tau else 'S>=τ'
        mode_str = 'DEON' if s['mode'] == 'deontic' else 'UTIL'
        lines.append(
            f"{s['step']:2d} {str(s['position']):>7} {s['action_name']:>5} "
            f"{t_str:>5} {s['D_min']:5d} {l_str:>6} "
            f"{s['stakes']:4.2f} {cond_str:>5} "
            f"{mode_str:>4} "
            f"{'Y' if s['action_was_shielded'] else '.':>4} "
            f"{'X' if s['violation'] > 0 else '.':>4} "
            f"{'←→' if s['switch_event'] else '  ':>6}"
        )
    return '\n'.join(lines)


def print_timer_sweep_debug(agent, layout_id, timers=None, n_episodes=3):
    """Run diagnostic sweep across multiple bomb timers.

    Args:
        agent: CDDPAgent instance
        layout_id: grid layout
        timers: list of bomb timers to test (None = no bomb)
        n_episodes: episodes per timer
    """
    if timers is None:
        timers = [None, 10, 8, 7, 6, 5, 4, 3]

    for timer in timers:
        timer_label = f'T={timer}' if timer is not None else 'No bomb'
        print(f"\n{'='*60}")
        print(f"Timer: {timer_label}, Grid {layout_id}, {n_episodes} episodes")
        print(f"Rule:  DEON if S < {agent.tau} else UTIL")
        print(f"{'='*60}")

        for ep in range(n_episodes):
            result = trace_episode(agent, layout_id, timer, deterministic=True)
            s = result['summary']

            print(f"\n── Episode {ep} ──")
            print(f"  Result: saved={s['saved']}, violations={s['total_violations']}, "
                  f"steps={s['total_steps']}")
            print(f"  Modes: deontic={s['deontic_steps']}, util={s['utilitarian_steps']}, "
                  f"switches={s['mode_switches']}, shielded={s['shield_overrides']}")
            print()
            print(result['grid_render'])
            print()
            print(format_compact_step_table(result['trajectory'], tau=agent.tau))


# ─── CLI ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CDDP Path Debugger')
    parser.add_argument('--grid', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--stakes', type=str, choices=['dmin', 'lsafe', 'feas'], default='dmin')
    parser.add_argument('--timers', type=str, default='None,10,8,7,6,5,4,3',
                        help='Comma-separated timers (use None for no bomb)')
    parser.add_argument('--episodes', type=int, default=3)
    args = parser.parse_args()

    # Parse timers
    timers = []
    for t in args.timers.split(','):
        t = t.strip()
        timers.append(None if t.lower() == 'none' else int(t))

    # Load agent
    actor = PPOActor.from_checkpoint(args.checkpoint)
    shield = ActionShield()
    
    if args.stakes == 'lsafe':
        stakes_fn = AnalyticalStakesLsafe()
    elif args.stakes == 'feas':
        stakes_fn = AnalyticalFeasibilityTrigger()
    else:
        stakes_fn = AnalyticalStakesDmin()
        
    agent = CDDPAgent(actor, shield, stakes_fn, tau=args.tau)

    print_timer_sweep_debug(agent, args.grid, timers=timers, n_episodes=args.episodes)


if __name__ == '__main__':
    main()
