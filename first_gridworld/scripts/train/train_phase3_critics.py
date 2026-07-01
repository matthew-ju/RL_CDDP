import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'omnisafe_SRL'))

from cddp.env.hostage_env import InstrumentedHostageEnv
from cddp.shield.action_shield import ActionShield
from cddp.agents.ppo_actor import PPOActor
from cddp.agents.deontic_oracle import DeonticOraclePlanner
from cddp.critics.qr_critic import DualCritics
from cddp.replay.buffers import ReplayBuffer, DeonticReplayBuffer, Transition
from cddp.eval.metrics import min_violations_needed


def print_coverage_report(buffer, name='Buffer'):
    """Phase 1B: Print data coverage breakdown to confirm Grid 2 = 0%."""
    n = len(buffer)
    if n == 0:
        print(f"  [{name}] EMPTY")
        return

    T_rem  = buffer.T_rem[:n]
    D_min  = buffer.D_min[:n]
    in_fb  = buffer.in_failure_band[:n].astype(bool)
    margins = T_rem - D_min

    print(f"\n  [{name}] size={n}  in_failure_band={in_fb.mean():.2%}")

    # Margin buckets
    bucket_defs = [(-999, 0, '(  <0]'), (0, 1, '(0, 1]'), (1, 2, '(1, 2]'),
                   (2, 5, '(2, 5]'), (5, 9999, '(5+  ]')]
    for lo, hi, label in bucket_defs:
        mask = (margins > lo) & (margins <= hi)
        print(f"    margin {label}: {mask.mean():.2%}  (in_fb: {(mask & in_fb).mean():.2%})")

    # Step-0 fraction (step_idx stored in buffer)
    if hasattr(buffer, 'step_idx'):
        step0 = (buffer.step_idx[:n] == 0)
        print(f"    step_0: {step0.mean():.2%}  step_0 & in_fb: {(step0 & in_fb).mean():.2%}")

    # Layout breakdown — stored in buffer.layout_id if present
    if hasattr(buffer, 'layout_id'):
        ids, counts = np.unique(buffer.layout_id[:n], return_counts=True)
        layout_str = '  '.join(f'grid{int(g)}={c/n:.1%}' for g, c in zip(ids, counts))
        print(f"    layout: {layout_str}")
    else:
        print(f"    layout: [not stored in buffer — add layout_id to Transition for exact Grid 2 coverage]")


def collect_rollouts(env, actor, oracle, buffer, dual_critics, n_episodes=50, mode='utilitarian'):
    """Collect transitions and populate buffer."""
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        
        while not done and step < 50:
            pos = tuple(env.agent_pos)
            T_rem = env.get_T_rem()
            D_min = env.get_D_min(pos)
            L_safe = env.get_L_safe(pos)
            min_v = min_violations_needed(env, start_pos=pos, start_t_rem=T_rem)
            in_fb = env.is_in_failure_band(pos)
            safe_actions = env.get_safe_actions(pos)
            
            if mode == 'deon_oracle':
                action = oracle.get_action(env, pos)
                cost = env.get_constraint_cost(action, pos)
                if cost != 0:
                    # Oracle/env cost-function disagreement (can happen on Grid 2).
                    # Fall back to a random safe action so collection continues.
                    if safe_actions:
                        action = safe_actions[0]
                    # Skip storing this transition — don't poison D_deon with unsafe actions.
                    obs = next_obs if False else obs  # no-op; skip via continue below
                    step += 1
                    continue
            elif mode == 'deontic':
                probs = actor.get_discrete_probs(obs, n_samples=100)
                shield_mask = np.zeros(4)
                for a in safe_actions:
                    shield_mask[a] = 1.0
                masked_probs = probs * shield_mask
                if masked_probs.sum() > 0:
                    masked_probs /= masked_probs.sum()
                    action = np.random.choice(4, p=masked_probs)
                else:
                    action = np.random.choice(safe_actions)
            else:
                probs = actor.get_discrete_probs(obs, n_samples=100)
                action = np.random.choice(4, p=probs)
                
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done_flag = terminated or truncated
            
            next_safe_actions_mask = np.zeros(4, dtype=np.float32)
            for sa in env.get_safe_actions(tuple(env.agent_pos)):
                next_safe_actions_mask[sa] = 1.0
                
            next_action_oracle = oracle.get_action_for_state(env, tuple(env.agent_pos))

            next_pos = tuple(env.agent_pos)
            next_in_fb = env.is_in_failure_band(next_pos)

            t = Transition(
                obs=np.array(obs, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_obs=np.array(next_obs, dtype=np.float32),
                done=bool(done_flag),
                mode=mode,
                T_rem=float(T_rem),
                D_min=int(D_min),
                L_safe=float(L_safe),
                min_v=int(min_v) if min_v != float('inf') else 999,
                in_failure_band=bool(in_fb),
                next_in_failure_band=bool(next_in_fb),
                episode_id=ep,
                step_idx=step,
                next_safe_actions_mask=next_safe_actions_mask,
                next_action_oracle=int(next_action_oracle)
            )
            buffer.add(t)
            
            obs = next_obs
            done = done_flag
            step += 1

def main():
    print("=== Phase 3: Train Distributional Critics ===")
    
    # ── CLI ────────────────────────────────────────────────────────
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout_ids', type=int, nargs='+', default=[1],
                        help='Grid layouts to collect data from (e.g. 1 2 for both grids)')
    parser.add_argument('--save_dir', type=str, default=os.path.join(project_root, 'cddp_results/phase3_critics'),
                        help='Output directory for checkpoint and plots')
    parser.add_argument('--coverage_only', action='store_true',
                        help='Print coverage report from existing buffer and exit (dry-run)')
    parser.add_argument('--ckpt_path', type=str,
                        default=os.path.join(project_root, 'cddp_results/grid1_oracle_tau0.5/cddp_full_epoch_499.pt'))
    cli = parser.parse_args()

    layout_ids = cli.layout_ids
    save_dir   = cli.save_dir
    ckpt_path  = cli.ckpt_path

    # ── Use first layout for actor checkpoint ──────────────────────
    layout_id = layout_ids[0]   # kept for backward-compat in evaluate_state() below
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
        
    actor = PPOActor.from_checkpoint(ckpt_path)
    shield = ActionShield()
    oracle = DeonticOraclePlanner()
    
    # Dual Critics Manager
    n_quantiles = 21
    # Critics observe (actor_obs + normalized timer)
    critics_obs_dim = actor.obs_dim + 1
    critics = DualCritics(obs_dim=critics_obs_dim, num_actions=4, n_quantiles=n_quantiles, lr=1e-3)
    
    # Buffers
    D_all = ReplayBuffer(capacity=100000, obs_dim=actor.obs_dim)
    D_deon = DeonticReplayBuffer(capacity=100000, obs_dim=actor.obs_dim)
    
    # Threshold for catastrophe
    c_cat = -0.5
    
    if layout_id == 2:
        timers_to_collect = [21, 20, 18, 16, 14, 12, 10, 8, 7, None]
    else:
        timers_to_collect = [15, 12, 10, 8, 7, 6, 5, 4, None]
    n_eps = 200

    print(f"\n  layout_ids: {layout_ids}  →  save_dir: {save_dir}")
    print(f"  {'coverage_only=True: skip training' if cli.coverage_only else 'Full training run'}")

    print("\n--- Collecting Utilitarian Rollouts (Z_free) ---")
    for layout_id_i in layout_ids:
        for timer in timers_to_collect:
            env = InstrumentedHostageEnv.make(layout_id=layout_id_i, bomb_timer=timer, random_start=True, p_slip=0.05)
            collect_rollouts(env, actor, oracle, D_all, critics, n_episodes=n_eps, mode='utilitarian')
        print(f" Grid {layout_id_i} collected. D_all size: {len(D_all)}")
        
    print("\n--- Collecting Deontic Rollouts (Z_deon) ---")
    for layout_id_i in layout_ids:
        for timer in timers_to_collect:
            env = InstrumentedHostageEnv.make(layout_id=layout_id_i, bomb_timer=timer, random_start=True, p_slip=0.05)
            collect_rollouts(env, actor, oracle, D_deon, critics, n_episodes=n_eps, mode='deontic')
        print(f" Grid {layout_id_i} collected. D_deon size: {len(D_deon)}")

    # ── Phase 1B: Coverage report (always print, exit early if --coverage_only) ──
    print("\n--- Data Coverage Report ---")
    print_coverage_report(D_all,  name='D_all  (utilitarian)')
    print_coverage_report(D_deon, name='D_deon (oracle)')

    if cli.coverage_only:
        print("\n[--coverage_only] Exiting before training.")
        return

    print("\n--- Training Critics ---")
    
    # Calculate pos_weight for BCE With Logits Loss
    labels_all = (D_all.min_v[:len(D_all)] > 0) | ((D_all.T_rem[:len(D_all)] - D_all.D_min[:len(D_all)]) <= 1.0)
    pos_count = np.sum(labels_all)
    neg_count = len(D_all) - pos_count
    pos_weight_val = max(1.0, float(neg_count) / max(1.0, float(pos_count)))
    print(f"Dataset summary: Size={len(D_all)}, Positives={pos_count} ({(pos_count/len(D_all))*100:.2f}%)")
    print(f"Applying pos_weight={pos_weight_val:.2f} to BCE loss.")
    
    critics.bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]))
    
    batch_size = 128
    epochs = 50
    steps_per_epoch = 100
    
    losses_free = []
    losses_deon = []
    
    for epoch in range(epochs):
        lf_sum = 0
        ld_sum = 0
        lp_sum = 0
        
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        
        for _ in range(steps_per_epoch):
            # Train Z_free
            b_free = D_all.sample(batch_size)
            
            # Predict PPO actions for SARSA target
            next_obs_actor = b_free['next_obs_critic'][:, :actor.obs_dim].numpy()
            next_a_free = []
            for i in range(batch_size):
                probs = actor.get_discrete_probs(next_obs_actor[i], n_samples=100)
                next_a_free.append(np.argmax(probs))
            next_action_eval = torch.LongTensor(next_a_free)
            
            lf = critics.update_z_free(b_free['obs_critic'], b_free['actions'], 
                                       b_free['rewards'], b_free['next_obs_critic'], 
                                       b_free['dones'], next_action_eval)
            lf_sum += lf
            
            # Train Z_deon
            b_deon = D_deon.sample(batch_size)
            
            # Predict strict \pi_d actions for SARSA target
            next_obs_actor_deon = b_deon['next_obs_critic'][:, :actor.obs_dim].numpy()
            next_a_deon = []
            for i in range(batch_size):
                probs = actor.get_discrete_probs(next_obs_actor_deon[i], n_samples=100)
                mask = b_deon['next_safe_actions_mask'][i].numpy()
                masked_probs = probs * mask
                if masked_probs.sum() > 0:
                    masked_probs /= masked_probs.sum()
                    next_a_deon.append(np.argmax(masked_probs))
                else:
                    next_a_deon.append(np.argmax(mask))
            next_action_deon_eval = torch.LongTensor(next_a_deon)

            ld = critics.update_z_deon(b_deon['obs_critic'], b_deon['actions'], 
                                       b_deon['rewards'], b_deon['next_obs_critic'], 
                                       b_deon['dones'], next_action_deon_eval)
            ld_sum += ld
            
            # Train Failure Predictor
            margin = b_free['T_rem'] - b_free['D_min']
            b_target = (b_free['min_v'] > 0) | (margin <= 1.0)
            lp = critics.update_failure_predictor(b_free['obs_critic'], b_target)
            lp_sum += lp
            
            with torch.no_grad():
                preds = critics.predict_failure(b_free['obs_critic'])
                pred_labels = (preds >= 0.5)
                tp += ((pred_labels == 1) & (b_target == 1)).sum().item()
                fp += ((pred_labels == 1) & (b_target == 0)).sum().item()
                fn += ((pred_labels == 0) & (b_target == 1)).sum().item()
                tn += ((pred_labels == 0) & (b_target == 0)).sum().item()
            
        losses_free.append(lf_sum/steps_per_epoch)
        losses_deon.append(ld_sum/steps_per_epoch)
        
        if (epoch+1) % 10 == 0:
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            print(f"Epoch {epoch+1}/{epochs} | Z_free: {losses_free[-1]:.4f} | Z_deon: {losses_deon[-1]:.4f} | "
                  f"Predictor Loss: {lp_sum/steps_per_epoch:.4f} | Acc: {acc:.3f} | Prec: {precision:.3f} | Rec: {recall:.3f}")
            
    # Save checkpoint
    os.makedirs(save_dir, exist_ok=True)
    ckpt_out = os.path.join(save_dir, f'grid{layout_id}_dual_critics.pt')
    torch.save(critics.state_dict(), ckpt_out)
    print(f"  ✓ Saved: {ckpt_out}")
    
    print("\n--- Generating Diagnostic Reports ---")
    
    def evaluate_state(timer, state_label):
        env = InstrumentedHostageEnv.make(layout_id=layout_id, bomb_timer=timer)
        obs, _ = env.reset()
        
        obs_np = np.array(obs, dtype=np.float32)
        if obs_np.shape[-1] < actor.obs_dim:
            obs_np = np.concatenate([obs_np, np.zeros(actor.obs_dim - obs_np.shape[-1], dtype=np.float32)])
            
        t_feat = np.clip(timer / 25.0, 0.0, 1.0)
        obs_critic_np = np.concatenate([obs_np, [t_feat]])
        obs_th = torch.FloatTensor(obs_critic_np).unsqueeze(0)
        
        q_free_mean = critics.z_free.get_mean(obs_th)[0]
        a_free = q_free_mean.argmax(dim=-1).item()
        
        # EXACT ORACLE ACTION FOR DEON
        a_deon = oracle.get_action(env, tuple(env.agent_pos))
        
        # get quantiles for chosen actions
        q_free = np.sort(critics.z_free(obs_th)[0, a_free].detach().numpy())
        q_deon = np.sort(critics.z_deon(obs_th)[0, a_deon].detach().numpy())
        
        # P_cat
        p_free = np.mean(q_free <= c_cat)
        p_deon = np.mean(q_deon <= c_cat)
        
        print(f"State {state_label} (T={timer}): Z_deon P_cat = {p_deon:.2f} | Z_free P_cat = {p_free:.2f}")
        return q_free, q_deon, p_free, p_deon

    q_free_A, q_deon_A, p_free_A, p_deon_A = evaluate_state(5, 'A')
    q_free_B, q_deon_B, p_free_B, p_deon_B = evaluate_state(10, 'B')
    
    plt.figure(figsize=(10, 5))
    taus = [(i + 0.5) / n_quantiles for i in range(n_quantiles)]
    plt.plot(taus, q_free_A, label=f'Z_free T=5 (P={p_free_A:.2f})', marker='o')
    plt.plot(taus, q_deon_A, label=f'Z_deon T=5 (P={p_deon_A:.2f})', marker='s')
    plt.plot(taus, q_free_B, label=f'Z_free T=10 (P={p_free_B:.2f})', marker='^', linestyle='--')
    plt.plot(taus, q_deon_B, label=f'Z_deon T=10 (P={p_deon_B:.2f})', marker='x', linestyle='--')
    plt.axhline(c_cat, color='r', linestyle=':', label='Catastrophe')
    plt.title('Return Distribution Quantiles (Sorted) - Comparing Feasibility')
    plt.xlabel('Quantile Fraction (tau)')
    plt.ylabel('Return Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(project_root, 'cddp_results/phase3_critics/quantiles_comparison.png'))
    
    print("\n[Monotonicity Diagnostic] Evaluating P_cat_hat at boundary pos as T_rem changes.")
    env = InstrumentedHostageEnv.make(layout_id=layout_id, bomb_timer=15)
    env.reset()
    # Pick a spot known to be interesting (e.g. D_min = 12 on Grid 1... Wait, Grid 1 is small)
    pos_test = (4, 4)
    if layout_id == 2:
        pos_test = (9, 0) # Interesting spot found in trace
        
    try:
        env.env.agent_pos = np.array(pos_test)
        d_min_test = env.get_D_min(pos_test)
        print(f"Selected Test Pos: {pos_test}, D_min: {d_min_test}")
        
        test_timers = list(range(15, d_min_test - 3, -1))
        p_cats_mono = []
        for t in test_timers:
            env.env.bomb_time_left = t
            t_feat = np.clip(t / 25.0, 0.0, 1.0)
            obs = env.env._get_obs()
            obs_np = np.array(obs, dtype=np.float32)
            if obs_np.shape[-1] < actor.obs_dim:
                obs_np = np.concatenate([obs_np, np.zeros(actor.obs_dim - obs_np.shape[-1], dtype=np.float32)])
            obs_critic_np = np.concatenate([obs_np, [t_feat]])
            obs_th = torch.FloatTensor(obs_critic_np).unsqueeze(0)
            with torch.no_grad():
                p_c = critics.predict_failure(obs_th)[0].item()
            p_cats_mono.append(p_c)
            print(f"  T_rem={t} | margin={t - d_min_test} | p_cat_hat={p_c:.4f}")
            
        plt.figure(figsize=(6, 4))
        plt.plot(test_timers, p_cats_mono, marker='o')
        plt.axvline(d_min_test, color='r', linestyle='--', label='D_min (Failure Band)')
        plt.title(f'Monotonicity of P_cat_hat at pos {pos_test}')
        plt.xlabel('Bomb Timer Remaining')
        plt.ylabel('P_cat_hat')
        plt.gca().invert_xaxis()
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(project_root, 'cddp_results/phase3_critics/pcat_monotonicity.png'))
    except Exception as e:
        print(f"Warning: monotonicity plot failed. {e}")

    print("\nPhase 3 Deliverables Complete! Plots saved to cddp_results/phase3_critics/.")

if __name__ == '__main__':
    main()
