"""
Prospective Stakes — Learned stakes from distributional critics.

Replaces the oracle S_analytic = D_min / T_rem with a learned quantity
computed from the QR distributional critics Z_free and Z_deon.

The prospective stakes function calculates the strict Cathedral-bound:
    S(s) = clip(P_cat(s) / P_ref, 0, 1)
where P_cat(s) is the fraction of Z_deon(s, a_d) quantiles <= c_cat.
"""

import math
import torch
import numpy as np

class ProspectiveStakes:
    """Prospective stakes: S_pros(s) = clip(P_cat(s) / P_ref, 0, 1).
    
    Evaluates P_cat(s) under the true Deontic Oracle action using Z_deon.
    """
    def __init__(self, critics, oracle, p_ref=1.0, c_cat=-0.5, debug_pcat=False):
        self.critics = critics
        self.oracle = oracle
        self.p_ref = p_ref
        self.c_cat = c_cat
        # Per-compute snapshot — updated by every call to compute()
        self.last_p_cat = 0.0              # sigmoid(logit)  [0, 1]
        self.last_failure_head_logit = 0.0  # raw pre-sigmoid logit (−∞, +∞)
        self.last_failure_head_prob = 0.0   # sigmoid(logit), alias for last_p_cat
        self.last_stakes_value = 0.0        # clip(p_cat / p_ref, 0, 1) — exact value returned
        self.debug_pcat = debug_pcat
        
    def _get_obs_tensor(self, env):
        obs = env.env._get_obs() if hasattr(env, 'env') else env._get_obs()
        obs_np = np.array(obs, dtype=np.float32)
        actor_obs_dim = self.critics.z_free.net[0].weight.shape[1] - 1
        if obs_np.shape[-1] < actor_obs_dim:
            obs_np = np.concatenate([obs_np, np.zeros(actor_obs_dim - obs_np.shape[-1], dtype=np.float32)])
        elif obs_np.shape[-1] > actor_obs_dim:
            obs_np = obs_np[:actor_obs_dim]
        timer = env.get_T_rem()
        T_NORM = 25.0
        t_feat = 1.0 if math.isinf(timer) else np.clip(timer / T_NORM, 0.0, 1.0)
        obs_critic_np = np.concatenate([obs_np, [t_feat]])
        return torch.FloatTensor(obs_critic_np).unsqueeze(0), obs_critic_np, timer, t_feat

    def compute(self, env) -> float:
        # No bomb -> no urgency -> deontic always
        T_rem = env.get_T_rem()
        if math.isinf(T_rem):
            return 0.0

        pos = tuple(env.agent_pos)
        a_d = self.oracle.get_action(env, pos)
        
        # Safety Check: Assert the oracle action is physically safe
        safe_actions = env.get_safe_actions(pos)
        if safe_actions:
            assert a_d in safe_actions, f"Oracle selected unsafe action {a_d} at {pos} not in {safe_actions}"
        
        obs_th, obs_critic_np, timer, t_feat = self._get_obs_tensor(env)
        
        with torch.no_grad():
            # Call failure_predictor directly to get the raw logit BEFORE sigmoid.
            # This lets the debug dump verify: logit → prob → stakes chain.
            raw_logit_tensor = self.critics.failure_predictor(obs_th).squeeze(-1)
            p_cat_tensor = torch.sigmoid(raw_logit_tensor)

        # Store all intermediate values for debug inspection
        self.last_failure_head_logit = float(raw_logit_tensor.item())
        self.last_failure_head_prob = float(p_cat_tensor.item())
        self.last_p_cat = self.last_failure_head_prob  # unchanged alias

        if getattr(self, 'debug_pcat', False):
            print(f"  [DEBUG P_CAT] obs_critic_np shape: {obs_critic_np.shape}, T_rem_used: {timer}, t_feat: {t_feat:.3f}")
            print(f"  [DEBUG P_CAT] Raw obs_critic_np: {np.round(obs_critic_np, 3)}")
            print(f"  [DEBUG P_CAT] logit: {self.last_failure_head_logit:.4f}, prob: {self.last_p_cat:.4f}")

        stakes = float(np.clip(self.last_p_cat / self.p_ref, 0.0, 1.0))
        self.last_stakes_value = stakes
        return stakes

    def __repr__(self):
        return f"ProspectiveStakes(p_ref={self.p_ref}, c_cat={self.c_cat})"


class WrongProspectiveStakes:
    """Ablation stakes: uses Z_free and PPO action selection.
    
    Modes:
      - 'argmax': Evaluates Z_free(s, argmax_a(pi_theta))
      - 'pi_expectation': sum_a pi_theta(a|s) P_cat(s, a)
    """
    def __init__(self, critics, actor, p_ref=1.0, c_cat=-0.5, mode='pi_expectation'):
        self.critics = critics
        self.actor = actor
        self.p_ref = p_ref
        self.c_cat = c_cat
        self.mode = mode
        self.last_p_cat = 0.0
        assert mode in ['argmax', 'pi_expectation']
        
    def _get_obs_tensor(self, env):
        obs = env.env._get_obs() if hasattr(env, 'env') else env._get_obs()
        obs_np = np.array(obs, dtype=np.float32)
        actor_obs_dim = self.critics.z_free.net[0].weight.shape[1] - 1
        if obs_np.shape[-1] < actor_obs_dim:
            obs_np = np.concatenate([obs_np, np.zeros(actor_obs_dim - obs_np.shape[-1], dtype=np.float32)])
        elif obs_np.shape[-1] > actor_obs_dim:
            obs_np = obs_np[:actor_obs_dim]
        timer = env.get_T_rem()
        T_NORM = 25.0
        t_feat = 1.0 if math.isinf(timer) else np.clip(timer / T_NORM, 0.0, 1.0)
        obs_critic_np = np.concatenate([obs_np, [t_feat]])
        return torch.FloatTensor(obs_critic_np).unsqueeze(0), obs_critic_np, timer, t_feat

    def compute(self, env) -> float:
        T_rem = env.get_T_rem()
        if math.isinf(T_rem):
            return 0.0

        obs_actor = env.env._get_obs() if hasattr(env, 'env') else env._get_obs()
        obs_th, obs_critic_np, timer, t_feat = self._get_obs_tensor(env)
        
        if self.mode == 'argmax':
            a_theta = self.actor.get_action(obs_actor, deterministic=True)
            with torch.no_grad():
                quantiles = self.critics.z_free(obs_th)[0, a_theta].numpy()
            sorted_q = np.sort(quantiles)
            p_cat = np.mean(sorted_q <= self.c_cat)
            
        elif self.mode == 'pi_expectation':
            probs = self.actor.get_discrete_probs(obs_actor, n_samples=100)
            with torch.no_grad():
                quantiles_all = self.critics.z_free(obs_th)[0].numpy() # shape (4, N)
            
            p_cat = 0.0
            for a in range(4):
                if probs[a] > 0:
                    sorted_q = np.sort(quantiles_all[a])
                    p_cat_a = np.mean(sorted_q <= self.c_cat)
                    p_cat += probs[a] * p_cat_a
                    
        self.last_p_cat = float(p_cat)
        
        return float(np.clip(p_cat / self.p_ref, 0.0, 1.0))

    def __repr__(self):
        return f"WrongProspectiveStakes(mode={self.mode}, p_ref={self.p_ref}, c_cat={self.c_cat})"
