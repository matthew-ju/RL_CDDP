"""
CDDPAgent — Orchestrator implementing Definition 3 (hard mode switching).

Per-step runtime flow:
    1. S(s) = stakes_fn.compute(env)
    2. if S(s) < τ:  DEONTIC mode
         safe_actions ← env.get_safe_actions()
         if actor's preferred action is safe → use it
         else → DeonticActionSelector picks the best safe action (deterministic)
    3. if S(s) ≥ τ:  UTILITARIAN mode
         discrete_action ← actor.get_action(obs)     # no shield

Shield + selector integration:
    ActionShield determines the LEGAL set of safe actions.
    DeonticActionSelector picks the BEST action within that set,
    using lexicographic ranking: L_safe(next) → D_min(next) → fixed priority.

Debug dump:
    When debug_switching_dump=True, per-step records are accumulated in
    _debug_episode_steps and flushed to JSON via flush_debug_episode().
    The dump contains all switching-relevant signals needed to diagnose
    wiring bugs (A), definition mismatches (B), and learning failures (C).
"""

import os
import json
import math
import numpy as np
from cddp.agents.ppo_actor import discretize_action, DISCRETE_TO_CONTINUOUS
from cddp.agents.deontic_selector import DeonticActionSelector
from cddp.eval.failure_band import is_failure_band


class CDDPAgent:
    """CDDP orchestrator: shield + stakes + mode switching.

    Wraps an actor (PPOActor), a shield (ActionShield), a stakes
    function, and an optional DeonticOraclePlanner for exact bounds.

    Usage:
        stakes_fn = build_stakes(config, critics, oracle, actor)
        agent = CDDPAgent(actor, shield, stakes_fn, tau=0.5, oracle=oracle)
        action, info = agent.select_action(obs, env)
    """

    def __init__(self, actor, shield, stakes_fn, tau: float = 0.3,
                 deontic_selector=None, debug_pcat=False,
                 oscillation_fix=None, tau_lower=0.3,
                 debug_switching_dump: bool = False,
                 debug_outdir: str = 'results/debug_switching'):
        """
        Args:
            actor: PPOActor instance
            shield: ActionShield instance
            stakes_fn: Constructed stakes function object
            tau: Mode-switching threshold (tau_up for hysteresis)
            deontic_selector: Legacy Fallback Selector
            debug_pcat: If True, print debug info for p_cat_hat at episode start.
            oscillation_fix: None | 'sticky' | 'hysteresis'
            tau_lower: Lower threshold for hysteresis (tau_down)
            debug_switching_dump: If True, accumulate per-step debug records.
            debug_outdir: Root directory for debug JSON output files.
        """
        self.actor = actor
        self.shield = shield
        self.stakes_fn = stakes_fn
        self.tau = tau
        self.tau_lower = tau_lower
        self.deontic_selector = deontic_selector or DeonticActionSelector()
        self._prev_mode = None  # Track for switch detection
        self.debug_pcat = debug_pcat
        self.episode_steps = 0

        # Oscillation Fixes (Phase 4.3)
        self._committed_util = False
        self.oscillation_fix = oscillation_fix
        self._is_utilitarian_sticky = False
        self._current_phase = 'deontic'  # for hysteresis
        
        self._max_s_score = 0.0
        self._last_t_rem = None

        # ── Debug switching dump ──────────────────────────────────────
        self.debug_switching_dump = debug_switching_dump
        self.debug_outdir = debug_outdir
        self._debug_episode_steps: list = []    # accumulates per-step records
        self._debug_ep_meta: dict = {}           # filled by flush_debug_episode()
        # Per-episode accumulators (reset in reset_mode_tracking)
        self._debug_first_failure_step_metric: int | None = None
        self._debug_first_utilitarian_step: int | None = None
        self._debug_entered_fb_env: bool = False
        self._debug_entered_fb_metric: bool = False
        self._debug_switched_ever: bool = False

    @property
    def grid_id(self):
        return getattr(self.actor, 'grid_id', None)

    @property
    def checkpoint_path(self):
        return getattr(self.actor, 'checkpoint_path', None)

    def compute_stakes(self, env) -> float:
        """Compute stakes for the current state."""
        return self.stakes_fn.compute(env)

    def get_mode(self, stakes: float) -> str:
        """Rule: 'deontic' if S < tau, 'utilitarian' if S >= tau."""
        return 'deontic' if stakes < self.tau else 'utilitarian'

    def select_action(self, obs: np.ndarray, env,
                      deterministic: bool = False) -> tuple:
        """Select action using CDDP mode switching.

        Args:
            obs: Current observation (numpy array, 7-dim)
            env: InstrumentedHostageEnv (for stakes + shield queries)
            deterministic: If True, use policy mean; else sample

        Returns:
            action: int (discrete: 0-3)
            info: dict with keys:
                'stakes': float
                'mode': str — 'deontic' or 'utilitarian'
                'safe_actions': list[int] or None
                'action_was_shielded': bool
                'switch_triggered': bool — True if mode changed from previous step
                'ranked_safe_actions': list or None (deontic mode only)
                'deontic_tiebreak_reason': str or None
        """
        # Capture prev_mode BEFORE computing stakes (avoid double-call contamination)
        prev_mode_snapshot = self._prev_mode

        stakes = self.compute_stakes(env)
        
        T_remaining = env.get_T_rem()
        if getattr(self, '_last_t_rem', None) is None:
            self._last_t_rem = T_remaining
            
        self._max_s_score = max(getattr(self, '_max_s_score', 0.0), stakes)
        
        L_safe_est = env.get_L_safe(tuple(env.agent_pos))
        
        budget_margin = T_remaining - L_safe_est
        is_pure_deontic = hasattr(self.stakes_fn, 'fixed_val') and self.stakes_fn.fixed_val == 0.0
        force_util = (budget_margin <= 1.5) and not is_pure_deontic

        if self._committed_util or force_util or stakes >= self.tau:
            self._committed_util = True
            mode_determined = 'utilitarian'
        else:
            mode_determined = 'deontic'

        # print(f"[CRITIC DEBUG] T_remaining={T_remaining}, L_safe_est={L_safe_est:.2f}, S_score={stakes:.4f}, tau={self.tau}, mode={'UTIL' if mode_determined == 'utilitarian' else 'DEON'}")

        # Update _prev_mode only after mode_determined is final
        switch_triggered = (self._prev_mode is not None and mode_determined != self._prev_mode)
        self._prev_mode = mode_determined

        # ── EXACT stakes value used in the mode decision ──────────────
        # This is the variable that flows into pred_up / pred_down

        stakes_value_used_for_switch = stakes
        pred_up = stakes_value_used_for_switch >= self.tau
        # pred_down threshold depends on oscillation_fix mode:
        #   hysteresis → tau_lower   (explicit lower band)
        #   sticky     → never       (once up, stay up forever)
        #   default    → tau_up      (switch down when stakes < tau)
        if self.oscillation_fix == 'hysteresis':
            pred_down = stakes_value_used_for_switch <= self.tau_lower
        elif self.oscillation_fix == 'sticky':
            pred_down = False  # sticky never switches down
        else:
            pred_down = stakes_value_used_for_switch < self.tau  # default: same boundary


        if mode_determined == 'utilitarian':
            mode = 'utilitarian'
            action = self.actor.get_action(obs, deterministic=deterministic)
        else:
            mode = 'deontic'
            # DeonticActionSelector ONLY — no PPO actor involvement whatsoever
            action = self.deontic_selector.select_action(env)

        p_cat_hat = getattr(self.stakes_fn, 'last_p_cat', 0.0)

        if self.debug_pcat and self.episode_steps == 0:
            print(f"\n[DEBUG P_CAT] start-state. T_rem = {env.get_T_rem()}, stakes = {stakes:.3f}, p_cat_hat = {p_cat_hat:.3f}")

        info = {
            'stakes': stakes,
            'p_cat_hat': p_cat_hat,
            'mode': mode,
            'safe_actions': None,
            'action_was_shielded': mode == 'deontic',
            'switch_triggered': switch_triggered,
            'ranked_safe_actions': None,
            'deontic_tiebreak_reason': None,
        }

        # ── Debug dump collection ──────────────────────────────────────
        if self.debug_switching_dump:
            self._collect_debug_step(
                env=env,
                step=self.episode_steps,
                stakes_value_used_for_switch=stakes_value_used_for_switch,
                pred_up=pred_up,
                pred_down=pred_down,
                mode=mode,
                prev_mode=prev_mode_snapshot,
            )

        self.episode_steps += 1
        return action, info

    # ── Debug helpers ──────────────────────────────────────────────────

    def _collect_debug_step(self, env, step: int,
                            stakes_value_used_for_switch: float,
                            pred_up: bool, pred_down: bool,
                            mode: str, prev_mode):
        """Collect one per-step debug record.  Called only when debug_switching_dump=True."""
        # ── Pull env quantities ──────────────────────────────────────
        pos = tuple(env.agent_pos)
        D_min = env.get_D_min(pos)
        T_rem = env.get_T_rem()
        L_safe = env.get_L_safe(pos)
        margin = int(T_rem - D_min) if not math.isinf(T_rem) else 999

        # ── Three distinct failure-band signals ──────────────────────
        # (a) env: what the env info dict contains (set in _cddp_info via is_failure_band)
        in_failure_band_env = env.is_in_failure_band(pos)
        # (b) metric_recomputed: apply the predicate fresh to dumped scalars (post-hoc, identical function)
        in_failure_band_metric_recomputed = is_failure_band(D_min, T_rem, L_safe)
        # (c) metric_logged: what metrics.py currently reads from info dict for t*
        #     We simulate this by reading from env info — metrics.py does info.get('in_failure_band')
        #     which is set by _cddp_info → same as in_failure_band_env here, but captured separately
        #     so the analysis can diff them per-step if the info dict ever lags or differs.
        _cddp_info = env._cddp_info()
        in_failure_band_metric_logged = bool(_cddp_info.get('in_failure_band', False))

        # ── Switch events (hysteresis-aware) ────────────────────────
        switch_up_event = (prev_mode == 'deontic' and mode == 'utilitarian')
        switch_down_event = (prev_mode == 'utilitarian' and mode == 'deontic')

        # ── Stakes fn internals ──────────────────────────────────────
        failure_head_logit = getattr(self.stakes_fn, 'last_failure_head_logit', float('nan'))
        failure_head_prob = getattr(self.stakes_fn, 'last_failure_head_prob', float('nan'))
        p_cat_hat_logged = getattr(self.stakes_fn, 'last_p_cat', float('nan'))

        # ── Assertions (hard failures → immediate crash) ─────────────
        if not math.isnan(failure_head_prob):
            assert 0.0 <= failure_head_prob <= 1.0, (
                f"[DEBUG ASSERT] failure_head_prob={failure_head_prob:.6f} out of [0,1] "
                f"at step={step}, pos={pos}, T_rem={T_rem}"
            )
        if not math.isnan(p_cat_hat_logged):
            assert 0.0 <= p_cat_hat_logged <= 1.0, (
                f"[DEBUG ASSERT] p_cat_hat_logged={p_cat_hat_logged:.6f} out of [0,1] "
                f"at step={step}, pos={pos}, T_rem={T_rem}"
            )

        # ── Accumulate episode trackers ──────────────────────────────
        if in_failure_band_metric_recomputed and self._debug_first_failure_step_metric is None:
            self._debug_first_failure_step_metric = step
            self._debug_entered_fb_metric = True
        if in_failure_band_env:
            self._debug_entered_fb_env = True
        if mode == 'utilitarian':
            if self._debug_first_utilitarian_step is None:
                self._debug_first_utilitarian_step = step
            self._debug_switched_ever = True

        # ── Build record ─────────────────────────────────────────────
        T_rem_serializable = T_rem if not math.isinf(T_rem) else None
        L_safe_serializable = L_safe if not math.isinf(L_safe) else None

        record = {
            # Identity
            'step': step,
            'pos_x': int(pos[0]),
            'pos_y': int(pos[1]),
            'T_rem': T_rem_serializable,
            # Path quantities
            'D_min': int(D_min),
            'min_v': None,  # filled by sweep script per episode
            'margin': margin,
            'slip_buffer': 0,   # base env has no explicit slip buffer variable
            'L_safe': L_safe_serializable,
            # Failure band — three independent signals
            'in_failure_band_env': bool(in_failure_band_env),
            'in_failure_band_metric_recomputed': bool(in_failure_band_metric_recomputed),
            'in_failure_band_metric_logged': bool(in_failure_band_metric_logged),
            # Predictor internals
            'failure_head_logit': failure_head_logit if not math.isnan(failure_head_logit) else None,
            'failure_head_prob': failure_head_prob if not math.isnan(failure_head_prob) else None,
            'p_cat_hat_logged': p_cat_hat_logged if not math.isnan(p_cat_hat_logged) else None,
            # Switching decision
            'stakes_value_used_for_switch': float(stakes_value_used_for_switch),
            'tau_up': float(self.tau),
            'tau_down': float(self.tau_lower),
            'pred_up': bool(pred_up),
            'pred_down': bool(pred_down),
            'mode': mode,
            'switch_up_event': bool(switch_up_event),
            'switch_down_event': bool(switch_down_event),
        }
        self._debug_episode_steps.append(record)

    def flush_debug_episode(self, episode_id: int, seed: int, grid_id: int,
                             T_init: int, min_v=None,
                             outcome: str = 'unknown',
                             violations: int = 0,
                             excess_violations: int = 0,
                             feasible: bool = True):
        """Write accumulated episode steps to JSON and reset accumulators.

        File path: {debug_outdir}/grid{grid_id}/T{T_init}/seed{seed}/ep{episode_id}.json

        Args:
            episode_id: Global episode counter.
            seed: RNG seed for this run.
            grid_id: Layout ID (1 or 2).
            T_init: Bomb timer at episode start.
            min_v: Minimum violations needed from start state (from min_violations_needed).
            outcome: 'rescued' | 'timeout' | 'bomb' | 'other'
            violations: Total civilian violations this episode.
            excess_violations: Violations above min_v.
            feasible: Whether episode passed feasibility filter.
        """
        # Fill in min_v for all steps
        for rec in self._debug_episode_steps:
            if rec['min_v'] is None:
                rec['min_v'] = min_v

        out = {
            'episode_id': episode_id,
            'seed': seed,
            'grid_id': grid_id,
            'T_init': T_init,
            # Per-episode summary
            'entered_failure_band_ever_env': bool(self._debug_entered_fb_env),
            'entered_failure_band_ever_metric': bool(self._debug_entered_fb_metric),
            'switched_ever': bool(self._debug_switched_ever),
            'first_failure_band_step_metric': self._debug_first_failure_step_metric,
            'first_utilitarian_step': self._debug_first_utilitarian_step,
            'outcome': outcome,
            'violations': int(violations),
            'excess_violations': int(excess_violations),
            'feasible': bool(feasible),
            # Step-level data
            'steps': self._debug_episode_steps,
        }

        out_dir = os.path.join(
            self.debug_outdir,
            f'grid{grid_id}',
            f'T{T_init}',
            f'seed{seed}',
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'ep{episode_id}.json')
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2)

        # Reset accumulators
        self._debug_episode_steps = []
        self._debug_first_failure_step_metric = None
        self._debug_first_utilitarian_step = None
        self._debug_entered_fb_env = False
        self._debug_entered_fb_metric = False
        self._debug_switched_ever = False

    # ── Mode tracking ──────────────────────────────────────────────────

    def reset_mode_tracking(self):
        """Reset previous-mode tracker (call at episode boundaries)."""
        self._prev_mode = None
        self.episode_steps = 0
        self._committed_util = False
        self._is_utilitarian_sticky = False
        self._current_phase = 'deontic'
        # Reset debug accumulators
        self._debug_episode_steps = []
        self._debug_first_failure_step_metric = None
        self._debug_first_utilitarian_step = None
        self._debug_entered_fb_env = False
        self._debug_entered_fb_metric = False
        self._debug_switched_ever = False
        # Reset stateful stakes (e.g. AnalyticalOracleStakes latch)
        if hasattr(self.stakes_fn, 'reset'):
            self.stakes_fn.reset()


    def select_deontic_action(self, obs: np.ndarray, env,
                              deterministic: bool = False) -> tuple:
        """Select action in forced deontic mode (ignoring stakes).
        Used for deontic warmup rollouts.
        """
        action = self.deontic_selector.select_action(env)

        info = {
            'stakes': 0.0,
            'mode': 'deontic',
            'safe_actions': None,
            'action_was_shielded': True,
            'switch_triggered': False,
            'ranked_safe_actions': None,
            'deontic_tiebreak_reason': None,
        }
        return action, info


    # Legacy method kept for backward compatibility
    def _pick_best_safe_action(self, obs: np.ndarray,
                                safe_actions: list) -> int:
        """DEPRECATED: Use deontic_selector.select_safe_action() instead.

        Legacy random MC fallback — kept only for backward compatibility
        with training code that hasn't been updated yet.
        """
        probs = self.actor.get_discrete_probs(obs, n_samples=500)
        masked_probs = np.zeros(4)
        for a in safe_actions:
            masked_probs[a] = probs[a]

        total = masked_probs.sum()
        if total > 0:
            masked_probs /= total
            return np.random.choice(4, p=masked_probs)
        else:
            return np.random.choice(safe_actions)

    def __repr__(self):
        return (f"CDDPAgent(tau={self.tau}, "
                f"stakes_fn={self.stakes_fn.__class__.__name__})")
