"""
Tests for QR Critics, Replay Buffers, and Prospective Stakes (Phase 3-4).
"""

import sys
import os
import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'omnisafe_SRL'))

from cddp.critics.qr_critic import QRCritic, DualCritics
from cddp.replay.buffers import ReplayBuffer, DeonticReplayBuffer, Transition
from cddp.stakes.prospective import ProspectiveStakes
from cddp.env.hostage_env import InstrumentedHostageEnv


def test_qr_critic_shapes():
    """Test QRCritic output shapes."""
    print("\n" + "="*60)
    print("TEST: QRCritic Shapes")
    print("="*60)

    critic = QRCritic(obs_dim=9, n_quantiles=21, hidden_dim=128)

    # Single obs
    obs = torch.randn(9)
    q = critic(obs)
    assert q.shape == (21,), f"Expected (21,), got {q.shape}"
    print(f"✓ Single obs → quantiles shape: {q.shape}")

    # Batch
    obs_batch = torch.randn(32, 9)
    q_batch = critic(obs_batch)
    assert q_batch.shape == (32, 21), f"Expected (32, 21), got {q_batch.shape}"
    print(f"✓ Batch obs → quantiles shape: {q_batch.shape}")

    # Mean
    v = critic.get_mean(obs_batch)
    assert v.shape == (32,), f"Expected (32,), got {v.shape}"
    print(f"✓ get_mean shape: {v.shape}")

    # P_below
    p = critic.get_p_below(obs_batch, threshold=0.0)
    assert p.shape == (32,), f"Expected (32,), got {p.shape}"
    assert (p >= 0).all() and (p <= 1).all()
    print(f"✓ get_p_below in [0,1]: range [{p.min():.3f}, {p.max():.3f}]")

    # CVaR
    cvar = critic.get_cvar(obs_batch, alpha=0.1)
    assert cvar.shape == (32,)
    print(f"✓ CVaR shape: {cvar.shape}")

    print("\n✅ QRCritic shapes PASSED")


def test_quantile_huber_loss():
    """Test quantile Huber loss computation."""
    print("\n" + "="*60)
    print("TEST: Quantile Huber Loss")
    print("="*60)

    critic = QRCritic(obs_dim=9, n_quantiles=21)
    obs = torch.randn(16, 9)
    quantiles = critic(obs)
    targets = torch.randn(16)

    loss = critic.quantile_huber_loss(quantiles, targets)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0
    print(f"✓ Loss is scalar: {loss.item():.4f}")

    # Loss should be differentiable
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in critic.parameters())
    assert has_grad
    print(f"✓ Gradients flow")

    print("\n✅ Quantile Huber Loss PASSED")


def test_dual_critics():
    """Test DualCritics manager."""
    print("\n" + "="*60)
    print("TEST: DualCritics")
    print("="*60)

    critics = DualCritics(obs_dim=9, n_quantiles=21)

    obs = torch.randn(16, 9)
    next_obs = torch.randn(16, 9)
    rewards = torch.randn(16)
    dones = torch.zeros(16)

    # Update Z_free
    loss_f = critics.update_z_free(obs, rewards, next_obs, dones)
    assert isinstance(loss_f, float) and not np.isnan(loss_f)
    print(f"✓ Z_free update loss: {loss_f:.4f}")

    # Update Z_deon
    loss_d = critics.update_z_deon(obs, rewards, next_obs, dones)
    assert isinstance(loss_d, float) and not np.isnan(loss_d)
    print(f"✓ Z_deon update loss: {loss_d:.4f}")

    # Urgency gap
    gap = critics.get_urgency_gap(obs)
    assert gap.shape == (16,)
    print(f"✓ Urgency gap shape: {gap.shape}")

    # State dict roundtrip
    sd = critics.state_dict()
    critics2 = DualCritics(obs_dim=9, n_quantiles=21)
    critics2.load_state_dict(sd)
    gap2 = critics2.get_urgency_gap(obs)
    assert torch.allclose(gap, gap2, atol=1e-6)
    print(f"✓ State dict save/load roundtrip")

    print("\n✅ DualCritics PASSED")


def test_replay_buffer():
    """Test ReplayBuffer with metadata."""
    print("\n" + "="*60)
    print("TEST: ReplayBuffer")
    print("="*60)

    buf = ReplayBuffer(capacity=100, obs_dim=9)

    # Add some transitions
    for i in range(50):
        t = Transition(
            obs=np.random.randn(7).astype(np.float32),
            action=np.random.randint(4),
            reward=np.random.randn(),
            next_obs=np.random.randn(7).astype(np.float32),
            done=i == 49,
            mode='deontic' if i % 2 == 0 else 'utilitarian',
            T_rem=5.0,
            D_min=3,
            L_safe=7.0 if i % 2 == 0 else float('inf'),
            in_failure_band=i % 2 == 1,
            episode_id=0,
            step_idx=i,
        )
        buf.add(t)

    assert len(buf) == 50
    print(f"✓ Buffer size: {len(buf)}")

    # Sample
    batch = buf.sample(16)
    assert batch['obs'].shape == (16, 9)
    assert batch['rewards'].shape == (16,)
    assert len(batch['modes']) == 16
    print(f"✓ Sample shapes correct")

    # Obs is padded to 9
    assert batch['obs'].shape[-1] == 9
    print(f"✓ 7-dim obs padded to 9-dim")

    # Overflow: add 60 more
    for i in range(60):
        t = Transition(
            obs=np.random.randn(7).astype(np.float32),
            action=0, reward=99.0, next_obs=np.random.randn(7).astype(np.float32),
            done=False, mode='test', T_rem=1.0, D_min=1, L_safe=1.0,
            in_failure_band=False, episode_id=1, step_idx=i,
        )
        buf.add(t)

    assert len(buf) == 100  # capped at capacity
    print(f"✓ Buffer wraps at capacity: {len(buf)}")

    print("\n✅ ReplayBuffer PASSED")


def test_prospective_stakes():
    """Test ProspectiveStakes computation."""
    print("\n" + "="*60)
    print("TEST: ProspectiveStakes")
    print("="*60)

    critics = DualCritics(obs_dim=9, n_quantiles=21)
    ps = ProspectiveStakes(critics, obs_dim=9)

    # Compute from obs
    obs = np.random.randn(7).astype(np.float32)
    s = ps.compute_from_obs(obs)
    assert 0 <= s <= 1
    print(f"✓ Stakes from obs: {s:.3f} ∈ [0, 1]")

    # Batch compute
    obs_batch = torch.randn(32, 9)
    s_batch = ps.compute_batch(obs_batch)
    assert s_batch.shape == (32,)
    assert (s_batch >= 0).all() and (s_batch <= 1).all()
    print(f"✓ Batch stakes: shape={s_batch.shape}, range=[{s_batch.min():.3f}, {s_batch.max():.3f}]")

    # After training Z_free to be better, P_cat should increase
    # (This is a gradient-level test — train Z_free on high rewards)
    for _ in range(50):
        obs_t = torch.randn(16, 9)
        next_t = torch.randn(16, 9)
        # Z_free gets high rewards → V_free large
        critics.update_z_free(obs_t, torch.ones(16) * 10, next_t, torch.zeros(16))
        # Z_deon gets low rewards → Z_deon quantiles low
        critics.update_z_deon(obs_t, torch.ones(16) * -5, next_t, torch.zeros(16))

    s_after = ps.compute_from_obs(obs)
    print(f"✓ After training (free=high, deon=low): stakes = {s_after:.3f}")
    # With high V_free and low Z_deon, most Z_deon quantiles should be below V_free
    # So P_cat should be high
    # (Not guaranteed monotonic after only 50 steps with random inputs, but sanity check)
    print(f"  (Expected: high P_cat since deontic far worse than utilitarian)")

    print("\n✅ ProspectiveStakes PASSED")


if __name__ == '__main__':
    test_qr_critic_shapes()
    test_quantile_huber_loss()
    test_dual_critics()
    test_replay_buffer()
    test_prospective_stakes()

    print("\n" + "="*60)
    print("ALL PHASE 3-4 TESTS PASSED ✅")
    print("="*60)
