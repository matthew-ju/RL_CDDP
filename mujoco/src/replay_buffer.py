"""
replay_buffer.py
================
Replay buffers for DDPG-family algorithms.

  ReplayBuffer       — standard uniform replay (DDPG, CDDP utilitarian)
  DeonticReplayBuffer — stores strict-policy rollouts with episode-outcome
                        labels (y_cat ∈ {0, 1}) for Z_deon BCE training
"""

from __future__ import annotations

import torch


class ReplayBuffer:
    """Circular buffer for (s, a, r, s', done) tuples."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int,
                 device: torch.device) -> None:
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.device = device

        self.s      = torch.zeros((capacity, state_dim),  dtype=torch.float32, device=device)
        self.a      = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.r      = torch.zeros((capacity, 1),          dtype=torch.float32, device=device)
        self.s_next = torch.zeros((capacity, state_dim),  dtype=torch.float32, device=device)
        self.done   = torch.zeros((capacity, 1),          dtype=torch.float32, device=device)

    # ------------------------------------------------------------------
    def push(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        r: float | torch.Tensor,
        s_next: torch.Tensor,
        done: bool | torch.Tensor,
    ) -> None:
        idx = self.ptr
        self.s[idx]      = s.detach().squeeze()
        self.a[idx]      = a.detach().squeeze()
        self.r[idx]      = float(r)
        self.s_next[idx] = s_next.detach().squeeze()
        self.done[idx]   = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        s_next: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        n = s.shape[0]
        if n == 0:
            return
        idx = torch.arange(n, device=self.device)
        idx = (self.ptr + idx) % self.capacity
        self.s[idx]      = s.detach().float()
        self.a[idx]      = a.detach().float()
        self.r[idx]      = r.detach().float().reshape(-1, 1)
        self.s_next[idx] = s_next.detach().float()
        self.done[idx]   = done.detach().float().reshape(-1, 1)
        self.ptr  = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch: int) -> tuple[torch.Tensor, ...]:
        idx = torch.randint(self.size, (batch,), device=self.device)
        return (
            self.s[idx], self.a[idx], self.r[idx],
            self.s_next[idx], self.done[idx],
        )

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------

class DeonticReplayBuffer:
    """Buffer for Z_deon training.

    Each transition carries y_cat (bool: did the episode end in catastrophe?)
    collected under STRICT (S=0) policy execution.  Labels are attached
    retroactively after episode termination.
    """

    def __init__(self, capacity: int, state_dim: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.device = device

        self.s     = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.y     = torch.zeros((capacity, 1),         dtype=torch.float32, device=device)
        self.valid = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.n_valid = 0

        # Pending slot for unlabeled current episode
        self._pending_start: int = 0
        self._pending_end: int = 0

    def start_episode(self) -> None:
        """Mark the start of a new strict episode."""
        self._pending_start = self.ptr
        self._pending_end = self.ptr

    def push_state(self, s: torch.Tensor) -> None:
        """Add one state from the current strict-policy rollout."""
        idx = self.ptr
        self.s[idx] = s.detach().squeeze().float()
        self.valid[idx] = False
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self._pending_end = self.ptr

    def label_episode(self, catastrophe: bool) -> None:
        """Attach the episode outcome label to all pending states."""
        label = float(catastrophe)
        start = self._pending_start
        end = self._pending_end
        if start <= end:
            indices = torch.arange(start, end, device=self.device) % self.capacity
        else:
            # Wrapped around
            first  = torch.arange(start, self.capacity, device=self.device)
            second = torch.arange(0, end, device=self.device)
            indices = torch.cat([first, second])
        self.y[indices] = label
        newly_valid = (~self.valid[indices]).sum().item()
        self.valid[indices] = True
        self.n_valid = min(self.size, self.n_valid + int(newly_valid))

    def sample(self, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
        valid_idx = torch.where(self.valid[: self.size])[0]
        idx = valid_idx[torch.randint(valid_idx.shape[0], (batch,), device=self.device)]
        return self.s[idx], self.y[idx]

    def __len__(self) -> int:
        return self.n_valid
