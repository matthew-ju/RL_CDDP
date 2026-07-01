"""
ActionShield — Hard action masking for deontic mode.

Restricts π_θ to A_safe(s) by setting unsafe action logits to -inf.
Implements the shield from Definition 3 of the formalization.
"""

import torch
import torch.nn.functional as F


class ActionShield:
    """Hard action masking: restricts policy to safe actions.

    In deontic mode (S(s) < τ), the shield filters out any action
    that would move the agent onto a civilian tile.

    Usage:
        shield = ActionShield()
        safe_actions = env.get_safe_actions()
        masked_logits = shield.apply_shield(logits, safe_actions)
    """

    def get_safe_actions(self, env) -> list:
        """Get safe actions from the instrumented environment.

        Args:
            env: InstrumentedHostageEnv instance

        Returns:
            List of safe action indices (subset of [0, 1, 2, 3])
        """
        return env.get_safe_actions()

    def apply_shield(self, logits: torch.Tensor, safe_actions: list) -> torch.Tensor:
        """Mask unsafe actions to -inf in logits.

        Args:
            logits: Raw action logits from π_θ, shape (num_actions,) or (batch, num_actions)
            safe_actions: List of safe action indices

        Returns:
            masked_logits: Same shape, unsafe actions set to -inf

        Raises:
            ValueError: If safe_actions is empty (Shield Feasibility B1 violated)
        """
        if len(safe_actions) == 0:
            raise ValueError(
                "Shield Feasibility violated: no safe actions available. "
                "This means Assumption B1 is broken for this state."
            )

        # Create mask: True for safe, False for unsafe
        if logits.dim() == 1:
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for a in safe_actions:
                mask[a] = True
            masked = logits.clone()
            masked[~mask] = float('-inf')
        else:
            # Batched: same safe_actions for all batch elements
            mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
            for a in safe_actions:
                mask[a] = True
            masked = logits.clone()
            masked[:, ~mask] = float('-inf')

        return masked

    def is_feasible(self, env) -> bool:
        """Check if shield feasibility (B1) holds: at least one safe action exists."""
        return len(self.get_safe_actions(env)) > 0
