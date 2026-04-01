"""
ppo_model.py — Shared-backbone Actor-Critic network for PPO.

Architecture overview:
    Shared CNN backbone (identical to DQNModel.conv):
        Conv2d(4→32, BN, ReLU) → Conv2d(32→64, BN, ReLU) → Flatten → (B, 5184)

    Late fusion (same as DQN):
        Concat([spatial_features, scalars]) → (B, 5186)

    Actor head (policy):
        Linear(5186→256) → ReLU → Linear(256→137) → logits
        Legal mask applied before Categorical distribution

    Value head (critic):
        Linear(5186→256) → ReLU → Linear(256→1) → scalar

Why share the backbone?
    The spatial features needed to choose good moves and the features needed to
    estimate board value are the same wall patterns and pawn positions. Sharing
    the CNN avoids learning them twice and halves backbone parameter count.

Why separate FC heads?
    Actor and critic outputs have fundamentally different semantics: actor
    logits are unnormalized log-probabilities, the value head outputs a scalar
    in roughly [-1, 1]. Separate heads let each branch specialize independently.

BC warm-start:
    DQNModel state dict has keys conv.* (backbone) and fc.* (DQN Q-value head).
    PPOModel has conv.*, actor_head.*, value_head.*. To warm-start from BC:

        bc = torch.load("checkpoints/bc_pretrained.pt", map_location=device)
        backbone = {k: v for k, v in bc.items() if k.startswith("conv.")}
        model.load_state_dict(backbone, strict=False)

    strict=False silently skips actor_head.* and value_head.* (random init).
    The DQN fc.* keys are filtered out before loading so no warnings are raised.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import (
    BOARD_SIZE,
    CONV1_FILTERS,
    CONV2_FILTERS,
    CONV_KERNEL_SIZE,
    FC_HIDDEN_SIZE,
    NUM_ACTIONS,
    NUM_CHANNELS,
    NUM_SCALARS,
)

# Mirror the private constants from dqn_model.py so both models stay in sync.
_CONV_OUT_SIZE: int   = CONV2_FILTERS * BOARD_SIZE * BOARD_SIZE  # 64 * 9 * 9 = 5184
_FC_IN_SIZE: int      = _CONV_OUT_SIZE + NUM_SCALARS              # 5184 + 2  = 5186
_ILLEGAL_LOGIT: float = -1e9   # applied to illegal actions before Categorical


class PPOModel(nn.Module):
    """
    Shared-backbone Actor-Critic for Quoridor PPO.

    The CNN backbone is byte-for-byte identical to DQNModel.conv, enabling
    direct weight transfer from BC-pretrained checkpoints. The actor and value
    heads branch from the same fused (B, 5186) representation.
    """

    def __init__(self) -> None:
        super().__init__()

        # Shared CNN backbone — identical architecture to DQNModel.conv.
        # Keeping the key names identical ("conv") means BC checkpoint weights
        # can be loaded with a simple startswith("conv.") filter.
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, CONV1_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(CONV1_FILTERS),
            nn.ReLU(),
            # (B, 32, 9, 9)

            nn.Conv2d(CONV1_FILTERS, CONV2_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(CONV2_FILTERS),
            nn.ReLU(),
            # (B, 64, 9, 9)

            nn.Flatten(),
            # (B, 5184)
        )

        # Actor head: maps fused features → action logits.
        # Masked before Categorical so illegal actions never get sampled.
        self.actor_head = nn.Sequential(
            nn.Linear(_FC_IN_SIZE, FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, NUM_ACTIONS),
            # (B, 137) raw logits — masking applied in forward()
        )

        # Value head: maps fused features → scalar board value estimate.
        # Separate from actor so value loss doesn't interfere with policy gradient.
        self.value_head = nn.Sequential(
            nn.Linear(_FC_IN_SIZE, FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, 1),
            # (B, 1)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _encode(
        self,
        spatial: torch.Tensor,  # (B, 4, 9, 9)
        scalars: torch.Tensor,  # (B, 2)
    ) -> torch.Tensor:
        """
        Shared CNN + late-fusion step. Returns fused representation.

        Returns:
            fused: (B, 5186)
        """
        spatial_features = self.conv(spatial)                         # (B, 5184)
        return torch.cat([spatial_features, scalars], dim=1)          # (B, 5186)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        spatial: torch.Tensor,    # (B, 4, 9, 9)
        scalars: torch.Tensor,    # (B, 2)
        legal_mask: torch.Tensor, # (B, 137) bool
    ) -> tuple[Categorical, torch.Tensor]:
        """
        Full actor-critic forward pass.

        Used during rollout collection (to sample actions + get log_probs + values)
        and during the PPO update (to recompute log_probs + entropy + values under
        the current, possibly updated, policy weights).

        The legal mask is applied to actor logits before constructing the
        Categorical distribution. PyTorch's Categorical applies softmax internally,
        so masking here (setting illegal logits to -1e9) is correct — the illegal
        actions get probability ~0 and can never be sampled.

        Args:
            spatial:    Board state, shape (B, 4, 9, 9).
            scalars:    Wall-count features, shape (B, 2).
            legal_mask: True where action is legal, shape (B, 137).

        Returns:
            dist:  Categorical distribution over legal actions.
                   Use dist.sample() for action selection,
                   dist.log_prob(a) for log π(a|s),
                   dist.entropy() for the entropy bonus.
            value: Scalar value estimate, shape (B, 1).
        """
        fused = self._encode(spatial, scalars)                        # (B, 5186)

        # Actor: mask illegal actions before creating the distribution.
        raw_logits    = self.actor_head(fused)                        # (B, 137)
        masked_logits = raw_logits.masked_fill(~legal_mask, _ILLEGAL_LOGIT)
        dist          = Categorical(logits=masked_logits)

        # Critic: scalar value estimate (no mask needed — not selecting actions).
        value = self.value_head(fused)                                # (B, 1)

        return dist, value

    def get_value(
        self,
        spatial: torch.Tensor,  # (B, 4, 9, 9)
        scalars: torch.Tensor,  # (B, 2)
    ) -> torch.Tensor:
        """
        Value head only — used to bootstrap V(s_{T+1}) at the end of a rollout.

        Called once per rollout on the observation following the last collected
        step. If the rollout ended on a terminal state, the caller passes 0
        directly instead of calling this method.

        No legal_mask needed: we are estimating state value, not selecting an action.

        Returns:
            value: shape (B, 1)
        """
        fused = self._encode(spatial, scalars)                        # (B, 5186)
        return self.value_head(fused)                                 # (B, 1)
