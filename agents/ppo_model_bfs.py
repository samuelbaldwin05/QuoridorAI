"""
ppo_model_bfs.py — PPO Actor-Critic with 6-channel BFS input (no residual).

Differences from ppo_model.py (baseline):
    1. 6 input channels instead of 4. The two extra channels are BFS distance
       maps computed by QuoridorEnv(use_bfs=True):
           ch4 : current player's normalised BFS distance map
           ch5 : opponent's normalised BFS distance map
       Each cell (r, c) holds the minimum steps to reach the player's goal row
       from that position given current walls, normalised to [0, 1].

       Why this helps: the 4-channel baseline gives the model no direct signal
       about path lengths. It must infer from wall positions alone whether a
       wall is strategically valuable. BFS channels make path obstruction
       immediately visible — the model can directly see "this wall increased
       the opponent's distance" in the next observation.

    2. LayerNorm instead of BatchNorm (same reasoning as ppo_model_resnet.py):
       PPO's train/eval mode switching makes BatchNorm's running-stat behaviour
       inconsistent early in training. LayerNorm is invariant to mode.

Architecture (identical to baseline except channels and norm):
    Conv2d(6 → 32, 3×3, pad=1) → LayerNorm(32, 9, 9) → ReLU
    Conv2d(32 → 64, 3×3, pad=1) → LayerNorm(64, 9, 9) → ReLU
    Flatten: (B, 64, 9, 9) → (B, 5184)
    Late fusion: cat([5184, scalars=2]) → (B, 5186)
    Actor head: Linear(5186→256) → ReLU → Linear(256→137)
    Value head: Linear(5186→256) → ReLU → Linear(256→1)

Usage:
    Must be paired with QuoridorEnv(use_bfs=True). The env returns (6, 9, 9)
    spatial observations; passing them to a 4-channel model would crash.
    train_ppo.py's build_model_and_env() handles this pairing automatically.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from config import (
    BOARD_SIZE,
    NUM_ACTIONS,
    NUM_CHANNELS_BFS,
    NUM_SCALARS,
    CONV1_FILTERS,
    CONV2_FILTERS,
    CONV_KERNEL_SIZE,
    FC_HIDDEN_SIZE,
)

_CONV_OUT_SIZE: int   = CONV2_FILTERS * BOARD_SIZE * BOARD_SIZE  # 64 * 9 * 9 = 5184
_FC_IN_SIZE: int      = _CONV_OUT_SIZE + NUM_SCALARS              # 5184 + 2  = 5186
_ILLEGAL_LOGIT: float = -1e9


class PPOModelBFS(nn.Module):
    """
    Shared-backbone Actor-Critic for Quoridor PPO with BFS distance channels.

    Drop-in replacement for PPOModel when used with QuoridorEnv(use_bfs=True).
    The public API (forward, get_value) is identical to PPOModel.
    """

    def __init__(self) -> None:
        super().__init__()

        # Shared CNN backbone — identical to baseline except:
        #   - First Conv2d takes NUM_CHANNELS_BFS (6) input channels
        #   - BatchNorm2d replaced by LayerNorm for train/eval consistency
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS_BFS, CONV1_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding=1, bias=False),
            nn.LayerNorm([CONV1_FILTERS, BOARD_SIZE, BOARD_SIZE]),
            nn.ReLU(),
            # (B, 32, 9, 9)

            nn.Conv2d(CONV1_FILTERS, CONV2_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding=1, bias=False),
            nn.LayerNorm([CONV2_FILTERS, BOARD_SIZE, BOARD_SIZE]),
            nn.ReLU(),
            # (B, 64, 9, 9)

            nn.Flatten(),
            # (B, 5184)
        )

        # Actor head: maps fused features → masked action distribution.
        self.actor_head = nn.Sequential(
            nn.Linear(_FC_IN_SIZE, FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, NUM_ACTIONS),
        )

        # Value head: maps fused features → scalar board value.
        self.value_head = nn.Sequential(
            nn.Linear(_FC_IN_SIZE, FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, 1),
        )

    def _encode(
        self,
        spatial: torch.Tensor,  # (B, 6, 9, 9)
        scalars: torch.Tensor,  # (B, 2)
    ) -> torch.Tensor:
        """Shared CNN + late fusion. Returns (B, 5186)."""
        spatial_features = self.conv(spatial)                         # (B, 5184)
        return torch.cat([spatial_features, scalars], dim=1)          # (B, 5186)

    def forward(
        self,
        spatial: torch.Tensor,    # (B, 6, 9, 9)
        scalars: torch.Tensor,    # (B, 2)
        legal_mask: torch.Tensor, # (B, 137) bool
    ) -> tuple[Categorical, torch.Tensor]:
        """
        Full actor-critic forward pass. Identical interface to PPOModel.forward().

        Returns:
            dist:  Categorical over legal actions.
            value: Scalar value estimate, shape (B, 1).
        """
        fused = self._encode(spatial, scalars)                        # (B, 5186)

        raw_logits    = self.actor_head(fused)                        # (B, 137)
        masked_logits = raw_logits.masked_fill(~legal_mask, _ILLEGAL_LOGIT)
        dist          = Categorical(logits=masked_logits)

        value = self.value_head(fused)                                # (B, 1)
        return dist, value

    def get_value(
        self,
        spatial: torch.Tensor,  # (B, 6, 9, 9)
        scalars: torch.Tensor,  # (B, 2)
    ) -> torch.Tensor:
        """Value head only — used to bootstrap V(s_{T+1}) at rollout end."""
        fused = self._encode(spatial, scalars)
        return self.value_head(fused)                                 # (B, 1)
