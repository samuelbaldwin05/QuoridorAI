"""
ppo_model_bfs_resnet.py — PPO Actor-Critic: BFS channels + ResNet backbone.

This is the full-combo variant. It combines:
    - 6-channel BFS input (see ppo_model_bfs.py for motivation)
    - ResNet backbone with skip connections (see ppo_model_resnet.py for motivation)
    - LayerNorm throughout

Architecture:
    Initial projection:
        Conv2d(6 → 64, 3×3, pad=1) → LayerNorm(64, 9, 9) → ReLU
    Residual blocks (NUM_RESIDUAL_BLOCKS = 2 by default):
        x → Conv(64→64) → LN → ReLU → Conv(64→64) → LN → ReLU(x + F(x))
    Flatten:
        (B, 64, 9, 9) → (B, 5184)
    Late fusion:
        cat([5184, scalars=2]) → (B, 5186)
    Actor head:
        Linear(5186→256) → ReLU → Linear(256→137) → masked logits → Categorical
    Value head:
        Linear(5186→256) → ReLU → Linear(256→1)

Usage:
    Must be paired with QuoridorEnv(use_bfs=True).
    train_ppo.py's build_model_and_env() handles this automatically when
    --model bfs_resnet is passed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import (
    BOARD_SIZE,
    NUM_ACTIONS,
    NUM_CHANNELS_BFS,
    NUM_SCALARS,
    CONV2_FILTERS,
    FC_HIDDEN_SIZE,
    NUM_RESIDUAL_BLOCKS,
)

_CONV_OUT_SIZE: int   = CONV2_FILTERS * BOARD_SIZE * BOARD_SIZE  # 64 * 9 * 9 = 5184
_FC_IN_SIZE: int      = _CONV_OUT_SIZE + NUM_SCALARS              # 5184 + 2  = 5186
_ILLEGAL_LOGIT: float = -1e9


class ResidualBlock(nn.Module):
    """
    Residual block with LayerNorm. Identical to the one in ppo_model_resnet.py.

    Defined here so each model file is self-contained — importing across agent
    files creates fragile coupling that breaks if files are renamed or moved.

    F(x) = Conv → LN → ReLU → Conv → LN
    output = ReLU(x + F(x))
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.ln1   = nn.LayerNorm([channels, BOARD_SIZE, BOARD_SIZE])
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.ln2   = nn.LayerNorm([channels, BOARD_SIZE, BOARD_SIZE])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        return F.relu(out + residual)


class PPOModelBFSResNet(nn.Module):
    """
    Shared-backbone Actor-Critic: 6-channel BFS input + ResNet encoder.

    Drop-in replacement for PPOModel when used with QuoridorEnv(use_bfs=True).
    The public API (forward, get_value) is identical to all other PPO model variants.
    """

    def __init__(self) -> None:
        super().__init__()

        # Initial projection: 6 BFS channels → 64 feature maps.
        self.proj = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS_BFS, CONV2_FILTERS, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm([CONV2_FILTERS, BOARD_SIZE, BOARD_SIZE]),
            nn.ReLU(),
        )

        # Residual tower: NUM_RESIDUAL_BLOCKS blocks at constant channel width.
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(CONV2_FILTERS) for _ in range(NUM_RESIDUAL_BLOCKS)]
        )

        self.flatten = nn.Flatten()  # (B, 64, 9, 9) → (B, 5184)

        # Actor head.
        self.actor_head = nn.Sequential(
            nn.Linear(_FC_IN_SIZE, FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, NUM_ACTIONS),
        )

        # Value head.
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
        """Projection + residual blocks + late fusion. Returns (B, 5186)."""
        x = self.proj(spatial)                                        # (B, 64, 9, 9)
        x = self.res_blocks(x)                                        # (B, 64, 9, 9)
        x = self.flatten(x)                                           # (B, 5184)
        return torch.cat([x, scalars], dim=1)                         # (B, 5186)

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
