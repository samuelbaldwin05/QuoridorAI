"""
ppo_model_resnet.py — PPO Actor-Critic with a ResNet backbone (4 input channels).

Differences from ppo_model.py (baseline):
    1. ResNet backbone: the CNN uses residual blocks instead of plain conv layers.
       Skip connections let gradients flow directly to early layers, making it
       practical to stack more computation without vanishing gradients.
    2. LayerNorm instead of BatchNorm: PPO alternates between model.eval() during
       rollout collection and model.train() during gradient updates. BatchNorm
       uses different statistics in each mode (batch stats vs running stats),
       which introduces noise early in training when running stats are poor.
       LayerNorm normalises per-sample over all channels+spatial dims, so
       behaviour is identical at train and eval time.

Architecture:
    Initial projection:
        Conv2d(4 → 64, 3×3, pad=1) → LayerNorm(64, 9, 9) → ReLU
    Residual blocks (NUM_RESIDUAL_BLOCKS = 2):
        x → Conv(64→64, 3×3) → LN → ReLU → Conv(64→64, 3×3) → LN → ReLU(x + F(x))
    Flatten:
        (B, 64, 9, 9) → (B, 5184)
    Late fusion (same as baseline):
        cat([spatial_features, scalars]) → (B, 5186)
    Actor head:
        Linear(5186→256) → ReLU → Linear(256→137) → masked logits → Categorical
    Value head:
        Linear(5186→256) → ReLU → Linear(256→1)

Why the same _CONV_OUT_SIZE as the baseline?
    64 filters × 9 × 9 spatial = 5184, identical to the baseline because we
    keep the same filter count and spatial dimensions throughout. The ResNet adds
    depth without changing the representation size at the flatten point.

BC warm-start:
    Not directly compatible (key names differ from baseline conv.* structure),
    but the actor/value heads are identical and could be initialised from a
    matching ResNet checkpoint with strict=False.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from config import (
    BOARD_SIZE,
    NUM_ACTIONS,
    NUM_CHANNELS,
    NUM_SCALARS,
    FC_HIDDEN_SIZE,
    CONV2_FILTERS,
    NUM_RESIDUAL_BLOCKS,
)

# After the initial projection and all residual blocks, every spatial position
# is described by CONV2_FILTERS = 64 features. Flattening gives 64 × 9 × 9.
_CONV_OUT_SIZE: int   = CONV2_FILTERS * BOARD_SIZE * BOARD_SIZE  # 64 * 9 * 9 = 5184
_FC_IN_SIZE: int      = _CONV_OUT_SIZE + NUM_SCALARS              # 5184 + 2  = 5186
_ILLEGAL_LOGIT: float = -1e9


class ResidualBlock(nn.Module):
    """
    One residual block: two conv layers with LayerNorm, skip connection.

    The skip connection adds the block's input directly to its output before
    the final ReLU. This means the block only needs to learn the *residual*
    (the difference from the identity), which is an easier optimisation target
    than learning the full transformation from scratch.

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
        # ReLU after the addition so the combined signal is non-negative.
        # This is the original ResNet formulation (He et al., 2016).
        return F.relu(out + residual)


class PPOModelResNet(nn.Module):
    """
    Shared-backbone Actor-Critic for Quoridor PPO with a ResNet CNN encoder.

    Drop-in replacement for PPOModel. The public API (forward, get_value) is
    identical, so train_ppo.py needs no changes beyond instantiating this class.
    """

    def __init__(self) -> None:
        super().__init__()

        # Initial projection: map input channels → CONV2_FILTERS (64).
        # Bias=False because LayerNorm has its own learnable affine parameters.
        self.proj = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, CONV2_FILTERS, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm([CONV2_FILTERS, BOARD_SIZE, BOARD_SIZE]),
            nn.ReLU(),
        )

        # Stack of residual blocks. All blocks operate at CONV2_FILTERS channels
        # so the spatial dimensions and channel count stay constant throughout.
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(CONV2_FILTERS) for _ in range(NUM_RESIDUAL_BLOCKS)]
        )

        self.flatten = nn.Flatten()  # (B, 64, 9, 9) → (B, 5184)

        # Actor head: fused features → masked action distribution.
        self.actor_head = nn.Sequential(
            nn.Linear(_FC_IN_SIZE, FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, NUM_ACTIONS),
        )

        # Value head: fused features → scalar board value.
        self.value_head = nn.Sequential(
            nn.Linear(_FC_IN_SIZE, FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, 1),
        )

    def _encode(
        self,
        spatial: torch.Tensor,  # (B, 4, 9, 9)
        scalars: torch.Tensor,  # (B, 2)
    ) -> torch.Tensor:
        """Shared projection + residual blocks + late fusion. Returns (B, 5186)."""
        x = self.proj(spatial)          # (B, 64, 9, 9)
        x = self.res_blocks(x)          # (B, 64, 9, 9)
        x = self.flatten(x)             # (B, 5184)
        return torch.cat([x, scalars], dim=1)  # (B, 5186)

    def forward(
        self,
        spatial: torch.Tensor,    # (B, 4, 9, 9)
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
        spatial: torch.Tensor,  # (B, 4, 9, 9)
        scalars: torch.Tensor,  # (B, 2)
    ) -> torch.Tensor:
        """Value head only — used to bootstrap V(s_{T+1}) at rollout end."""
        fused = self._encode(spatial, scalars)
        return self.value_head(fused)                                 # (B, 1)
