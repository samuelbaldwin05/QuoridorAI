"""
dqn_model.py — Neural network for the DQN agent.

Architecture: CNN spatial encoder + late fusion of scalar features → Q-values.

Why CNN? Quoridor state is naturally spatial (9x9 board). CNNs exploit
translational structure in wall patterns and pawn positions, which fully-
connected layers would have to re-learn from scratch.

Why late fusion? Wall counts (scalars) are global features with no spatial
meaning. Fusing them after the CNN flatten avoids polluting the spatial
feature maps with scalar information.

Input:
    spatial : (B, 4, 9, 9)  — 4 board channels (see config.NUM_CHANNELS)
    scalars : (B, 2)         — [my_walls_left/10, opp_walls_left/10]
    legal_mask : (B, 137)    — True where action is legal

Output:
    q_values : (B, 137)      — masked Q-values; illegal actions set to -1e9
"""

import torch
import torch.nn as nn

from config import (
    BOARD_SIZE,
    NUM_ACTIONS,
    NUM_CHANNELS,
    NUM_SCALARS,
    CONV1_FILTERS,
    CONV2_FILTERS,
    CONV_KERNEL_SIZE,
    FC_HIDDEN_SIZE,
)

# Spatial size is preserved through both conv layers because padding=1 with
# kernel=3 leaves height and width unchanged: (9 - 3 + 2*1) / 1 + 1 = 9
_CONV_OUT_SIZE = CONV2_FILTERS * BOARD_SIZE * BOARD_SIZE  # 64 * 9 * 9 = 5184
_FC_IN_SIZE = _CONV_OUT_SIZE + NUM_SCALARS                # 5184 + 2 = 5186

# Q-values for illegal actions are set to this before argmax/softmax so they
# are never selected. Must be large enough to be below any real Q-value.
_ILLEGAL_ACTION_PENALTY = -1e9


class DQNModel(nn.Module):
    """
    CNN Q-network for Quoridor.

    The network maps a (spatial, scalars) observation to a vector of
    Q-values over all 137 actions. Illegal actions are masked out before
    the values are returned so the caller never needs to handle them.
    """

    def __init__(self) -> None:
        super().__init__()

        # --- Spatial encoder ---
        # Two conv layers with BatchNorm. BatchNorm stabilizes training by
        # normalizing activations within each mini-batch, which is especially
        # helpful early on when Q-value targets are noisy.
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

        # --- Late-fusion head ---
        # Scalars are concatenated onto the flattened spatial features, then
        # passed through two fully-connected layers to produce Q-values.
        self.fc = nn.Sequential(
            nn.Linear(_FC_IN_SIZE, FC_HIDDEN_SIZE),
            nn.ReLU(),
            # (B, 256)

            nn.Linear(FC_HIDDEN_SIZE, NUM_ACTIONS),
            # (B, 137) — raw Q-values before masking
        )

    def forward(
        self,
        spatial: torch.Tensor,    # (B, 4, 9, 9)
        scalars: torch.Tensor,    # (B, 2)
        legal_mask: torch.Tensor, # (B, 137) bool
    ) -> torch.Tensor:
        """
        Compute masked Q-values for a batch of observations.

        Illegal actions are set to _ILLEGAL_ACTION_PENALTY so that any
        downstream argmax or softmax will never select them.

        Args:
            spatial:    Board state tensor, shape (B, 4, 9, 9).
            scalars:    Wall-count features, shape (B, 2).
            legal_mask: Boolean mask, True where action is legal, shape (B, 137).

        Returns:
            q_values: Masked Q-value tensor, shape (B, 137).
        """
        spatial_features = self.conv(spatial)           # (B, 5184)
        fused = torch.cat([spatial_features, scalars], dim=1)  # (B, 5186)
        q_values = self.fc(fused)                       # (B, 137)

        # Zero out illegal actions so they are never selected.
        # We use masked_fill rather than direct indexing to stay
        # differentiable and avoid in-place modification of the graph.
        q_values = q_values.masked_fill(~legal_mask, _ILLEGAL_ACTION_PENALTY)

        return q_values  # (B, 137)
