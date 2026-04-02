"""
CNN Q-network for Quoridor DQN agent.

Input: (B, 4, 9, 9) spatial + (B, 2) scalars + (B, 137) legal mask
Output: (B, 137) masked Q-values (illegal actions set to -1e9)
"""

import torch
import torch.nn as nn

from config import (
    BOARD_SIZE, NUM_ACTIONS, NUM_CHANNELS, NUM_SCALARS,
    CONV1_FILTERS, CONV2_FILTERS, CONV_KERNEL_SIZE, FC_HIDDEN_SIZE,
)

_CONV_OUT_SIZE = CONV2_FILTERS * BOARD_SIZE * BOARD_SIZE  # 64 * 9 * 9 = 5184
_FC_IN_SIZE = _CONV_OUT_SIZE + NUM_SCALARS                # 5186
_ILLEGAL_ACTION_PENALTY = -1e9


class DQNModel(nn.Module):
    """CNN encoder with late scalar fusion -> Q-values over all 137 actions."""

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, CONV1_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(CONV1_FILTERS),
            nn.ReLU(),
            nn.Conv2d(CONV1_FILTERS, CONV2_FILTERS, kernel_size=CONV_KERNEL_SIZE, padding=1),
            nn.BatchNorm2d(CONV2_FILTERS),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.Linear(_FC_IN_SIZE, FC_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(FC_HIDDEN_SIZE, NUM_ACTIONS),
        )

    def forward(self, spatial, scalars, legal_mask):
        """Compute masked Q-values. Illegal actions get -1e9."""
        spatial_features = self.conv(spatial)
        fused = torch.cat([spatial_features, scalars], dim=1)
        q_values = self.fc(fused)
        q_values = q_values.masked_fill(~legal_mask, _ILLEGAL_ACTION_PENALTY)
        return q_values
