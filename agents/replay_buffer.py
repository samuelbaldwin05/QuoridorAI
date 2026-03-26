"""
Experience replay buffer for DQN training.

Preallocated numpy circular buffer. Stores transitions and samples
random mini-batches for training. Also stores next_legal_mask since
Double DQN needs it for action selection in the target computation.
"""

import numpy as np

from config import NUM_ACTIONS, BOARD_SIZE, NUM_CHANNELS, NUM_SCALARS, REPLAY_BUFFER_SIZE


class ReplayBuffer:
    """Fixed-capacity circular buffer for DQN transitions."""

    def __init__(self):
        cap = REPLAY_BUFFER_SIZE
        self._spatial      = np.zeros((cap, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self._scalars      = np.zeros((cap, NUM_SCALARS), dtype=np.float32)
        self._actions      = np.zeros((cap,), dtype=np.int64)
        self._rewards      = np.zeros((cap,), dtype=np.float32)
        self._next_spatial = np.zeros((cap, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self._next_scalars = np.zeros((cap, NUM_SCALARS), dtype=np.float32)
        self._dones        = np.zeros((cap,), dtype=bool)
        self._next_legal_mask = np.zeros((cap, NUM_ACTIONS), dtype=bool)

        self._ptr  = 0
        self._size = 0

    def push(self, transition: dict):
        """Store a single transition. Overwrites oldest when full."""
        i = self._ptr
        self._spatial[i]         = transition["spatial"]
        self._scalars[i]         = transition["scalars"]
        self._actions[i]         = transition["action"]
        self._rewards[i]         = transition["reward"]
        self._next_spatial[i]    = transition["next_spatial"]
        self._next_scalars[i]    = transition["next_scalars"]
        self._dones[i]           = transition["done"]
        self._next_legal_mask[i] = transition["next_legal_mask"]

        self._ptr  = (self._ptr + 1) % REPLAY_BUFFER_SIZE
        self._size = min(self._size + 1, REPLAY_BUFFER_SIZE)

    def sample(self, batch_size: int) -> dict:
        """Sample a random mini-batch. Raises ValueError if not enough data."""
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} from buffer of size {self._size}"
            )
        indices = np.random.choice(self._size, size=batch_size, replace=False)
        return {
            "spatial":         self._spatial[indices],
            "scalars":         self._scalars[indices],
            "actions":         self._actions[indices],
            "rewards":         self._rewards[indices],
            "next_spatial":    self._next_spatial[indices],
            "next_scalars":    self._next_scalars[indices],
            "dones":           self._dones[indices],
            "next_legal_mask": self._next_legal_mask[indices],
        }

    def __len__(self):
        return self._size
