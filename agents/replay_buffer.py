"""
replay_buffer.py — Experience replay buffer for the DQN agent.

Why experience replay? The DQN agent generates transitions sequentially during
gameplay. Consecutive transitions are highly correlated (the board barely
changes step-to-step), and training directly on correlated data causes the
network to overfit to recent experience and forget earlier patterns. By storing
transitions in a buffer and sampling random mini-batches, we break that
correlation and give gradient descent the diverse, i.i.d.-like updates it needs.

Implementation: preallocated NumPy arrays in a circular (ring) buffer.
Allocating upfront avoids repeated memory allocation and makes random-index
sampling fast (a single np.random.choice + array slicing, no Python loops).
When the buffer is full, the oldest transitions are silently overwritten.

Why store next_legal_mask? During the TD target computation, we need to find
the best legal action in next_state. That requires the legal mask for the
*next* observation, which is only available at the time the transition is
collected — so it must be stored here alongside the transition.
"""

import numpy as np

from config import NUM_ACTIONS, BOARD_SIZE, NUM_CHANNELS, NUM_SCALARS, REPLAY_BUFFER_SIZE


class ReplayBuffer:
    """
    Fixed-capacity circular buffer storing DQN transitions.

    Each transition contains:
        state_spatial   : (4, 9, 9)  float32  — board channels for current state
        state_scalars   : (2,)       float32  — wall counts for current state
        action          : int                 — action index taken (0–136)
        reward          : float               — reward received (+1, -1, or 0)
        next_spatial    : (4, 9, 9)  float32  — board channels for next state
        next_scalars    : (2,)       float32  — wall counts for next state
        done            : bool                — True if episode ended
        next_legal_mask : (137,)     bool     — legal actions in next state
    """

    def __init__(self) -> None:
        # Preallocate all arrays at full capacity upfront.
        # Using float32 throughout to match PyTorch's default dtype.
        self._spatial      = np.zeros((REPLAY_BUFFER_SIZE, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self._scalars      = np.zeros((REPLAY_BUFFER_SIZE, NUM_SCALARS),  dtype=np.float32)
        self._actions      = np.zeros((REPLAY_BUFFER_SIZE,),              dtype=np.int64)
        self._rewards      = np.zeros((REPLAY_BUFFER_SIZE,),              dtype=np.float32)
        self._next_spatial = np.zeros((REPLAY_BUFFER_SIZE, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self._next_scalars = np.zeros((REPLAY_BUFFER_SIZE, NUM_SCALARS),  dtype=np.float32)
        self._dones        = np.zeros((REPLAY_BUFFER_SIZE,),              dtype=bool)
        self._next_legal_mask = np.zeros((REPLAY_BUFFER_SIZE, NUM_ACTIONS), dtype=bool)

        self._ptr  = 0  # write head: index of the next slot to overwrite
        self._size = 0  # number of valid transitions currently stored

    def push(self, transition: dict) -> None:
        """
        Store a single transition in the buffer.

        Args:
            transition: dict with keys:
                'spatial'         : np.ndarray (4, 9, 9)
                'scalars'         : np.ndarray (2,)
                'action'          : int
                'reward'          : float
                'next_spatial'    : np.ndarray (4, 9, 9)
                'next_scalars'    : np.ndarray (2,)
                'done'            : bool
                'next_legal_mask' : np.ndarray (137,) bool
        """
        i = self._ptr

        self._spatial[i]         = transition["spatial"]
        self._scalars[i]         = transition["scalars"]
        self._actions[i]         = transition["action"]
        self._rewards[i]         = transition["reward"]
        self._next_spatial[i]    = transition["next_spatial"]
        self._next_scalars[i]    = transition["next_scalars"]
        self._dones[i]           = transition["done"]
        self._next_legal_mask[i] = transition["next_legal_mask"]

        # Advance write head, wrapping around when the buffer is full
        self._ptr  = (self._ptr + 1) % REPLAY_BUFFER_SIZE
        self._size = min(self._size + 1, REPLAY_BUFFER_SIZE)

    def sample(self, batch_size: int) -> dict:
        """
        Sample a random mini-batch of transitions.

        Returns a dict of NumPy arrays; the training loop is responsible for
        converting these to PyTorch tensors on the appropriate device.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            dict with keys matching push() inputs, each value shaped (B, ...).

        Raises:
            ValueError: If the buffer contains fewer transitions than batch_size.
        """
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer of size {self._size}."
            )

        # Sample without replacement so no transition appears twice in a batch
        indices = np.random.choice(self._size, size=batch_size, replace=False)

        return {
            "spatial":         self._spatial[indices],         # (B, 4, 9, 9)
            "scalars":         self._scalars[indices],         # (B, 2)
            "actions":         self._actions[indices],         # (B,)
            "rewards":         self._rewards[indices],         # (B,)
            "next_spatial":    self._next_spatial[indices],    # (B, 4, 9, 9)
            "next_scalars":    self._next_scalars[indices],    # (B, 2)
            "dones":           self._dones[indices],           # (B,)
            "next_legal_mask": self._next_legal_mask[indices], # (B, 137)
        }

    def __len__(self) -> int:
        """Return the number of transitions currently stored."""
        return self._size
