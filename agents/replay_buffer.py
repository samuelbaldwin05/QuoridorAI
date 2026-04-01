"""
replay_buffer.py — Experience replay buffers for the DQN agent.

Contains two implementations:
    ReplayBuffer              — uniform random sampling (original)
    PrioritizedReplayBuffer   — prioritized sampling via SumTree (PER)

Why experience replay? The DQN agent generates transitions sequentially during
gameplay. Consecutive transitions are highly correlated (the board barely
changes step-to-step), and training directly on correlated data causes the
network to overfit to recent experience and forget earlier patterns. By storing
transitions in a buffer and sampling random mini-batches, we break that
correlation and give gradient descent the diverse, i.i.d.-like updates it needs.

Why PER? With sparse rewards (+1/-1 only at terminal steps), the vast majority
of transitions have reward 0 and low TD error once the network stabilises. Uniform
sampling buries rare high-value transitions (e.g. a wall placement that caused a
win) in thousands of routine pawn-move transitions. PER samples high-TD-error
transitions more frequently, so the network trains harder on the experiences it
is most wrong about — exactly the wall-placement wins we need to amplify.

SumTree: a binary tree where each leaf stores a priority and each internal node
stores the sum of its subtree. Supports O(log N) priority updates and O(log N)
weighted sampling, versus O(N) for a naive sorted array.
"""

import numpy as np

from config import (
    BOARD_SIZE,
    NUM_ACTIONS,
    NUM_CHANNELS,
    NUM_SCALARS,
    PER_ALPHA,
    PER_EPSILON,
    REPLAY_BUFFER_SIZE,
)


# ---------------------------------------------------------------------------
# Sum-tree
# ---------------------------------------------------------------------------

class SumTree:
    """
    Binary tree where leaves hold priorities and internal nodes hold subtree sums.

    Array layout (size = 2*capacity - 1, 0-indexed):
        Internal nodes : indices 0 .. capacity-2
        Leaves         : indices capacity-1 .. 2*capacity-2
        Root           : index 0
        Left child of i  : 2i + 1
        Right child of i : 2i + 2
        Parent of i      : (i - 1) // 2

    Leaf k (0-indexed data position) lives at tree index k + capacity - 1.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write_ptr = 0   # circular write head into the leaf array
        self.n_entries = 0   # number of valid leaves written so far

    @property
    def total(self) -> float:
        """Sum of all leaf priorities (stored at root)."""
        return float(self.tree[0])

    def _propagate(self, tree_idx: int, change: float) -> None:
        """Walk up the tree from tree_idx, adding change to every ancestor."""
        parent = (tree_idx - 1) // 2
        self.tree[parent] += change
        if parent > 0:
            self._propagate(parent, change)

    def update(self, leaf_idx: int, priority: float) -> None:
        """
        Set the priority for leaf at data position leaf_idx ∈ [0, capacity).
        Propagates the delta up to the root.
        """
        tree_idx = leaf_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        if tree_idx > 0:
            self._propagate(tree_idx, change)

    def add(self, priority: float) -> int:
        """
        Write a new leaf with the given priority at the current write position.
        Returns the data index (leaf position) that was written.
        Advances the circular write pointer and updates n_entries.
        """
        leaf_idx = self.write_ptr
        self.update(leaf_idx, priority)
        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return leaf_idx

    def sample(self, s: float) -> tuple[int, float]:
        """
        Traverse the tree to find the leaf whose cumulative prefix sum first
        reaches s. s must be in [0, total).

        Returns (leaf_idx, priority) where leaf_idx ∈ [0, capacity).
        """
        idx = 0  # start at root
        while idx < self.capacity - 1:  # while not a leaf
            left  = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        leaf_idx = idx - (self.capacity - 1)
        return leaf_idx, float(self.tree[idx])


# ---------------------------------------------------------------------------
# Uniform replay buffer (original — kept for reference and non-PER runs)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Fixed-capacity circular buffer storing DQN transitions.
    Samples uniformly at random — no prioritization.

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
        self._spatial         = np.zeros((REPLAY_BUFFER_SIZE, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self._scalars         = np.zeros((REPLAY_BUFFER_SIZE, NUM_SCALARS),  dtype=np.float32)
        self._actions         = np.zeros((REPLAY_BUFFER_SIZE,),              dtype=np.int64)
        self._rewards         = np.zeros((REPLAY_BUFFER_SIZE,),              dtype=np.float32)
        self._next_spatial    = np.zeros((REPLAY_BUFFER_SIZE, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self._next_scalars    = np.zeros((REPLAY_BUFFER_SIZE, NUM_SCALARS),  dtype=np.float32)
        self._dones           = np.zeros((REPLAY_BUFFER_SIZE,),              dtype=bool)
        self._next_legal_mask = np.zeros((REPLAY_BUFFER_SIZE, NUM_ACTIONS),  dtype=bool)

        self._ptr  = 0
        self._size = 0

    def push(self, transition: dict) -> None:
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
        if self._size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} from buffer of size {self._size}.")
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

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# Prioritized replay buffer (PER)
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """
    Experience replay buffer with prioritized sampling (Schaul et al., 2015).

    Transitions are sampled with probability proportional to |TD error|^alpha.
    New transitions receive the current maximum priority so they are sampled
    at least once before their actual TD error is known.

    Importance-sampling (IS) weights correct the bias introduced by non-uniform
    sampling. Beta is annealed from PER_BETA_START → 1.0 over training; at
    beta=1.0 the correction is exact, at beta=0 it is ignored entirely.

    Interface differences from ReplayBuffer:
        sample(batch_size, beta)  — returns extra keys "weights" and "indices"
        update_priorities(indices, td_errors)  — call after every training step
    """

    def __init__(self) -> None:
        self._tree         = SumTree(REPLAY_BUFFER_SIZE)
        self._max_priority = 1.0  # new transitions always get max priority

        # Same data layout as ReplayBuffer
        self._spatial         = np.zeros((REPLAY_BUFFER_SIZE, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self._scalars         = np.zeros((REPLAY_BUFFER_SIZE, NUM_SCALARS),  dtype=np.float32)
        self._actions         = np.zeros((REPLAY_BUFFER_SIZE,),              dtype=np.int64)
        self._rewards         = np.zeros((REPLAY_BUFFER_SIZE,),              dtype=np.float32)
        self._next_spatial    = np.zeros((REPLAY_BUFFER_SIZE, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self._next_scalars    = np.zeros((REPLAY_BUFFER_SIZE, NUM_SCALARS),  dtype=np.float32)
        self._dones           = np.zeros((REPLAY_BUFFER_SIZE,),              dtype=bool)
        self._next_legal_mask = np.zeros((REPLAY_BUFFER_SIZE, NUM_ACTIONS),  dtype=bool)

    def push(self, transition: dict) -> None:
        """
        Store a transition with the current maximum priority.

        Assigning max priority ensures new transitions are sampled at least once
        before their actual TD error is computed and used to set a true priority.
        """
        idx = self._tree.write_ptr  # data slot to write into

        self._spatial[idx]         = transition["spatial"]
        self._scalars[idx]         = transition["scalars"]
        self._actions[idx]         = transition["action"]
        self._rewards[idx]         = transition["reward"]
        self._next_spatial[idx]    = transition["next_spatial"]
        self._next_scalars[idx]    = transition["next_scalars"]
        self._dones[idx]           = transition["done"]
        self._next_legal_mask[idx] = transition["next_legal_mask"]

        self._tree.add(self._max_priority)

    def sample(self, batch_size: int, beta: float) -> dict:
        """
        Sample a prioritized mini-batch using stratified sampling.

        Stratified sampling: divide [0, total) into batch_size equal segments
        and draw one sample per segment. This reduces variance compared to
        purely random sampling from the priority distribution.

        Args:
            batch_size : number of transitions to sample
            beta       : IS correction exponent in [0, 1]; 1.0 = full correction

        Returns:
            dict with all transition arrays plus:
                "weights" : (B,) float32 — IS correction weights, max-normalised
                "indices" : list[int]    — leaf indices for update_priorities()
        """
        if len(self) < batch_size:
            raise ValueError(f"Cannot sample {batch_size} from buffer of size {len(self)}.")

        indices   = []
        priorities = []
        segment   = self._tree.total / batch_size

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            leaf_idx, priority = self._tree.sample(s)
            indices.append(leaf_idx)
            priorities.append(priority)

        # IS weights: w_i = (N * P(i))^(-beta), normalised by max weight
        # so the largest weight is always 1.0 (keeps gradients scaled properly)
        priorities_arr = np.array(priorities, dtype=np.float32)
        n              = len(self)
        probs          = priorities_arr / self._tree.total
        weights        = (n * probs) ** (-beta)
        weights       /= weights.max()  # normalise

        idx = np.array(indices)
        return {
            "spatial":         self._spatial[idx],
            "scalars":         self._scalars[idx],
            "actions":         self._actions[idx],
            "rewards":         self._rewards[idx],
            "next_spatial":    self._next_spatial[idx],
            "next_scalars":    self._next_scalars[idx],
            "dones":           self._dones[idx],
            "next_legal_mask": self._next_legal_mask[idx],
            "weights":         weights,   # (B,) float32
            "indices":         indices,   # list[int] — needed for update_priorities
        }

    def update_priorities(self, indices: list[int], td_errors: np.ndarray) -> None:
        """
        Recompute and store priorities after a training step.

        priority = (|TD error| + PER_EPSILON)^PER_ALPHA

        PER_EPSILON ensures every transition has non-zero probability.
        PER_ALPHA controls how much prioritization is applied.

        Also tracks _max_priority so new pushes always receive the current
        highest priority.

        Args:
            indices   : leaf indices returned by the last sample() call
            td_errors : per-sample |Q(s,a) - target| as a NumPy array, shape (B,)
        """
        for leaf_idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + PER_EPSILON) ** PER_ALPHA
            self._tree.update(leaf_idx, priority)
            self._max_priority = max(self._max_priority, priority)

    def __len__(self) -> int:
        return self._tree.n_entries
