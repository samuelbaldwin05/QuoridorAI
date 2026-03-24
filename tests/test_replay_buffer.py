"""
Tests for agents/replay_buffer.py.

Covers:
  - Initial state (empty buffer)
  - push() increments length correctly up to capacity
  - Circular overwrite: length caps at REPLAY_BUFFER_SIZE, oldest data is gone
  - sample() output shapes match expected dimensions
  - sample() raises ValueError when buffer has fewer transitions than batch_size
  - Data integrity: a pushed transition can be recovered exactly (single-item buffer)
  - sample() does not return duplicate transitions in one batch (replace=False)
"""

import numpy as np
import pytest

from agents.replay_buffer import ReplayBuffer
from config import NUM_ACTIONS, NUM_CHANNELS, BOARD_SIZE, NUM_SCALARS, REPLAY_BUFFER_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transition(seed: int = 0) -> dict:
    """Return a deterministic transition dict for a given seed."""
    rng = np.random.default_rng(seed)
    return {
        "spatial":         rng.random((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
        "scalars":         rng.random((NUM_SCALARS,), dtype=np.float32),
        "action":          int(rng.integers(0, NUM_ACTIONS)),
        "reward":          float(rng.choice([-1.0, 0.0, 1.0])),
        "next_spatial":    rng.random((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
        "next_scalars":    rng.random((NUM_SCALARS,), dtype=np.float32),
        "done":            bool(rng.integers(0, 2)),
        "next_legal_mask": rng.integers(0, 2, size=(NUM_ACTIONS,), dtype=bool),
    }


def _fill_buffer(buf: ReplayBuffer, n: int) -> None:
    """Push n distinct transitions into buf."""
    for i in range(n):
        buf.push(_make_transition(seed=i))


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def test_initial_length_is_zero():
    """A freshly created buffer must report length 0."""
    buf = ReplayBuffer()
    assert len(buf) == 0


# ---------------------------------------------------------------------------
# push() and length
# ---------------------------------------------------------------------------

def test_push_increments_length():
    """Length grows by 1 for each push, up to REPLAY_BUFFER_SIZE."""
    buf = ReplayBuffer()
    for i in range(1, 6):
        buf.push(_make_transition(seed=i))
        assert len(buf) == i


def test_length_caps_at_capacity():
    """Length must not exceed REPLAY_BUFFER_SIZE even after excess pushes."""
    buf = ReplayBuffer()
    _fill_buffer(buf, REPLAY_BUFFER_SIZE + 10)
    assert len(buf) == REPLAY_BUFFER_SIZE


# ---------------------------------------------------------------------------
# Circular overwrite
# ---------------------------------------------------------------------------

def test_circular_overwrite_does_not_crash():
    """Pushing beyond capacity must not raise — the buffer wraps silently."""
    buf = ReplayBuffer()
    # Push twice the capacity — should not raise
    _fill_buffer(buf, REPLAY_BUFFER_SIZE * 2)
    assert len(buf) == REPLAY_BUFFER_SIZE


# ---------------------------------------------------------------------------
# sample() output shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 32, 64])
def test_sample_output_shapes(batch_size: int):
    """sample() must return arrays with the correct shapes for every field."""
    buf = ReplayBuffer()
    _fill_buffer(buf, batch_size + 10)  # ensure enough data

    batch = buf.sample(batch_size)

    assert batch["spatial"].shape         == (batch_size, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert batch["scalars"].shape         == (batch_size, NUM_SCALARS)
    assert batch["actions"].shape         == (batch_size,)
    assert batch["rewards"].shape         == (batch_size,)
    assert batch["next_spatial"].shape    == (batch_size, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    assert batch["next_scalars"].shape    == (batch_size, NUM_SCALARS)
    assert batch["dones"].shape           == (batch_size,)
    assert batch["next_legal_mask"].shape == (batch_size, NUM_ACTIONS)


# ---------------------------------------------------------------------------
# sample() error handling
# ---------------------------------------------------------------------------

def test_sample_raises_when_buffer_empty():
    """sample() must raise ValueError when the buffer is empty."""
    buf = ReplayBuffer()
    with pytest.raises(ValueError):
        buf.sample(1)


def test_sample_raises_when_batch_larger_than_buffer():
    """sample() must raise ValueError when batch_size > len(buffer)."""
    buf = ReplayBuffer()
    _fill_buffer(buf, 5)
    with pytest.raises(ValueError):
        buf.sample(10)


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------

def test_single_transition_round_trip():
    """
    When only one transition is in the buffer, sample(1) must return exactly
    that transition — verifying that push() writes and sample() reads correctly.
    """
    buf = ReplayBuffer()
    t = _make_transition(seed=99)
    buf.push(t)

    batch = buf.sample(1)

    np.testing.assert_array_equal(batch["spatial"][0],         t["spatial"])
    np.testing.assert_array_equal(batch["scalars"][0],         t["scalars"])
    assert batch["actions"][0]                               == t["action"]
    assert batch["rewards"][0]                               == pytest.approx(t["reward"])
    np.testing.assert_array_equal(batch["next_spatial"][0],    t["next_spatial"])
    np.testing.assert_array_equal(batch["next_scalars"][0],    t["next_scalars"])
    assert batch["dones"][0]                                 == t["done"]
    np.testing.assert_array_equal(batch["next_legal_mask"][0], t["next_legal_mask"])


# ---------------------------------------------------------------------------
# No duplicate samples
# ---------------------------------------------------------------------------

def test_sample_has_no_duplicates():
    """
    sample() uses replace=False, so no transition index should appear twice
    in a single batch. We stamp each transition with a unique float reward
    (its push index) so duplicates in the sample are unambiguous.
    """
    buf = ReplayBuffer()
    n = 50

    # Push n transitions, each with a unique reward equal to its index
    for i in range(n):
        t = _make_transition(seed=i)
        t["reward"] = float(i)  # guaranteed unique across all n transitions
        buf.push(t)

    batch = buf.sample(n)

    rewards = batch["rewards"].tolist()
    assert len(rewards) == len(set(rewards)), (
        "Duplicate transitions found in a single sample — replace=False may not be working"
    )
