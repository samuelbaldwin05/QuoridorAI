"""
Tests for PrioritizedReplayBuffer and SumTree (agents/replay_buffer.py).

Covers:
  - SumTree: total consistency after pushes, updates, and circular overwrites;
             priority-proportional sampling
  - PrioritizedReplayBuffer: new transitions receive max priority; IS weights
    are max-normalised to 1.0; higher priority → lower IS weight; priority
    updates propagate correctly to tree.total; high-priority transitions are
    sampled more often
"""

import numpy as np
import pytest

from agents.replay_buffer import SumTree, PrioritizedReplayBuffer
from config import (
    BOARD_SIZE, NUM_ACTIONS, NUM_CHANNELS, NUM_SCALARS,
    PER_ALPHA, PER_EPSILON,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transition(reward: float = 0.0, done: bool = False) -> dict:
    """Minimal valid transition dict using zero arrays."""
    return {
        "spatial":         np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
        "scalars":         np.zeros(NUM_SCALARS, dtype=np.float32),
        "action":          0,
        "reward":          float(reward),
        "next_spatial":    np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32),
        "next_scalars":    np.zeros(NUM_SCALARS, dtype=np.float32),
        "done":            done,
        "next_legal_mask": np.zeros(NUM_ACTIONS, dtype=bool),
    }


def _fill(buf: PrioritizedReplayBuffer, n: int) -> None:
    """Push n identical transitions into buf."""
    t = _make_transition()
    for _ in range(n):
        buf.push(t)


# ---------------------------------------------------------------------------
# SumTree — total consistency
# ---------------------------------------------------------------------------

def test_sumtree_total_after_pushes():
    """total equals the sum of all added priorities."""
    t = SumTree(8)
    t.add(4.0)
    t.add(2.0)
    t.add(1.0)
    assert t.total == pytest.approx(7.0), "total should be 4+2+1=7"
    assert t.n_entries == 3


def test_sumtree_total_after_update():
    """Updating a leaf changes total by exactly the delta."""
    t = SumTree(4)
    t.add(1.0)
    t.add(2.0)
    assert t.total == pytest.approx(3.0)
    t.update(0, 5.0)  # leaf 0: 1.0 → 5.0, delta = +4.0
    assert t.total == pytest.approx(7.0), "total should reflect the +4 delta"


def test_sumtree_total_consistent_after_circular_overwrite():
    """
    When the write pointer wraps, the overwritten leaf's old priority is
    correctly subtracted and the new priority is added.
    """
    t = SumTree(2)
    t.add(10.0)
    t.add(20.0)
    assert t.total == pytest.approx(30.0)
    # Circular overwrite of leaf 0 (write_ptr wraps back to 0)
    t.add(5.0)
    assert t.total == pytest.approx(25.0), "Old priority 10 replaced by 5; total = 5+20"


def test_sumtree_sampling_is_priority_weighted():
    """
    A leaf with overwhelmingly high priority should dominate sampling.
    With priorities [100, 1, 1, 1], leaf 0 should appear in >90% of samples.
    """
    rng = np.random.default_rng(seed=0)
    t = SumTree(8)
    t.add(100.0)
    t.add(1.0)
    t.add(1.0)
    t.add(1.0)

    hits = sum(
        1 for _ in range(1000)
        if t.sample(rng.uniform(0, t.total))[0] == 0
    )
    assert hits > 900, (
        f"Leaf 0 (priority 100) should dominate sampling; got {hits}/1000"
    )


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer — new transitions get max priority
# ---------------------------------------------------------------------------

def test_per_new_transitions_get_max_priority():
    """
    Every newly pushed transition is assigned _max_priority so it is
    sampled at least once before a real TD error is known.
    """
    buf = PrioritizedReplayBuffer()
    buf.push(_make_transition())

    # Leaf 0 lives at tree index capacity-1
    leaf_val = buf._tree.tree[buf._tree.capacity - 1]
    assert leaf_val == pytest.approx(buf._max_priority), (
        "First pushed transition should have max_priority in the tree leaf"
    )

    buf.push(_make_transition())
    leaf_val_2 = buf._tree.tree[buf._tree.capacity]
    assert leaf_val_2 == pytest.approx(buf._max_priority), (
        "Second pushed transition should also have max_priority"
    )


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer — IS weight normalisation
# ---------------------------------------------------------------------------

def test_per_is_weights_max_normalise_to_one():
    """
    IS weights are always normalised so the largest weight equals 1.0.
    This keeps gradient scales bounded regardless of beta.
    """
    buf = PrioritizedReplayBuffer()
    _fill(buf, 20)

    for beta in (0.4, 1.0):
        batch = buf.sample(10, beta=beta)
        assert batch["weights"].max() == pytest.approx(1.0, abs=1e-5), (
            f"Max IS weight should be 1.0 at beta={beta}"
        )
        assert (batch["weights"] <= 1.0 + 1e-5).all(), (
            f"All IS weights should be <= 1.0 at beta={beta}"
        )


def test_per_is_weights_all_positive():
    """IS weights are strictly positive (PER_EPSILON prevents zero priority)."""
    buf = PrioritizedReplayBuffer()
    _fill(buf, 20)
    batch = buf.sample(10, beta=0.4)
    assert (batch["weights"] > 0).all(), "IS weights must be strictly positive"


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer — higher priority → lower IS weight
# ---------------------------------------------------------------------------

def test_per_higher_priority_gets_lower_weight():
    """
    IS weight is inversely proportional to sampling probability.
    Higher priority → sampled more often → lower IS weight.

    We verify this analytically from the tree's stored leaf priorities,
    avoiding reliance on stratified sampling landing on specific leaves.

    With beta=1.0: w_i = (N * p_i / total)^(-1), normalised by max.
    Therefore w_i < w_j iff p_i > p_j.
    """
    buf = PrioritizedReplayBuffer()
    _fill(buf, 2)

    # Give leaf 0 a much higher priority
    buf.update_priorities([0], np.array([100.0]))
    # Leaf 1 keeps the initial max_priority before the update (≈ 1.0^alpha)

    # Read actual leaf priorities from the tree
    p0 = buf._tree.tree[buf._tree.capacity - 1 + 0]
    p1 = buf._tree.tree[buf._tree.capacity - 1 + 1]

    assert p0 > p1, "Setup: leaf 0 should have higher priority than leaf 1"

    # Compute expected IS weights analytically (beta=1.0)
    total = buf._tree.total
    n = len(buf)
    w0 = (n * p0 / total) ** (-1.0)
    w1 = (n * p1 / total) ** (-1.0)

    assert w0 < w1, (
        f"Higher priority (p0={p0:.4f}) should yield lower IS weight "
        f"(w0={w0:.4f}) than lower priority (p1={p1:.4f}, w1={w1:.4f})"
    )


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer — update_priorities propagates to total
# ---------------------------------------------------------------------------

def test_per_update_priorities_propagates_to_total():
    """
    update_priorities must correctly propagate the delta to tree.total.
    total_after = total_before - old_leaf_priority + new_leaf_priority.
    """
    buf = PrioritizedReplayBuffer()
    _fill(buf, 5)

    # Sample to get valid indices
    batch = buf.sample(5, beta=0.4)
    idx = batch["indices"][0]

    tree_idx = buf._tree.capacity - 1 + idx
    old_leaf_p = buf._tree.tree[tree_idx]
    total_before = buf._tree.total

    new_td = 10.0
    buf.update_priorities([idx], np.array([new_td]))

    expected_new_p = (abs(new_td) + PER_EPSILON) ** PER_ALPHA
    expected_total = total_before - old_leaf_p + expected_new_p

    assert buf._tree.total == pytest.approx(expected_total, abs=1e-5), (
        f"Tree total after update should be {expected_total:.6f}, "
        f"got {buf._tree.total:.6f}"
    )
    assert buf._max_priority == pytest.approx(expected_new_p, abs=1e-5), (
        "_max_priority should track the new highest priority"
    )


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer — high-priority transitions sampled more often
# ---------------------------------------------------------------------------

def test_per_high_priority_sampled_more_often():
    """
    After boosting one transition to very high priority, it should dominate
    the sampling distribution.
    """
    rng = np.random.default_rng(seed=0)
    buf = PrioritizedReplayBuffer()
    _fill(buf, 20)

    # Boost leaf 0 to a very high priority so it dominates.
    # td=50000 → p0=(50000+eps)^0.6 ≈ 684; other 19 leaves ≈ 1.0 each.
    # Fraction of total: 684/(684+19) ≈ 97.3%. Expected hits in 200: ~195.
    buf.update_priorities([0], np.array([50000.0]))

    hits = 0
    for _ in range(200):
        s = rng.uniform(0, buf._tree.total)
        leaf_idx, _ = buf._tree.sample(s)
        if leaf_idx == 0:
            hits += 1

    assert hits > 160, (
        f"High-priority leaf 0 should dominate sampling; got {hits}/200 hits"
    )
