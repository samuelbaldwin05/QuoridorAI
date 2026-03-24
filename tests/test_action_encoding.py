"""
Tests for quoridor/action_encoding.py.

Verifies the bijective mapping between integer action indices (0–136) and
game action tuples. Every component that depends on the action encoding —
the legal mask, replay buffer, and network output head — relies on this
invariant holding exactly.
"""

import pytest

from quoridor.action_encoding import (
    NUM_ACTIONS,
    PASS_ACTION,
    H_WALL_OFFSET,
    V_WALL_OFFSET,
    FENCE_GRID,
    PAWN_DELTAS,
    DELTA_TO_INDEX,
    action_to_index,
    index_to_action,
)


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------

def _all_valid_actions() -> list[tuple]:
    """Generate the full set of 137 valid action tuples."""
    actions: list[tuple] = []
    # 8 pawn moves
    for dr, dc in PAWN_DELTAS.values():
        actions.append(("move", dr, dc))
    # 64 horizontal walls
    for row in range(FENCE_GRID):
        for col in range(FENCE_GRID):
            actions.append(("fence", row, col, "h"))
    # 64 vertical walls
    for row in range(FENCE_GRID):
        for col in range(FENCE_GRID):
            actions.append(("fence", row, col, "v"))
    # 1 pass
    actions.append(("pass",))
    return actions


def test_round_trip_all_actions():
    """index_to_action(action_to_index(a)) == a for every valid action."""
    for action in _all_valid_actions():
        assert index_to_action(action_to_index(action)) == action, (
            f"Round-trip failed for action {action}"
        )


def test_round_trip_all_indices():
    """action_to_index(index_to_action(i)) == i for every index 0–136."""
    for i in range(NUM_ACTIONS):
        assert action_to_index(index_to_action(i)) == i, (
            f"Round-trip failed for index {i}"
        )


# ---------------------------------------------------------------------------
# Pawn move index tests
# ---------------------------------------------------------------------------

def test_pawn_move_indices():
    """Each of the 8 direction deltas maps to its expected index (0–7)."""
    expected = {
        (-1,  0): 0,   # up
        (+1,  0): 1,   # down
        ( 0, -1): 2,   # left
        ( 0, +1): 3,   # right
        (-1, -1): 4,   # NW
        (-1, +1): 5,   # NE
        (+1, -1): 6,   # SW
        (+1, +1): 7,   # SE
    }
    for delta, expected_idx in expected.items():
        dr, dc = delta
        assert action_to_index(("move", dr, dc)) == expected_idx, (
            f"Delta {delta} should map to index {expected_idx}"
        )
        assert index_to_action(expected_idx) == ("move", dr, dc), (
            f"Index {expected_idx} should decode to ('move', {dr}, {dc})"
        )


# ---------------------------------------------------------------------------
# Horizontal wall boundary tests
# ---------------------------------------------------------------------------

def test_h_wall_boundary():
    """Spot-check horizontal wall index arithmetic at corners and edges."""
    # First wall: (0, 0) → H_WALL_OFFSET + 0*8 + 0 = 8
    assert action_to_index(("fence", 0, 0, "h")) == 8
    assert index_to_action(8) == ("fence", 0, 0, "h")

    # Last wall: (7, 7) → 8 + 7*8 + 7 = 71
    assert action_to_index(("fence", 7, 7, "h")) == 71
    assert index_to_action(71) == ("fence", 7, 7, "h")

    # End of first row: (0, 7) → 8 + 0*8 + 7 = 15
    assert action_to_index(("fence", 0, 7, "h")) == 15
    assert index_to_action(15) == ("fence", 0, 7, "h")

    # Start of last row: (7, 0) → 8 + 7*8 + 0 = 64
    assert action_to_index(("fence", 7, 0, "h")) == 64
    assert index_to_action(64) == ("fence", 7, 0, "h")


# ---------------------------------------------------------------------------
# Vertical wall boundary tests
# ---------------------------------------------------------------------------

def test_v_wall_boundary():
    """Spot-check vertical wall index arithmetic at corners."""
    # First vertical wall: (0, 0) → V_WALL_OFFSET + 0 = 72
    assert action_to_index(("fence", 0, 0, "v")) == 72
    assert index_to_action(72) == ("fence", 0, 0, "v")

    # Last vertical wall: (7, 7) → 72 + 7*8 + 7 = 135
    assert action_to_index(("fence", 7, 7, "v")) == 135
    assert index_to_action(135) == ("fence", 7, 7, "v")


# ---------------------------------------------------------------------------
# Pass action
# ---------------------------------------------------------------------------

def test_pass_action():
    """Pass action encodes to and from index 136."""
    assert action_to_index(("pass",)) == PASS_ACTION
    assert PASS_ACTION == 136
    assert index_to_action(136) == ("pass",)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_invalid_index_raises():
    """index_to_action raises ValueError for out-of-range indices."""
    with pytest.raises(ValueError):
        index_to_action(-1)
    with pytest.raises(ValueError):
        index_to_action(NUM_ACTIONS)  # 137, one past the end


def test_invalid_action_raises():
    """action_to_index raises ValueError for an unrecognised move delta."""
    with pytest.raises(ValueError):
        action_to_index(("move", 99, 99))


def test_invalid_fence_out_of_bounds():
    """action_to_index raises ValueError when a fence coordinate is out of bounds."""
    with pytest.raises(ValueError):
        action_to_index(("fence", 8, 0, "h"))   # row 8 is out of the 8×8 grid
    with pytest.raises(ValueError):
        action_to_index(("fence", 0, 8, "v"))   # col 8 is out of the 8×8 grid
    with pytest.raises(ValueError):
        action_to_index(("fence", -1, 0, "h"))  # negative row


# ---------------------------------------------------------------------------
# Exhaustive uniqueness check
# ---------------------------------------------------------------------------

def test_total_action_count():
    """Exactly 137 unique indices are produced by all valid actions."""
    indices = [action_to_index(a) for a in _all_valid_actions()]
    assert len(indices) == NUM_ACTIONS, (
        f"Expected {NUM_ACTIONS} actions, got {len(indices)}"
    )
    assert len(set(indices)) == NUM_ACTIONS, (
        "Duplicate indices detected — encoding is not injective"
    )
    assert set(indices) == set(range(NUM_ACTIONS)), (
        "Indices do not cover the full range [0, NUM_ACTIONS)"
    )
