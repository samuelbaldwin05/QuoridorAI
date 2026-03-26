"""Tests for quoridor/action_encoding.py."""

import pytest

from quoridor.action_encoding import (
    NUM_ACTIONS, FENCE_GRID, PAWN_DELTAS,
    action_to_index, index_to_action,
)


def _all_valid_actions():
    """Generate all 137 valid action tuples."""
    actions = []
    for dr, dc in PAWN_DELTAS.values():
        actions.append(("move", dr, dc))
    for row in range(FENCE_GRID):
        for col in range(FENCE_GRID):
            actions.append(("fence", row, col, "h"))
    for row in range(FENCE_GRID):
        for col in range(FENCE_GRID):
            actions.append(("fence", row, col, "v"))
    actions.append(("pass",))
    return actions


def test_round_trip_all_actions():
    for action in _all_valid_actions():
        assert index_to_action(action_to_index(action)) == action


def test_round_trip_all_indices():
    for i in range(NUM_ACTIONS):
        assert action_to_index(index_to_action(i)) == i


def test_pawn_move_indices():
    expected = {
        (-1,  0): 0, (+1,  0): 1, ( 0, -1): 2, ( 0, +1): 3,
        (-1, -1): 4, (-1, +1): 5, (+1, -1): 6, (+1, +1): 7,
    }
    for delta, idx in expected.items():
        dr, dc = delta
        assert action_to_index(("move", dr, dc)) == idx
        assert index_to_action(idx) == ("move", dr, dc)


def test_h_wall_boundary():
    assert action_to_index(("fence", 0, 0, "h")) == 8
    assert action_to_index(("fence", 7, 7, "h")) == 71
    assert action_to_index(("fence", 0, 7, "h")) == 15
    assert action_to_index(("fence", 7, 0, "h")) == 64


def test_v_wall_boundary():
    assert action_to_index(("fence", 0, 0, "v")) == 72
    assert action_to_index(("fence", 7, 7, "v")) == 135


def test_pass_action():
    assert action_to_index(("pass",)) == 136
    assert index_to_action(136) == ("pass",)


def test_invalid_index_raises():
    with pytest.raises(ValueError):
        index_to_action(-1)
    with pytest.raises(ValueError):
        index_to_action(NUM_ACTIONS)


def test_invalid_action_raises():
    with pytest.raises(ValueError):
        action_to_index(("move", 99, 99))


def test_invalid_fence_out_of_bounds():
    with pytest.raises(ValueError):
        action_to_index(("fence", 8, 0, "h"))
    with pytest.raises(ValueError):
        action_to_index(("fence", 0, 8, "v"))
    with pytest.raises(ValueError):
        action_to_index(("fence", -1, 0, "h"))


def test_total_action_count():
    indices = [action_to_index(a) for a in _all_valid_actions()]
    assert len(indices) == NUM_ACTIONS
    assert len(set(indices)) == NUM_ACTIONS
    assert set(indices) == set(range(NUM_ACTIONS))
