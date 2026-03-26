"""Tests for QuoridorState.get_legal_mask()."""

import numpy as np
import pytest
from quoridor.game import QuoridorState
from quoridor.action_encoding import (
    NUM_ACTIONS, DELTA_TO_INDEX, H_WALL_OFFSET,
    PASS_ACTION, index_to_action,
)


def _execute_action(state, idx):
    """Execute action index on state in-place."""
    action = index_to_action(idx)
    if action[0] == "move":
        dr, dc = action[1], action[2]
        cur_r, cur_c = int(state.pos[state.turn, 0]), int(state.pos[state.turn, 1])
        for dest_r, dest_c in state.get_valid_moves():
            if (np.sign(dest_r - cur_r) == np.sign(dr) and
                    np.sign(dest_c - cur_c) == np.sign(dc)):
                state.move_to(dest_r, dest_c)
                return
        raise ValueError(f"No valid destination for direction ({dr}, {dc})")
    elif action[0] == "fence":
        _, row, col, orientation = action
        state.place_fence(row, col, orientation)
    else:
        raise ValueError(f"Unexpected action type: {action[0]}")


def test_shape_and_dtype():
    state = QuoridorState()
    mask = state.get_legal_mask()
    assert mask.shape == (NUM_ACTIONS,)
    assert mask.dtype == bool


def test_initial_has_legal_moves():
    state = QuoridorState()
    assert state.get_legal_mask().any()


def test_pass_always_illegal():
    state = QuoridorState()
    assert not state.get_legal_mask()[PASS_ACTION]


def test_pawn_moves_included():
    state = QuoridorState()
    assert state.get_legal_mask()[:8].any()


def test_walls_included():
    state = QuoridorState()
    assert state.get_legal_mask()[8:136].any()


def test_no_walls_disables_fence_actions():
    """When current player has 0 walls, all fence actions should be False."""
    state = QuoridorState()
    state.walls_left[state.turn] = 0
    mask = state.get_legal_mask()
    assert not mask[H_WALL_OFFSET:PASS_ACTION].any()


def test_legal_moves_are_executable():
    """Every True in the mask should be executable without error."""
    state = QuoridorState()
    mask = state.get_legal_mask()
    for idx in np.where(mask)[0]:
        clone = state.clone()
        try:
            _execute_action(clone, int(idx))
        except ValueError as e:
            pytest.fail(f"Legal action {idx} raised: {e}")


def test_straight_jump_index():
    """When P1 is directly above P0, move-up (index 0) should be legal."""
    state = QuoridorState()
    state.pos[0] = [4, 4]
    state.pos[1] = [3, 4]
    assert state.get_legal_mask()[0]


def test_diagonal_jump_index():
    """When straight jump is wall-blocked, diagonal jumps should be legal."""
    state = QuoridorState()
    state.pos[0] = [4, 4]
    state.pos[1] = [3, 4]
    state.h_walls[2, 4] = True  # block straight jump

    mask = state.get_legal_mask()
    assert mask[DELTA_TO_INDEX[(-1, -1)]]  # NW
    assert mask[DELTA_TO_INDEX[(-1, +1)]]  # NE
    assert not mask[0]  # straight up should be blocked
