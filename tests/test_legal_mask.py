"""
Tests for QuoridorState.get_legal_mask().

Covers shape/dtype, pawn and wall action coverage, edge cases (no walls left,
diagonal jumps), and executability of every legal action.
"""

import numpy as np
import pytest
from quoridor.game import QuoridorState
from quoridor.action_encoding import (
    NUM_ACTIONS, DELTA_TO_INDEX, H_WALL_OFFSET, V_WALL_OFFSET,
    PASS_ACTION, index_to_action,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _execute_action(state: QuoridorState, idx: int) -> None:
    """Execute action *idx* on *state* in-place. Raises ValueError if illegal."""
    action = index_to_action(idx)
    if action[0] == "move":
        dr, dc = action[1], action[2]
        cur_r, cur_c = int(state.pos[state.turn, 0]), int(state.pos[state.turn, 1])
        # Find the valid destination that matches this direction sign
        for dest_r, dest_c in state.get_valid_moves():
            if (np.sign(dest_r - cur_r) == np.sign(dr) and
                    np.sign(dest_c - cur_c) == np.sign(dc)):
                state.move_to(dest_r, dest_c)
                return
        raise ValueError(f"No valid destination found for direction ({dr}, {dc})")
    elif action[0] == "fence":
        _, row, col, orientation = action
        state.place_fence(row, col, orientation)
    else:
        raise ValueError(f"Unexpected action type: {action[0]}")


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

def test_shape_and_dtype():
    state = QuoridorState()
    mask = state.get_legal_mask()
    assert mask.shape == (NUM_ACTIONS,), f"mask shape should be ({NUM_ACTIONS},), got {mask.shape}"
    assert mask.dtype == bool, f"mask dtype should be bool, got {mask.dtype}"


# ---------------------------------------------------------------------------
# Basic coverage
# ---------------------------------------------------------------------------

def test_initial_has_legal_moves():
    """The initial state has at least one legal action."""
    state = QuoridorState()
    assert state.get_legal_mask().any(), "No legal actions found in initial state"


def test_pass_always_illegal():
    """Pass (index 136) is never legal."""
    state = QuoridorState()
    assert not state.get_legal_mask()[PASS_ACTION], "Pass action should always be illegal"


def test_pawn_moves_included():
    """At least one pawn action (index 0–7) is True in the initial state."""
    state = QuoridorState()
    mask = state.get_legal_mask()
    assert mask[:8].any(), "No pawn moves found in initial state"


def test_walls_included():
    """At least one wall action (index 8–135) is True in the initial state."""
    state = QuoridorState()
    mask = state.get_legal_mask()
    assert mask[8:136].any(), "No wall actions found in initial state"


# ---------------------------------------------------------------------------
# Wall-count edge case
# ---------------------------------------------------------------------------

def test_no_walls_disables_fence_actions():
    """
    When the current player has no walls left, every fence action must be False.
    Wall checks inside _fence_ok() return False immediately when walls_left == 0.
    """
    state = QuoridorState()
    state.walls_left[state.turn] = 0
    mask = state.get_legal_mask()
    assert not mask[H_WALL_OFFSET:PASS_ACTION].any(), (
        "Fence actions should all be False when current player has no walls"
    )


# ---------------------------------------------------------------------------
# Executability
# ---------------------------------------------------------------------------

def test_legal_moves_are_executable():
    """Every True index in the mask can be executed without raising ValueError."""
    state = QuoridorState()
    mask = state.get_legal_mask()
    for idx in np.where(mask)[0]:
        clone = state.clone()
        try:
            _execute_action(clone, int(idx))
        except ValueError as e:
            pytest.fail(f"Legal action index {idx} raised ValueError: {e}")


# ---------------------------------------------------------------------------
# Jump encoding
# ---------------------------------------------------------------------------

def test_straight_jump_index():
    """
    When P1 is directly above P0 and the jump is unobstructed, "move up"
    (index 0, delta (-1, 0)) must be True in P0's mask.
    """
    state = QuoridorState()
    # Place pawns so P1 is one row above P0 with room to jump
    state.pos[0] = [4, 4]  # P0 at (4, 4)
    state.pos[1] = [3, 4]  # P1 at (3, 4) — directly above P0
    # P0 can jump straight to (2, 4); sign(-2) == -1 → index 0
    mask = state.get_legal_mask()
    assert mask[0], "Move-up (index 0) should be legal when P1 is directly above P0"


def test_diagonal_jump_index():
    """
    When P1 is directly above P0 but the straight jump is wall-blocked,
    diagonal jump actions (NW=4, NE=5) should appear in P0's mask.
    """
    state = QuoridorState()
    state.pos[0] = [4, 4]  # P0 at (4, 4)
    state.pos[1] = [3, 4]  # P1 at (3, 4)

    # Block the step from (3,4) to (2,4): _blocked(3,4,2,4) checks h_walls[2,3] or h_walls[2,4]
    state.h_walls[2, 4] = True

    mask = state.get_legal_mask()
    # Straight jump is blocked → diagonal jumps to (2,3) and (2,5) should be legal
    assert mask[DELTA_TO_INDEX[(-1, -1)]], "NW diagonal jump (index 4) should be legal when straight jump is blocked"
    assert mask[DELTA_TO_INDEX[(-1, +1)]], "NE diagonal jump (index 5) should be legal when straight jump is blocked"
    # The straight-up index should NOT be legal (jump is blocked, and 1-step is occupied by opponent)
    assert not mask[0], "Move-up (index 0) should not be legal when P1 is adjacent and jump is blocked"
