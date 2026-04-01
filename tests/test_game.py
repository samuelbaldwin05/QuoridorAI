"""
Tests for QuoridorState core game logic (quoridor/game.py).

Covers:
  - _blocked(): each direction with wall present/absent; boundary cases
  - _fence_ok(): path-blocking rejection, overlap rejection, crossing rejection,
                 no-walls rejection
  - get_valid_moves(): straight jump, diagonal jump, blocked diagonal, no backward diagonal
  - Win detection: P0 reaches row 0, P1 reaches row 8, no false positives mid-game
  - clone() independence: position, walls, wall counts, scalar fields
"""

import numpy as np
import pytest

from quoridor.game import QuoridorState
from config import FENCE_GRID, INITIAL_WALLS_PER_PLAYER


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fresh(p0=(8, 4), p1=(0, 4), turn=0, walls=INITIAL_WALLS_PER_PLAYER):
    """
    Build a QuoridorState with custom positions, turn, and wall counts.
    No walls are placed on the board.
    """
    s = QuoridorState()
    s.pos[0] = list(p0)
    s.pos[1] = list(p1)
    s.turn = turn
    s.walls_left[:] = walls
    return s


# ---------------------------------------------------------------------------
# _blocked — stepping DOWN
# ---------------------------------------------------------------------------

def test_blocked_down_left_wall():
    """h_walls[row, col-1] blocks the downward step at column col (left-side check)."""
    s = _fresh()
    # Step DOWN from (3,2) to (4,2). Left-side: col1=2 > 0 → h_walls[3, 1].
    s.h_walls[3, 1] = True
    assert s._blocked(3, 2, 4, 2), "h_walls[3,1] should block downward step at col 2"
    s.h_walls[3, 1] = False
    assert not s._blocked(3, 2, 4, 2), "No wall — step should be unblocked"


def test_blocked_down_right_wall():
    """h_walls[row, col] blocks the downward step at column col (right-side check)."""
    s = _fresh()
    # Step DOWN from (3,2) to (4,2). Right-side: col1=2 < FENCE_GRID=8 → h_walls[3, 2].
    s.h_walls[3, 2] = True
    assert s._blocked(3, 2, 4, 2), "h_walls[3,2] should block downward step at col 2"
    s.h_walls[3, 2] = False
    assert not s._blocked(3, 2, 4, 2), "No wall — step should be unblocked"


def test_blocked_no_wall_down_returns_false():
    """Clean board: downward step is never blocked."""
    s = _fresh()
    assert not s._blocked(3, 4, 4, 4), "No walls — downward step should be free"


# ---------------------------------------------------------------------------
# _blocked — stepping UP (symmetry branch)
# ---------------------------------------------------------------------------

def test_blocked_up_delegates_to_down():
    """
    Stepping UP is the reverse of stepping DOWN.
    h_walls[3,3] should block (4,4)→(3,4) just as it blocks (3,4)→(4,4).
    """
    s = _fresh()
    # Down from (3,4): col1=4 > 0 → left-side check h_walls[3, 3]
    s.h_walls[3, 3] = True
    assert s._blocked(4, 4, 3, 4), "h_walls[3,3] should block upward step at col 4"
    s.h_walls[3, 3] = False
    assert not s._blocked(4, 4, 3, 4)


# ---------------------------------------------------------------------------
# _blocked — stepping RIGHT
# ---------------------------------------------------------------------------

def test_blocked_right_top_wall():
    """v_walls[row-1, col] blocks the rightward step at row (top-side check)."""
    s = _fresh()
    # Step RIGHT from (4,3) to (4,4). Top-side: row1=4 > 0 → v_walls[3, 3].
    s.v_walls[3, 3] = True
    assert s._blocked(4, 3, 4, 4), "v_walls[3,3] should block rightward step at row 4"
    s.v_walls[3, 3] = False
    assert not s._blocked(4, 3, 4, 4)


def test_blocked_right_bottom_wall():
    """v_walls[row, col] blocks the rightward step at row (bottom-side check)."""
    s = _fresh()
    # Step RIGHT from (4,3) to (4,4). Bottom-side: row1=4 < FENCE_GRID=8 → v_walls[4, 3].
    s.v_walls[4, 3] = True
    assert s._blocked(4, 3, 4, 4), "v_walls[4,3] should block rightward step at row 4"
    s.v_walls[4, 3] = False
    assert not s._blocked(4, 3, 4, 4)


def test_blocked_no_wall_right_returns_false():
    """Clean board: rightward step is never blocked."""
    s = _fresh()
    assert not s._blocked(4, 3, 4, 4), "No walls — rightward step should be free"


# ---------------------------------------------------------------------------
# _blocked — boundary cases
# ---------------------------------------------------------------------------

def test_blocked_boundary_col0_down():
    """
    Stepping down at column 0: only the right-side check fires (col > 0 is False).
    Verify no IndexError and correct blocking via h_walls[row, 0].
    """
    s = _fresh()
    # col1=0: left-side guard (col1 > 0) is False; only h_walls[3, 0] is checked.
    assert not s._blocked(3, 0, 4, 0), "No wall at col 0 — should be free"
    s.h_walls[3, 0] = True
    assert s._blocked(3, 0, 4, 0), "h_walls[3,0] should block downward step at col 0"


def test_blocked_boundary_col7_down():
    """
    Stepping down at column 7: both left-side (h_walls[row,6]) and right-side
    (h_walls[row,7], since 7 < FENCE_GRID=8) checks fire independently.
    """
    s = _fresh()
    assert not s._blocked(7, 7, 8, 7)

    s.h_walls[7, 6] = True   # left-side: col1=7 > 0 → h_walls[7, 6]
    assert s._blocked(7, 7, 8, 7), "h_walls[7,6] should block step at col 7"
    s.h_walls[7, 6] = False

    s.h_walls[7, 7] = True   # right-side: col1=7 < 8 → h_walls[7, 7]
    assert s._blocked(7, 7, 8, 7), "h_walls[7,7] should block step at col 7"


# ---------------------------------------------------------------------------
# _fence_ok — path-blocking rejection
# ---------------------------------------------------------------------------

def test_fence_ok_allows_placement_without_path_cut():
    """Positive control: _fence_ok returns True when no path is cut."""
    s = _fresh(p0=(4, 1), p1=(0, 4))
    assert s._fence_ok(3, 1, "h"), "Should be allowed — P0 still has many paths to goal"


def test_fence_ok_rejects_path_cut():
    """
    When P0 has only one exit, the wall that seals it must be rejected.

    Setup: P0 at (4,1). Three walls placed directly (bypassing _fence_ok):
      - h_walls[4,0]: blocks DOWN from (4,1)
      - v_walls[4,0]: blocks LEFT from (4,1)
      - v_walls[4,1]: blocks RIGHT from (4,1)
    Only exit left: UP (4,1)→(3,1), blocked by h_walls[3,1].
    _fence_ok(3,1,'h') must return False.
    """
    s = _fresh(p0=(4, 1), p1=(0, 4))

    # Block DOWN: DOWN from (4,1)→(5,1) — left-side: col1=1>0 → h_walls[4, 0]
    s.h_walls[4, 0] = True
    # Block LEFT: RIGHT from (4,0)→(4,1) reversed — bottom-side: v_walls[4, 0]
    s.v_walls[4, 0] = True
    # Block RIGHT: RIGHT from (4,1)→(4,2) — bottom-side: v_walls[4, 1]
    s.v_walls[4, 1] = True

    # Confirm only UP is reachable
    assert s.get_valid_moves() == [(3, 1)], (
        "Setup error: P0 should have exactly one valid move (UP to (3,1))"
    )

    # h_walls[3,1] blocks UP: reversed DOWN from (3,1)→(4,1), right-side h_walls[3,1]
    # v_walls[3,1] is False so no crossing conflict
    assert not s._fence_ok(3, 1, "h"), (
        "_fence_ok should reject h_walls[3,1] because it cuts off P0's only exit"
    )


# ---------------------------------------------------------------------------
# _fence_ok — overlap rejection
# ---------------------------------------------------------------------------

def test_fence_ok_rejects_overlap_right():
    """A wall adjacent to the right of an existing wall (shared post) is rejected."""
    s = _fresh()
    s.h_walls[3, 4] = True
    # col=5: col > 0 → h_walls[3, 4] is True → rejected
    assert not s._fence_ok(3, 5, "h"), (
        "h_wall at col 5 overlaps with existing wall at col 4 (shared post)"
    )


def test_fence_ok_rejects_overlap_left():
    """A wall adjacent to the left of an existing wall (shared post) is rejected."""
    s = _fresh()
    s.h_walls[3, 4] = True
    # col=3: col < FENCE_GRID-1=7 → h_walls[3, 4] is True → rejected
    assert not s._fence_ok(3, 3, "h"), (
        "h_wall at col 3 overlaps with existing wall at col 4 (shared post)"
    )


def test_fence_ok_allows_non_adjacent_same_orientation():
    """A wall with a gap of 1+ between it and existing walls is allowed."""
    s = _fresh()
    s.h_walls[3, 4] = True
    # col=2: col+1=3 ≠ 4, col-1=1 ≠ 4 — no shared post
    assert s._fence_ok(3, 2, "h"), (
        "h_wall at col 2 is not adjacent to existing wall at col 4 — should be allowed"
    )


# ---------------------------------------------------------------------------
# _fence_ok — crossing rejection
# ---------------------------------------------------------------------------

def test_fence_ok_rejects_h_crossing_existing_v():
    """An h-wall crossing a v-wall at the same post is rejected."""
    s = _fresh()
    s.v_walls[3, 4] = True
    assert not s._fence_ok(3, 4, "h"), (
        "h_wall at (3,4) crosses existing v_wall at (3,4)"
    )


def test_fence_ok_rejects_v_crossing_existing_h():
    """A v-wall crossing an h-wall at the same post is rejected."""
    s = _fresh()
    s.h_walls[3, 4] = True
    assert not s._fence_ok(3, 4, "v"), (
        "v_wall at (3,4) crosses existing h_wall at (3,4)"
    )


# ---------------------------------------------------------------------------
# _fence_ok — no walls remaining
# ---------------------------------------------------------------------------

def test_fence_ok_rejects_when_no_walls_left():
    """A player with 0 walls remaining cannot place any fence."""
    s = _fresh(walls=0)
    assert not s._fence_ok(3, 4, "h"), "Player has no walls left"
    assert not s._fence_ok(3, 4, "v"), "Player has no walls left"


# ---------------------------------------------------------------------------
# get_valid_moves — jump rules
# ---------------------------------------------------------------------------

def test_straight_jump_over_opponent():
    """P0 can jump straight over P1 when the landing square is clear."""
    s = _fresh(p0=(5, 4), p1=(4, 4), turn=0)
    moves = s.get_valid_moves()
    assert (3, 4) in moves, "Straight jump to (3,4) over opponent at (4,4) should be legal"
    assert (4, 4) not in moves, "Cannot land on opponent's square"


def test_straight_jump_blocked_activates_diagonals():
    """When the straight-ahead jump is blocked, diagonal jumps become available."""
    s = _fresh(p0=(5, 4), p1=(4, 4), turn=0)
    # Block (4,4)→(3,4): DOWN from (3,4), right-side → h_walls[3,4]
    s.h_walls[3, 4] = True
    moves = s.get_valid_moves()
    assert (3, 4) not in moves, "Straight jump should be blocked by h_walls[3,4]"
    assert (4, 3) in moves, "NW diagonal jump should be available"
    assert (4, 5) in moves, "NE diagonal jump should be available"


def test_no_backward_diagonal_jump():
    """Diagonal jumps are only allowed perpendicular to the approach direction."""
    s = _fresh(p0=(5, 4), p1=(4, 4), turn=0)
    s.h_walls[3, 4] = True   # block straight jump
    moves = s.get_valid_moves()
    # P0 is approaching from below (dr=-1). Backward diagonals would be (6,3) and (6,5).
    assert (6, 3) not in moves, "Backward-left diagonal is not a valid jump"
    assert (6, 5) not in moves, "Backward-right diagonal is not a valid jump"


def test_diagonal_jump_blocked_by_wall():
    """A wall on the diagonal path removes that specific diagonal option."""
    s = _fresh(p0=(5, 4), p1=(4, 4), turn=0)
    s.h_walls[3, 4] = True   # block straight jump
    # Block NE diagonal: RIGHT from (4,4)→(4,5) — top-side: v_walls[3,4]
    s.v_walls[3, 4] = True
    moves = s.get_valid_moves()
    assert (4, 5) not in moves, "NE diagonal blocked by v_walls[3,4]"
    assert (4, 3) in moves, "NW diagonal should still be available"


# ---------------------------------------------------------------------------
# Win detection
# ---------------------------------------------------------------------------

def test_p0_wins_reaching_row_0():
    """P0 moving to row 0 triggers game end with winner=0."""
    s = _fresh(p0=(1, 4), p1=(7, 4), turn=0)
    result = s.move_to(0, 4)
    assert result is True, "move_to should return True on win"
    assert s.done is True
    assert s.winner == 0


def test_p1_wins_reaching_row_8():
    """P1 moving to row 8 triggers game end with winner=1."""
    s = _fresh(p0=(1, 4), p1=(7, 4), turn=1)
    result = s.move_to(8, 4)
    assert result is True, "move_to should return True on win"
    assert s.done is True
    assert s.winner == 1


def test_no_win_mid_game():
    """Moving to a non-goal square does not end the game."""
    s = _fresh(p0=(5, 4), p1=(3, 4), turn=0)
    # P1 is at (3,4) so P0 cannot step to (4,4). Move to (5,3) instead.
    result = s.move_to(5, 3)
    assert result is False, "Mid-game move should return False"
    assert s.done is False
    assert s.winner == -1


# ---------------------------------------------------------------------------
# clone() independence
# ---------------------------------------------------------------------------

def test_clone_pos_independent():
    """Mutating the clone's position does not affect the original."""
    s = _fresh(p0=(5, 3))
    c = s.clone()
    c.pos[0] = [2, 2]
    assert list(s.pos[0]) == [5, 3], "Original pos[0] should be unchanged"


def test_clone_h_walls_independent():
    """Mutating the clone's h_walls does not affect the original."""
    s = _fresh()
    c = s.clone()
    c.h_walls[3, 3] = True
    assert not s.h_walls[3, 3], "Original h_walls[3,3] should remain False"


def test_clone_v_walls_independent():
    """Mutating the clone's v_walls does not affect the original."""
    s = _fresh()
    c = s.clone()
    c.v_walls[5, 5] = True
    assert not s.v_walls[5, 5], "Original v_walls[5,5] should remain False"


def test_clone_walls_left_independent():
    """Mutating the clone's walls_left does not affect the original."""
    s = _fresh()
    c = s.clone()
    c.walls_left[0] = 3
    assert s.walls_left[0] == INITIAL_WALLS_PER_PLAYER, (
        "Original walls_left[0] should be unchanged"
    )


def test_clone_scalar_fields_independent():
    """Mutating turn/done/winner on the clone does not affect the original."""
    s = _fresh()
    c = s.clone()
    c.turn = 1
    c.done = True
    c.winner = 1
    assert s.turn == 0, "Original turn should be 0"
    assert s.done is False, "Original done should be False"
    assert s.winner == -1, "Original winner should be -1"
