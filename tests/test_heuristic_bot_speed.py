"""
Tests for HeuristicBot speed improvements in agents/bot.py.

Covers two optimizations to _find_high_impact_fence():
  1. Path-guided candidate selection (_bfs_path + _fences_blocking_path):
     only fences that cross the opponent's current shortest path are evaluated,
     reducing candidates from ~128 to ~18.
  2. Cached me_dist: our distance is precomputed once outside the loop instead
     of being recomputed on every fence evaluation.

Test strategy:
  - Correctness: _bfs_path and _fences_blocking_path produce geometrically
    correct outputs against known game states and hand-computed expectations.
  - Speed (BFS call counting): patch QuoridorState.shortest_path with a counter
    and assert the new code calls it far fewer times than the old O(200+) approach.
  - Wall-clock regression guard: full choose_action() must finish in < 50ms.
"""

import time
import pytest

from agents.bot import HeuristicBot
from quoridor.game import QuoridorState, BOARD_SIZE, FENCE_GRID


# ---------------------------------------------------------------------------
# Board state helpers
# ---------------------------------------------------------------------------

def _fresh_game() -> QuoridorState:
    """Standard starting position, no walls placed."""
    return QuoridorState()


def _mid_game_state() -> QuoridorState:
    """
    Mid-game board: both players advanced, 4 walls placed, 6 walls remaining.

    Layout:
        Player 0 (goal=row 0) at (5, 4)
        Player 1 (goal=row 8) at (3, 4)
        h_walls at (4, 2) and (4, 5) — partial barrier between them
        v_walls at (3, 3) and (3, 5) — side walls forcing a detour

    Both players have 6 walls remaining (started with 10, 4 used total but
    wall counts are tracked per-player; assign 2 used per player).
    The state is hand-verified to keep both players' paths open.
    """
    state = QuoridorState()
    state.pos[0] = [5, 4]
    state.pos[1] = [3, 4]
    state.h_walls[4, 2] = True
    state.h_walls[4, 5] = True
    state.v_walls[3, 3] = True
    state.v_walls[3, 5] = True
    state.walls_left[0] = 8   # 2 walls used by player 0's placement
    state.walls_left[1] = 8
    return state


# ---------------------------------------------------------------------------
# Correctness: _bfs_path
# ---------------------------------------------------------------------------

def test_bfs_path_player0_ends_at_goal_row():
    """Player 0's BFS path must end at row 0 on a fresh board."""
    game = _fresh_game()
    bot  = HeuristicBot()
    path = bot._bfs_path(game, player=0)
    assert len(path) > 1, "Path should have at least start + one step"
    assert path[-1][0] == int(game.goals[0]), (
        f"Last cell row {path[-1][0]} != goal row {game.goals[0]}"
    )


def test_bfs_path_player1_ends_at_goal_row():
    """Player 1's BFS path must end at row 8 on a fresh board."""
    game = _fresh_game()
    bot  = HeuristicBot()
    path = bot._bfs_path(game, player=1)
    assert path[-1][0] == int(game.goals[1])


def test_bfs_path_length_matches_shortest_path_open_board():
    """
    len(bfs_path) - 1 must equal game.shortest_path() on an open board.
    Both measure the same BFS distance; the path version just stores the route.
    """
    game = _fresh_game()
    bot  = HeuristicBot()
    for player in (0, 1):
        path = bot._bfs_path(game, player)
        dist = game.shortest_path(player)
        assert len(path) - 1 == dist, (
            f"Player {player}: path length {len(path)-1} != shortest_path {dist}"
        )


def test_bfs_path_length_matches_shortest_path_with_walls():
    """Path length must match shortest_path even when walls create detours."""
    game = _mid_game_state()
    bot  = HeuristicBot()
    for player in (0, 1):
        path = bot._bfs_path(game, player)
        dist = game.shortest_path(player)
        assert len(path) - 1 == dist, (
            f"Player {player}: path length {len(path)-1} != shortest_path {dist}"
        )


def test_bfs_path_all_steps_adjacent():
    """Every consecutive cell pair in the path must be one step apart."""
    game = _mid_game_state()
    bot  = HeuristicBot()
    path = bot._bfs_path(game, player=0)
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        step = abs(r2 - r1) + abs(c2 - c1)
        assert step == 1, (
            f"Non-adjacent step at index {i}: {path[i]} → {path[i+1]}"
        )


def test_bfs_path_already_at_goal_returns_single_cell():
    """If a player is already at their goal row, path should be [start]."""
    game = _fresh_game()
    # Player 1 starts at row 0, which IS their start but NOT their goal (row 8).
    # Manually move player 1 to their goal row.
    game.pos[1] = [8, 4]
    bot  = HeuristicBot()
    path = bot._bfs_path(game, player=1)
    assert path == [(8, 4)], f"Expected [(8,4)], got {path}"


# ---------------------------------------------------------------------------
# Correctness: _fences_blocking_path
# ---------------------------------------------------------------------------

def test_fences_blocking_path_all_in_bounds():
    """Every candidate from _fences_blocking_path must be within FENCE_GRID."""
    game = _mid_game_state()
    bot  = HeuristicBot()
    path = bot._bfs_path(game, player=1)
    candidates = bot._fences_blocking_path(path)
    for row, col, ori in candidates:
        assert 0 <= row < FENCE_GRID, f"row {row} out of bounds in {(row,col,ori)}"
        assert 0 <= col < FENCE_GRID, f"col {col} out of bounds in {(row,col,ori)}"
        assert ori in ("h", "v"),     f"bad orientation {ori}"


def test_fences_blocking_path_vertical_move_gives_h_walls():
    """A purely vertical path (same column) must produce only h-wall candidates."""
    bot  = HeuristicBot()
    # Straight down path: (0,4) → (1,4) → (2,4)
    path = [(0, 4), (1, 4), (2, 4)]
    candidates = bot._fences_blocking_path(path)
    assert candidates, "Expected at least one candidate for a 2-step path"
    for _, _, ori in candidates:
        assert ori == "h", f"Vertical move should yield h-walls, got '{ori}'"


def test_fences_blocking_path_horizontal_move_gives_v_walls():
    """A purely horizontal path (same row) must produce only v-wall candidates."""
    bot  = HeuristicBot()
    # Straight right path: (4,2) → (4,3) → (4,4)
    path = [(4, 2), (4, 3), (4, 4)]
    candidates = bot._fences_blocking_path(path)
    assert candidates, "Expected at least one candidate for a 2-step path"
    for _, _, ori in candidates:
        assert ori == "v", f"Horizontal move should yield v-walls, got '{ori}'"


def test_fences_blocking_path_no_duplicates():
    """Returned candidates must not contain duplicate (row, col, ori) entries."""
    game = _mid_game_state()
    bot  = HeuristicBot()
    path = bot._bfs_path(game, player=1)
    candidates = bot._fences_blocking_path(path)
    assert len(candidates) == len(set(candidates)), "Duplicate candidates found"


def test_fences_blocking_path_single_step():
    """A single-step path (2 cells) produces at most 2 candidates."""
    bot       = HeuristicBot()
    path      = [(4, 4), (3, 4)]   # one step up
    candidates = bot._fences_blocking_path(path)
    # One vertical step → h-walls at fence_row=3, cols 3 and 4 (both in bounds)
    assert len(candidates) <= 2


# ---------------------------------------------------------------------------
# Correctness: _find_high_impact_fence
# ---------------------------------------------------------------------------

def test_find_high_impact_fence_returns_none_on_fresh_board():
    """
    On a fresh board, the opponent has a clear 8-step path and no fence can
    add 3+ steps in one placement. Should return None.
    """
    game = _fresh_game()
    bot  = HeuristicBot()
    result = bot._find_high_impact_fence(game)
    assert result is None


def test_find_high_impact_fence_result_is_valid_fence_or_none():
    """Result must be None or a 4-tuple ('fence', row, col, orientation)."""
    game = _mid_game_state()
    bot  = HeuristicBot()
    result = bot._find_high_impact_fence(game)
    if result is not None:
        kind, row, col, ori = result
        assert kind == "fence"
        assert 0 <= row < FENCE_GRID
        assert 0 <= col < FENCE_GRID
        assert ori in ("h", "v")


# ---------------------------------------------------------------------------
# Speed: BFS call counting
# ---------------------------------------------------------------------------

@pytest.fixture
def call_counter(monkeypatch):
    """
    Patch QuoridorState.shortest_path to count invocations.

    Returns a dict with key "n" incremented on every call. Using monkeypatch
    ensures the original is restored after each test regardless of outcome.
    """
    counter  = {"n": 0}
    original = QuoridorState.shortest_path

    def counting_sp(self, player):
        counter["n"] += 1
        return original(self, player)

    monkeypatch.setattr(QuoridorState, "shortest_path", counting_sp)
    return counter


def test_find_high_impact_fence_call_count_mid_game(call_counter):
    """
    On a mid-game board, _find_high_impact_fence() must call shortest_path()
    fewer than 25 times.

    The old implementation (_all_valid_fences + full _advantage per fence)
    made ~150–200 shortest_path calls on the same board. 25 is a generous
    ceiling that verifies path-guided candidates and cached me_dist are active.
    """
    game = _mid_game_state()
    bot  = HeuristicBot()
    bot._find_high_impact_fence(game)
    assert call_counter["n"] < 25, (
        f"Expected <25 shortest_path calls, got {call_counter['n']}. "
        "Path-guided candidate selection or me_dist caching may not be active."
    )


def test_find_high_impact_fence_call_count_fresh_board(call_counter):
    """
    On a fresh board (no walls), opponent path is a straight 8-step column → 16
    raw candidates, all pass _fence_ok() structural checks.

    Call breakdown per candidate: 2 from _fence_ok() (_has_path calls shortest_path
    once per player) + 1 evaluation = 3. Plus 2 setup (me_dist + opp_dist).
    Expected total: 2 + 16 * 3 = 50.

    Old approach on same board: _all_valid_fences() checked all 128 positions
    (all valid on a fresh board) → 256 calls from _fence_ok + 256 from evaluation
    + 2 setup ≈ 514 calls. New approach is ~10× faster.
    """
    game = _fresh_game()
    bot  = HeuristicBot()
    bot._find_high_impact_fence(game)
    assert call_counter["n"] <= 60, (
        f"Expected ≤60 shortest_path calls on open board, got {call_counter['n']}. "
        "Old approach made ~514 calls on the same board."
    )


def test_find_high_impact_fence_fewer_calls_than_old_approach(call_counter):
    """
    Explicit regression guard: new approach must use fewer calls than the
    lower bound of the old approach (~50 calls even in a best case with few valid fences).
    """
    game = _mid_game_state()
    bot  = HeuristicBot()
    bot._find_high_impact_fence(game)
    # Old lower bound: 2 (current_adv) + 2 * ~25 (valid fence evaluation) = ~52
    assert call_counter["n"] < 50, (
        f"Call count {call_counter['n']} is not better than old approach lower bound (~50)."
    )


# ---------------------------------------------------------------------------
# Wall-clock regression guard
# ---------------------------------------------------------------------------

def test_choose_action_under_time_limit():
    """
    Full choose_action() on a mid-game board must complete in < 50ms.

    Threshold is ~10× the expected runtime on modern hardware (~3–5ms).
    This guards against accidentally re-introducing O(n²) BFS behaviour
    in future modifications to the fence search.
    """
    game = _mid_game_state()
    bot  = HeuristicBot()
    t0   = time.perf_counter()
    bot.choose_action(game)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 50, (
        f"choose_action took {elapsed_ms:.1f}ms — exceeds 50ms limit. "
        "Check for O(n²) BFS regression in _find_high_impact_fence()."
    )
