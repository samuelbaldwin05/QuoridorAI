"""
Tests for QuoridorEnv(use_bfs=True) BFS observation channels (quoridor/env.py).

Covers:
  - Spatial shape is (6, 9, 9) with use_bfs=True, (4, 9, 9) with use_bfs=False
  - BFS channels (ch4, ch5) are bounded in [0, 1]
  - Goal row has distance 0.0 in the current player's BFS channel
  - BFS channel values change (monotonically increase) after a wall is placed
    that lengthens a player's path
  - BFS channel value at the player's position matches shortest_path() / BFS_NORM_FACTOR
  - Channels remain bounded after multiple wall placements
"""

import numpy as np
import pytest

from quoridor.env import QuoridorEnv
from quoridor.game import QuoridorState
from agents.bot import HeuristicBot
from config import BOARD_SIZE, BFS_NORM_FACTOR


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

def test_bfs_reset_returns_six_channel_spatial():
    """QuoridorEnv(use_bfs=True).reset() returns (6, 9, 9) spatial."""
    env = QuoridorEnv(use_bfs=True)
    spatial, scalars = env.reset()
    assert spatial.shape == (6, BOARD_SIZE, BOARD_SIZE), (
        f"Expected (6, {BOARD_SIZE}, {BOARD_SIZE}), got {spatial.shape}"
    )
    assert spatial.dtype == np.float32
    assert scalars.shape == (2,)


def test_non_bfs_reset_returns_four_channel_spatial():
    """QuoridorEnv(use_bfs=False) still returns (4, 9, 9) spatial."""
    env = QuoridorEnv(use_bfs=False)
    spatial, _ = env.reset()
    assert spatial.shape == (4, BOARD_SIZE, BOARD_SIZE), (
        f"use_bfs=False should give (4, {BOARD_SIZE}, {BOARD_SIZE}); got {spatial.shape}"
    )


# ---------------------------------------------------------------------------
# BFS channel bounds
# ---------------------------------------------------------------------------

def test_bfs_channels_bounded_after_reset():
    """ch4 and ch5 must lie in [0, 1] on a fresh board."""
    env = QuoridorEnv(use_bfs=True)
    spatial, _ = env.reset()
    assert (spatial[4] >= 0.0).all() and (spatial[4] <= 1.0).all(), (
        "ch4 (current player BFS) must be in [0, 1]"
    )
    assert (spatial[5] >= 0.0).all() and (spatial[5] <= 1.0).all(), (
        "ch5 (opponent BFS) must be in [0, 1]"
    )


def test_bfs_channels_bounded_after_wall_placements():
    """ch4 and ch5 remain in [0, 1] after several walls are placed."""
    game = QuoridorState()
    # Place a handful of walls directly to perturb path lengths
    game.h_walls[7, 3] = True
    game.h_walls[6, 5] = True
    game.v_walls[4, 2] = True

    spatial, _ = game.get_observation(use_bfs=True)
    assert (spatial[4] >= 0.0).all() and (spatial[4] <= 1.0).all(), (
        "ch4 must remain in [0, 1] after wall placements"
    )
    assert (spatial[5] >= 0.0).all() and (spatial[5] <= 1.0).all(), (
        "ch5 must remain in [0, 1] after wall placements"
    )


# ---------------------------------------------------------------------------
# Goal row has zero distance
# ---------------------------------------------------------------------------

def test_bfs_goal_row_has_zero_distance():
    """
    The current player's BFS channel must be 0.0 on every cell of their
    goal row — those cells are seeded at distance 0 by the reverse BFS.

    On reset, turn=0, P0's goal is row 0. No flip. ch4 is P0's BFS map.
    Row 0 of ch4 must be all-zero.
    """
    env = QuoridorEnv(use_bfs=True)
    spatial, _ = env.reset()
    assert np.allclose(spatial[4, 0, :], 0.0), (
        "Row 0 of ch4 should be 0.0 — P0's goal row has distance 0 from goal"
    )


# ---------------------------------------------------------------------------
# BFS values change after a wall lengthens the path
# ---------------------------------------------------------------------------

def test_bfs_channel_changes_after_wall():
    """
    Placing a wall that forces a detour must change the BFS distance map.
    No cell should become closer (walls can only block, never shorten paths).
    At least one cell must become farther.
    """
    game = QuoridorState()
    spatial_before, _ = game.get_observation(use_bfs=True)
    ch4_before = spatial_before[4].copy()

    # Block P0's direct upward step from (8,4) to (7,4):
    # DOWN from (7,4)→(8,4): col1=4 > 0 → h_walls[7, 3].
    game.h_walls[7, 3] = True

    spatial_after, _ = game.get_observation(use_bfs=True)
    ch4_after = spatial_after[4]

    assert not np.array_equal(ch4_after, ch4_before), (
        "BFS channel should change after placing a blocking wall"
    )
    assert (ch4_after >= ch4_before - 1e-6).all(), (
        "No cell should become closer after a wall is placed (walls only block)"
    )
    assert (ch4_after > ch4_before).any(), (
        "At least one cell should be farther from goal after the wall"
    )


def test_bfs_channel_value_matches_shortest_path():
    """
    The BFS channel value at P0's position equals
    shortest_path(0) / BFS_NORM_FACTOR.

    This cross-checks bfs_distance_map against shortest_path for consistency.
    """
    game = QuoridorState()
    # Place wall that forces P0 at (8,4) to detour
    game.h_walls[7, 3] = True

    # turn=0, no flip — P0 is at (8,4)
    spatial, _ = game.get_observation(use_bfs=True)

    sp = game.shortest_path(0)
    expected = sp / BFS_NORM_FACTOR

    assert spatial[4, 8, 4] == pytest.approx(expected, abs=1e-5), (
        f"ch4[8,4] should equal shortest_path(0)/BFS_NORM_FACTOR = "
        f"{sp}/{BFS_NORM_FACTOR} = {expected:.4f}; got {spatial[4, 8, 4]:.4f}"
    )
