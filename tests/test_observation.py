"""
Tests for QuoridorState.get_observation().

Covers shape/dtype, channel content, scalar normalization, and the
perspective-flip that makes every observation appear from the current
player's POV.
"""

import numpy as np
import pytest
from quoridor.game import QuoridorState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh() -> tuple[QuoridorState, np.ndarray, np.ndarray]:
    """Return a new state plus its initial observation."""
    state = QuoridorState()
    spatial, scalars = state.get_observation()
    return state, spatial, scalars


# ---------------------------------------------------------------------------
# Shape and dtype
# ---------------------------------------------------------------------------

def test_shape():
    _, spatial, scalars = _fresh()
    assert spatial.shape == (4, 9, 9), f"spatial shape should be (4,9,9), got {spatial.shape}"
    assert scalars.shape == (2,), f"scalars shape should be (2,), got {scalars.shape}"


def test_dtype():
    _, spatial, scalars = _fresh()
    assert spatial.dtype == np.float32, f"spatial dtype should be float32, got {spatial.dtype}"
    assert scalars.dtype == np.float32, f"scalars dtype should be float32, got {scalars.dtype}"


# ---------------------------------------------------------------------------
# Pawn channels (ch0, ch1)
# ---------------------------------------------------------------------------

def test_pawn_channels_binary():
    """Each pawn channel has exactly one 1.0 and all other values are 0.0."""
    _, spatial, _ = _fresh()
    for ch in (0, 1):
        channel = spatial[ch]
        assert channel.sum() == 1.0, f"ch{ch} should sum to 1.0, got {channel.sum()}"
        assert set(channel.flat) <= {0.0, 1.0}, f"ch{ch} contains values other than 0 and 1"


def test_pawn_channel_positions_initial():
    """
    At game start it is P0's turn (no flip).
    ch0 should mark P0's starting position [8, 4].
    ch1 should mark P1's starting position [0, 4].
    """
    _, spatial, _ = _fresh()
    assert spatial[0, 8, 4] == 1.0, "ch0 should show P0 at row 8 col 4"
    assert spatial[1, 0, 4] == 1.0, "ch1 should show P1 at row 0 col 4"


# ---------------------------------------------------------------------------
# Scalar features
# ---------------------------------------------------------------------------

def test_scalars_normalized():
    """Scalar values must be in [0, 1]; both start at 1.0 (10 walls / 10)."""
    _, _, scalars = _fresh()
    assert np.all((scalars >= 0.0) & (scalars <= 1.0)), "scalars out of [0,1] range"
    np.testing.assert_array_equal(scalars, [1.0, 1.0],
                                   err_msg="Both players start with 10 walls → scalars should be [1.0, 1.0]")


# ---------------------------------------------------------------------------
# Wall channels (ch2, ch3)
# ---------------------------------------------------------------------------

def test_wall_channel_empty_initial():
    """No walls have been placed at game start → ch2 and ch3 are all zeros."""
    _, spatial, _ = _fresh()
    assert spatial[2].sum() == 0.0, "ch2 should be zero at start"
    assert spatial[3].sum() == 0.0, "ch3 should be zero at start"


def test_wall_channel_reflects_placement():
    """
    After placing an h-wall at (3, 3), ch2[3, 3] should be 1.0 in P0's view.
    place_fence() switches the turn to P1, so we manually restore turn=0 to
    observe the board from P0's unflipped perspective.
    """
    state = QuoridorState()
    state.place_fence(3, 3, "h")
    # Override turn so we see P0's (unflipped) perspective
    state.turn = 0
    spatial, _ = state.get_observation()
    assert spatial[2, 3, 3] == 1.0, "ch2[3,3] should be 1.0 after placing h-wall at (3,3)"
    assert spatial[3].sum() == 0.0, "ch3 should remain zero after an h-wall placement"


# ---------------------------------------------------------------------------
# Perspective flip (P1's turn)
# ---------------------------------------------------------------------------

def test_perspective_flip_p1():
    """
    At game start, manually set turn=1 to trigger the vertical flip.
    ch0 is the *current* player (P1) whose unflipped position is [0, 4].
    After flip: row = 8 - 0 = 8  →  ch0 should be at [8, 4].
    ch1 is P0 whose unflipped position is [8, 4].
    After flip: row = 8 - 8 = 0  →  ch1 should be at [0, 4].
    """
    state = QuoridorState()
    state.turn = 1
    spatial, _ = state.get_observation()
    assert spatial[0, 8, 4] == 1.0, "ch0 (P1 in P1's POV) should appear at bottom [8,4] after flip"
    assert spatial[1, 0, 4] == 1.0, "ch1 (P0 in P1's POV) should appear at top [0,4] after flip"


def test_wall_flip():
    """
    An h-wall placed at row 3 should appear at row (7 - 3) = 4 in P1's flipped view.
    After place_fence(), turn naturally advances to P1, so no manual override needed.
    """
    state = QuoridorState()
    state.place_fence(3, 3, "h")  # now it's P1's turn → flip applies
    spatial, _ = state.get_observation()
    # np.flipud on 8 rows maps index i → (7 - i): row 3 → row 4
    assert spatial[2, 4, 3] == 1.0, (
        "h-wall at row 3 should appear at row 4 (= 7-3) in P1's flipped observation"
    )
