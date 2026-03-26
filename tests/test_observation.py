"""Tests for QuoridorState.get_observation()."""

import numpy as np

from quoridor.game import QuoridorState


def _fresh():
    state = QuoridorState()
    spatial, scalars = state.get_observation()
    return state, spatial, scalars


def test_shape():
    _, spatial, scalars = _fresh()
    assert spatial.shape == (4, 9, 9)
    assert scalars.shape == (2,)


def test_dtype():
    _, spatial, scalars = _fresh()
    assert spatial.dtype == np.float32
    assert scalars.dtype == np.float32


def test_pawn_channels_binary():
    """Each pawn channel has exactly one 1.0, rest are 0.0."""
    _, spatial, _ = _fresh()
    for ch in (0, 1):
        assert spatial[ch].sum() == 1.0
        assert set(spatial[ch].flat) <= {0.0, 1.0}


def test_pawn_channel_positions_initial():
    """P0 starts at [8,4], P1 at [0,4]. No flip since it's P0's turn."""
    _, spatial, _ = _fresh()
    assert spatial[0, 8, 4] == 1.0  # P0
    assert spatial[1, 0, 4] == 1.0  # P1


def test_scalars_normalized():
    """Both players start with 10 walls -> scalars should be [1.0, 1.0]."""
    _, _, scalars = _fresh()
    assert np.all((scalars >= 0.0) & (scalars <= 1.0))
    np.testing.assert_array_equal(scalars, [1.0, 1.0])


def test_wall_channel_empty_initial():
    _, spatial, _ = _fresh()
    assert spatial[2].sum() == 0.0
    assert spatial[3].sum() == 0.0


def test_wall_channel_reflects_placement():
    """After placing h-wall at (3,3), ch2[3,3] should be 1.0 from P0's view."""
    state = QuoridorState()
    state.place_fence(3, 3, "h")
    state.turn = 0  # override so we see P0's unflipped view
    spatial, _ = state.get_observation()
    assert spatial[2, 3, 3] == 1.0
    assert spatial[3].sum() == 0.0


def test_perspective_flip_p1():
    """When turn=1, board flips vertically. Both pawns should appear mirrored."""
    state = QuoridorState()
    state.turn = 1
    spatial, _ = state.get_observation()
    # P1 (now current) was at [0,4], flipped to [8,4]
    assert spatial[0, 8, 4] == 1.0
    # P0 (now opponent) was at [8,4], flipped to [0,4]
    assert spatial[1, 0, 4] == 1.0


def test_wall_flip():
    """h-wall at row 3 appears at row 4 (=7-3) in P1's flipped view."""
    state = QuoridorState()
    state.place_fence(3, 3, "h")  # now P1's turn -> flip applies
    spatial, _ = state.get_observation()
    assert spatial[2, 4, 3] == 1.0
