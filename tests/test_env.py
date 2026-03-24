"""
Tests for QuoridorEnv (quoridor/env.py).

Verifies the Gym-style wrapper: reset/step shapes, reward structure,
legal mask in info, error handling, and bot state management.
"""

import numpy as np
import pytest

from quoridor.env import QuoridorEnv
from quoridor.action_encoding import NUM_ACTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_episode(env: QuoridorEnv) -> tuple[list[float], dict]:
    """
    Run a full episode picking the first legal action each step.
    Returns (reward_list, final_info).
    """
    env.reset()
    rewards = []
    done = False
    info = {}
    while not done:
        mask = env.get_legal_mask()
        action = int(mask.argmax())          # first True index
        _, _, reward, done, info = env.step(action)
        rewards.append(reward)
    return rewards, info


# ---------------------------------------------------------------------------
# Shape and dtype tests
# ---------------------------------------------------------------------------

def test_reset_shapes():
    """reset() must return spatial (4,9,9) float32 and scalars (2,) float32."""
    env = QuoridorEnv()
    spatial, scalars = env.reset()

    assert spatial.shape == (4, 9, 9), f"Expected (4,9,9), got {spatial.shape}"
    assert scalars.shape == (2,),      f"Expected (2,), got {scalars.shape}"
    assert spatial.dtype == np.float32
    assert scalars.dtype == np.float32


def test_observation_shape_after_step():
    """step() must return spatial (4,9,9) and scalars (2,) float32."""
    env = QuoridorEnv()
    env.reset()

    mask = env.get_legal_mask()
    spatial, scalars, reward, done, info = env.step(int(mask.argmax()))

    assert spatial.shape == (4, 9, 9)
    assert scalars.shape == (2,)
    assert spatial.dtype == np.float32
    assert scalars.dtype == np.float32


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------

def test_episode_terminates():
    """A full episode must reach done=True without raising exceptions."""
    env = QuoridorEnv()
    rewards, _ = _run_episode(env)
    # If we got here without error, the episode terminated properly.
    assert True


# ---------------------------------------------------------------------------
# Reward structure
# ---------------------------------------------------------------------------

def test_reward_sparse():
    """All rewards except the terminal step must be 0.0."""
    env = QuoridorEnv()
    rewards, _ = _run_episode(env)

    for r in rewards[:-1]:
        assert r == 0.0, f"Non-terminal reward was {r}, expected 0.0"


def test_reward_terminal_is_nonzero():
    """The terminal reward must be exactly +1.0 or -1.0."""
    env = QuoridorEnv()
    rewards, _ = _run_episode(env)

    terminal = rewards[-1]
    assert terminal in (1.0, -1.0), (
        f"Terminal reward was {terminal}, expected +1.0 or -1.0"
    )


def test_reward_nonzero_exactly_once():
    """Exactly one non-zero reward per episode."""
    env = QuoridorEnv()
    rewards, _ = _run_episode(env)

    nonzero = [r for r in rewards if r != 0.0]
    assert len(nonzero) == 1, (
        f"Expected exactly 1 non-zero reward, got {len(nonzero)}: {nonzero}"
    )


# ---------------------------------------------------------------------------
# Legal mask in info
# ---------------------------------------------------------------------------

def test_legal_mask_in_info():
    """info['legal_mask'] must be (137,) bool with at least one True on non-terminal steps."""
    env = QuoridorEnv()
    env.reset()

    mask = env.get_legal_mask()
    _, _, _, done, info = env.step(int(mask.argmax()))

    assert "legal_mask" in info
    lm = info["legal_mask"]
    assert lm.shape == (NUM_ACTIONS,), f"Expected ({NUM_ACTIONS},), got {lm.shape}"
    assert lm.dtype == bool

    if not done:
        assert lm.any(), "Non-terminal legal_mask has no True entries"


def test_legal_mask_all_false_on_terminal():
    """info['legal_mask'] must be all False on the terminal step."""
    env = QuoridorEnv()
    rewards, info = _run_episode(env)

    lm = info["legal_mask"]
    assert not lm.any(), "Terminal legal_mask should be all False"


# ---------------------------------------------------------------------------
# get_legal_mask passthrough
# ---------------------------------------------------------------------------

def test_get_legal_mask_passthrough():
    """env.get_legal_mask() must match env.state.get_legal_mask()."""
    env = QuoridorEnv()
    env.reset()

    np.testing.assert_array_equal(
        env.get_legal_mask(),
        env.state.get_legal_mask(),
    )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_illegal_action_raises():
    """Stepping with a masked-out (illegal) action must raise ValueError."""
    env = QuoridorEnv()
    env.reset()

    mask = env.get_legal_mask()
    # Find the first False entry (illegal action).
    illegal_actions = np.where(~mask)[0]
    assert len(illegal_actions) > 0, "All actions are legal — cannot test illegal action"

    with pytest.raises(ValueError):
        env.step(int(illegal_actions[0]))


# ---------------------------------------------------------------------------
# Bot state management
# ---------------------------------------------------------------------------

def test_reset_resets_bot():
    """After reset(), the bot's move_count must be 0 (fresh state)."""
    env = QuoridorEnv()

    # Play a few steps to advance bot state.
    env.reset()
    mask = env.get_legal_mask()
    env.step(int(mask.argmax()))

    # Now reset and confirm bot is fresh.
    env.reset()
    assert env.bot.move_count == 0, (
        f"Bot move_count after reset() is {env.bot.move_count}, expected 0"
    )
