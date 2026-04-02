"""Tests for QuoridorEnv."""

import numpy as np
import pytest

from quoridor.env import QuoridorEnv
from quoridor.action_encoding import NUM_ACTIONS


def _run_episode(env):
    """Run a full episode picking the first legal action each step."""
    env.reset()
    rewards = []
    done = False
    info = {}
    while not done:
        mask = env.get_legal_mask()
        action = int(mask.argmax())
        _, _, reward, done, info = env.step(action)
        rewards.append(reward)
    return rewards, info


def test_reset_shapes():
    env = QuoridorEnv()
    spatial, scalars = env.reset()
    assert spatial.shape == (4, 9, 9)
    assert scalars.shape == (2,)
    assert spatial.dtype == np.float32
    assert scalars.dtype == np.float32


def test_observation_shape_after_step():
    env = QuoridorEnv()
    env.reset()
    mask = env.get_legal_mask()
    spatial, scalars, reward, done, info = env.step(int(mask.argmax()))
    assert spatial.shape == (4, 9, 9)
    assert scalars.shape == (2,)


def test_episode_terminates():
    env = QuoridorEnv()
    _run_episode(env)  # should not raise


def test_reward_step_penalty():
    """Non-terminal rewards should be the step penalty when shaping is off."""
    from config import STEP_PENALTY
    env = QuoridorEnv(reward_shaping=False)
    rewards, _ = _run_episode(env)
    for r in rewards[:-1]:
        assert r == STEP_PENALTY


def test_reward_terminal_is_nonzero():
    env = QuoridorEnv()
    rewards, _ = _run_episode(env)
    assert rewards[-1] in (1.0, -1.0)


def test_reward_shaping_changes_reward():
    """With shaping on, at least one non-terminal reward differs from STEP_PENALTY."""
    from config import STEP_PENALTY
    env = QuoridorEnv(reward_shaping=True)
    rewards, _ = _run_episode(env)
    shaped = rewards[:-1]
    assert any(r != STEP_PENALTY for r in shaped), "Shaping produced no change"


def test_reward_shaping_terminal_unchanged():
    """Terminal reward should be exactly +1 or -1 regardless of shaping."""
    env = QuoridorEnv(reward_shaping=True)
    rewards, _ = _run_episode(env)
    assert rewards[-1] in (1.0, -1.0)


def test_reward_shaping_magnitudes():
    """Shaped non-terminal rewards should be small relative to win/loss."""
    env = QuoridorEnv(reward_shaping=True)
    rewards, _ = _run_episode(env)
    for r in rewards[:-1]:
        assert abs(r) < 0.5, f"Shaped reward {r} is too large"


def test_reward_shaping_disabled_override():
    """Constructor override should disable shaping regardless of config."""
    from config import STEP_PENALTY
    env = QuoridorEnv(reward_shaping=False)
    rewards, _ = _run_episode(env)
    for r in rewards[:-1]:
        assert r == STEP_PENALTY


def test_legal_mask_in_info():
    env = QuoridorEnv()
    env.reset()
    mask = env.get_legal_mask()
    _, _, _, done, info = env.step(int(mask.argmax()))

    lm = info["legal_mask"]
    assert lm.shape == (NUM_ACTIONS,)
    assert lm.dtype == bool
    if not done:
        assert lm.any()


def test_legal_mask_all_false_on_terminal():
    env = QuoridorEnv()
    _, info = _run_episode(env)
    assert not info["legal_mask"].any()


def test_get_legal_mask_passthrough():
    env = QuoridorEnv()
    env.reset()
    np.testing.assert_array_equal(env.get_legal_mask(), env.state.get_legal_mask())


def test_illegal_action_raises():
    env = QuoridorEnv()
    env.reset()
    mask = env.get_legal_mask()
    illegal_actions = np.where(~mask)[0]
    assert len(illegal_actions) > 0
    with pytest.raises(ValueError):
        env.step(int(illegal_actions[0]))


def test_reset_resets_bot():
    env = QuoridorEnv()
    env.reset()
    mask = env.get_legal_mask()
    env.step(int(mask.argmax()))

    env.reset()
    assert env.bot.move_count == 0
