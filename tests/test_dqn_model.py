"""Tests for the DQN model."""

import torch
import pytest

from agents.dqn_model import DQNModel, _ILLEGAL_ACTION_PENALTY
from config import NUM_ACTIONS, NUM_CHANNELS, BOARD_SIZE, NUM_SCALARS


def _random_inputs(batch_size, seed=0):
    rng = torch.Generator()
    rng.manual_seed(seed)

    spatial    = torch.rand(batch_size, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE, generator=rng)
    scalars    = torch.rand(batch_size, NUM_SCALARS, generator=rng)
    legal_mask = torch.zeros(batch_size, NUM_ACTIONS, dtype=torch.bool)
    for i in range(batch_size):
        n_legal = torch.randint(1, NUM_ACTIONS, (1,), generator=rng).item()
        perm = torch.randperm(NUM_ACTIONS, generator=rng)[:n_legal]
        legal_mask[i, perm] = True

    return spatial, scalars, legal_mask


@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
def test_output_shape(batch_size):
    model = DQNModel()
    spatial, scalars, legal_mask = _random_inputs(batch_size)
    q = model(spatial, scalars, legal_mask)
    assert q.shape == (batch_size, NUM_ACTIONS)


def test_illegal_actions_are_penalised():
    model = DQNModel()
    spatial, scalars, legal_mask = _random_inputs(batch_size=8, seed=1)
    q = model(spatial, scalars, legal_mask)
    assert (q[~legal_mask] == _ILLEGAL_ACTION_PENALTY).all()


def test_legal_actions_are_not_penalised():
    model = DQNModel()
    spatial, scalars, legal_mask = _random_inputs(batch_size=8, seed=2)
    q = model(spatial, scalars, legal_mask)
    assert (q[legal_mask] != _ILLEGAL_ACTION_PENALTY).all()


def test_single_legal_action_is_always_selected():
    """When only one action is legal, argmax must return that action."""
    model = DQNModel()
    batch_size = 16
    spatial, scalars, _ = _random_inputs(batch_size, seed=3)

    legal_mask = torch.zeros(batch_size, NUM_ACTIONS, dtype=torch.bool)
    expected_actions = torch.arange(batch_size) % NUM_ACTIONS
    for i, a in enumerate(expected_actions):
        legal_mask[i, a] = True

    q = model(spatial, scalars, legal_mask)
    assert (q.argmax(dim=1) == expected_actions).all()


def test_backward_pass_does_not_error():
    model = DQNModel()
    spatial, scalars, legal_mask = _random_inputs(batch_size=4, seed=4)
    q = model(spatial, scalars, legal_mask)
    q.mean().backward()


def test_gradients_are_not_none_after_backward():
    model = DQNModel()
    spatial, scalars, legal_mask = _random_inputs(batch_size=4, seed=5)
    q = model(spatial, scalars, legal_mask)
    q.mean().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for '{name}'"


def test_forward_works_in_eval_mode():
    model = DQNModel()
    model.eval()
    spatial, scalars, legal_mask = _random_inputs(batch_size=4, seed=6)
    with torch.no_grad():
        q = model(spatial, scalars, legal_mask)
    assert q.shape == (4, NUM_ACTIONS)
    assert (q[~legal_mask] == _ILLEGAL_ACTION_PENALTY).all()


def test_train_and_eval_outputs_differ():
    """BatchNorm should produce different outputs in train vs eval mode."""
    model = DQNModel()
    spatial, scalars, legal_mask = _random_inputs(batch_size=4, seed=7)

    model.train()
    q_train = model(spatial, scalars, legal_mask).detach()

    model.eval()
    with torch.no_grad():
        q_eval = model(spatial, scalars, legal_mask)

    assert not torch.allclose(q_train, q_eval)
