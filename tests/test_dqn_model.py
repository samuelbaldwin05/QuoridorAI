"""
Tests for agents/dqn_model.py.

Covers:
  - Output tensor shape for various batch sizes
  - Illegal action masking (penalised actions are -1e9, legal ones are not)
  - Gradient flow through masked_fill (autograd must not break)
  - Model works correctly in both train() and eval() modes
    (BatchNorm behaves differently in each — eval() uses running stats)
"""

import torch
import pytest

from agents.dqn_model import DQNModel, _ILLEGAL_ACTION_PENALTY
from config import NUM_ACTIONS, NUM_CHANNELS, BOARD_SIZE, NUM_SCALARS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_inputs(batch_size: int, seed: int = 0):
    """Return random (spatial, scalars, legal_mask) tensors for a given batch size."""
    rng = torch.Generator()
    rng.manual_seed(seed)

    spatial    = torch.rand(batch_size, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE, generator=rng)
    scalars    = torch.rand(batch_size, NUM_SCALARS, generator=rng)
    # Random legal mask — ensure at least one legal action per row so the
    # model always has something valid to select
    legal_mask = torch.zeros(batch_size, NUM_ACTIONS, dtype=torch.bool)
    for i in range(batch_size):
        n_legal = torch.randint(1, NUM_ACTIONS, (1,), generator=rng).item()
        perm = torch.randperm(NUM_ACTIONS, generator=rng)[:n_legal]
        legal_mask[i, perm] = True

    return spatial, scalars, legal_mask


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
def test_output_shape(batch_size: int):
    """forward() always returns a tensor of shape (B, NUM_ACTIONS)."""
    model = DQNModel()
    spatial, scalars, legal_mask = _random_inputs(batch_size)
    q = model(spatial, scalars, legal_mask)
    assert q.shape == (batch_size, NUM_ACTIONS), (
        f"Expected ({batch_size}, {NUM_ACTIONS}), got {q.shape}"
    )


# ---------------------------------------------------------------------------
# Illegal action masking
# ---------------------------------------------------------------------------

def test_illegal_actions_are_penalised():
    """Q-values for illegal actions must equal _ILLEGAL_ACTION_PENALTY."""
    model = DQNModel()
    spatial, scalars, legal_mask = _random_inputs(batch_size=8, seed=1)
    q = model(spatial, scalars, legal_mask)

    illegal_q = q[~legal_mask]
    assert (illegal_q == _ILLEGAL_ACTION_PENALTY).all(), (
        "Some illegal actions were not set to the penalty value"
    )


def test_legal_actions_are_not_penalised():
    """Q-values for legal actions must not equal _ILLEGAL_ACTION_PENALTY."""
    model = DQNModel()
    spatial, scalars, legal_mask = _random_inputs(batch_size=8, seed=2)
    q = model(spatial, scalars, legal_mask)

    legal_q = q[legal_mask]
    assert (legal_q != _ILLEGAL_ACTION_PENALTY).all(), (
        "Some legal actions were incorrectly set to the penalty value"
    )


def test_single_legal_action_is_always_selected():
    """When only one action is legal, argmax must always return that action."""
    model = DQNModel()
    batch_size = 16
    spatial, scalars, _ = _random_inputs(batch_size, seed=3)

    # Build a mask with exactly one legal action per sample (action index == sample index % NUM_ACTIONS)
    legal_mask = torch.zeros(batch_size, NUM_ACTIONS, dtype=torch.bool)
    expected_actions = torch.arange(batch_size) % NUM_ACTIONS
    for i, a in enumerate(expected_actions):
        legal_mask[i, a] = True

    q = model(spatial, scalars, legal_mask)
    chosen = q.argmax(dim=1)

    assert (chosen == expected_actions).all(), (
        "argmax did not select the sole legal action for every sample"
    )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_backward_pass_does_not_error():
    """loss.backward() must complete without error through masked_fill."""
    model = DQNModel()
    spatial, scalars, legal_mask = _random_inputs(batch_size=4, seed=4)

    q = model(spatial, scalars, legal_mask)
    # Simulate a simple MSE loss against a zero target
    loss = q.mean()
    loss.backward()  # should not raise


def test_gradients_are_not_none_after_backward():
    """All model parameters must have gradients after a backward pass."""
    model = DQNModel()
    spatial, scalars, legal_mask = _random_inputs(batch_size=4, seed=5)

    q = model(spatial, scalars, legal_mask)
    q.mean().backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' has no gradient after backward()"


# ---------------------------------------------------------------------------
# train() vs eval() mode
# ---------------------------------------------------------------------------

def test_forward_works_in_eval_mode():
    """Model must produce valid output in eval() mode (BatchNorm uses running stats)."""
    model = DQNModel()
    model.eval()

    spatial, scalars, legal_mask = _random_inputs(batch_size=4, seed=6)

    with torch.no_grad():
        q = model(spatial, scalars, legal_mask)

    assert q.shape == (4, NUM_ACTIONS)
    # Illegal actions must still be masked in eval mode
    assert (q[~legal_mask] == _ILLEGAL_ACTION_PENALTY).all()


def test_train_and_eval_outputs_differ():
    """
    With BatchNorm, train() and eval() produce different outputs on the same
    input (before the running stats have warmed up). This test checks that the
    two modes are actually distinct so we catch any accidental mode mixup.
    """
    model = DQNModel()
    spatial, scalars, legal_mask = _random_inputs(batch_size=4, seed=7)

    model.train()
    q_train = model(spatial, scalars, legal_mask).detach()

    model.eval()
    with torch.no_grad():
        q_eval = model(spatial, scalars, legal_mask)

    # They should differ because BatchNorm uses batch stats in train vs running stats in eval
    assert not torch.allclose(q_train, q_eval), (
        "train() and eval() produced identical outputs — BatchNorm may not be functioning"
    )
