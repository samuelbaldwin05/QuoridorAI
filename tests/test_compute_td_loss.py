"""
Tests for the compute_td_loss TD target computation in scripts/train_dqn.py.

Covers:
  - Bootstrap term is discounted by GAMMA ** N_STEP_RETURNS, not GAMMA
  - Terminal transitions zero out the bootstrap regardless of N
  - Regression guard: GAMMA ** N_STEP_RETURNS != GAMMA (catches N=1 regression)

Strategy: compute_td_loss doesn't return the target tensor directly, so we
infer it via td_errors = |Q(s,a) - target|. Mock networks produce deterministic
Q-values so the expected target is calculable analytically.

Mock wiring:
  - online_net is called twice: once for the current state (all-True mask)
    and once for the next state (next_legal_mask).
  - actions = 0 throughout, so Q(s,a) = q_current_val (first column of first call).
  - Second online_net call (next state): action 0 has the highest value so
    argmax is deterministic.
  - target_net returns q_next_val uniformly, so the gathered value is always
    q_next_val regardless of which action was selected.
"""

import sys
import torch
import numpy as np
import pytest
from unittest.mock import MagicMock

from config import GAMMA, N_STEP_RETURNS, NUM_ACTIONS, NUM_CHANNELS, BOARD_SIZE, NUM_SCALARS


# ---------------------------------------------------------------------------
# Import compute_td_loss without triggering wandb / argparse at module level
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def compute_loss_fn():
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "train_dqn",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "train_dqn.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("wandb", MagicMock())
    spec.loader.exec_module(mod)
    return mod.compute_td_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(
    batch_size: int,
    rewards_val: float,
    dones_val: float,
    q_current_val: float,
    q_next_val: float,
) -> tuple:
    """
    Build a batch dict and deterministic mock networks.

    online_net is called with side_effect so the two calls return different
    tensors:
      call 1 (current state, all-True mask): action 0 = q_current_val, rest = 0
      call 2 (next state, next_legal_mask):  action 0 = 1.0 (argmax stays 0), rest = 0

    target_net returns q_next_val uniformly so gather at action 0 = q_next_val.

    Returns (batch, online_net, target_net, expected_td_errors).
    """
    spatial      = torch.zeros(batch_size, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    scalars      = torch.zeros(batch_size, NUM_SCALARS)
    actions      = torch.zeros(batch_size, dtype=torch.int64)   # always action 0
    rewards      = torch.full((batch_size,), rewards_val)
    next_spatial = torch.zeros(batch_size, NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    next_scalars = torch.zeros(batch_size, NUM_SCALARS)
    dones        = torch.full((batch_size,), dones_val)
    next_legal   = torch.ones(batch_size, NUM_ACTIONS, dtype=torch.bool)
    weights      = torch.ones(batch_size)

    batch = {
        "spatial":         spatial,
        "scalars":         scalars,
        "actions":         actions,
        "rewards":         rewards,
        "next_spatial":    next_spatial,
        "next_scalars":    next_scalars,
        "dones":           dones,
        "next_legal_mask": next_legal,
        "weights":         weights,
        "indices":         list(range(batch_size)),
    }

    # Call 1: action 0 = q_current_val (the action taken in this batch)
    q_current_out = torch.zeros(batch_size, NUM_ACTIONS)
    q_current_out[:, 0] = q_current_val

    # Call 2: action 0 = 1.0 so argmax is deterministic at action 0
    q_next_online_out = torch.zeros(batch_size, NUM_ACTIONS)
    q_next_online_out[:, 0] = 1.0

    online_net = MagicMock(side_effect=[q_current_out, q_next_online_out])

    # target_net: uniform q_next_val — gathered value at action 0 = q_next_val
    q_target_out = torch.full((batch_size, NUM_ACTIONS), q_next_val)
    target_net = MagicMock(return_value=q_target_out)

    # Analytically derive the expected TD errors
    expected_target = rewards_val + GAMMA ** N_STEP_RETURNS * q_next_val * (1.0 - dones_val)
    expected_td_errors = np.full(batch_size, abs(q_current_val - expected_target), dtype=np.float32)

    return batch, online_net, target_net, expected_td_errors


# ---------------------------------------------------------------------------
# Test 1: Bootstrap discount exponent is N, not 1
# ---------------------------------------------------------------------------

def test_bootstrap_uses_gamma_to_the_n(compute_loss_fn):
    """
    td_errors must reflect GAMMA ** N_STEP_RETURNS in the bootstrap.

    If the code mistakenly uses GAMMA (i.e. exponent=1 instead of N), the
    computed target is too large and td_errors won't match.
    """
    batch, online_net, target_net, expected_td_errors = _make_inputs(
        batch_size=8,
        rewards_val=0.0,
        dones_val=0.0,
        q_current_val=0.3,
        q_next_val=0.5,
    )

    _, _, td_errors, _, _, _ = compute_loss_fn(
        batch, online_net, target_net, torch.device("cpu")
    )

    assert np.allclose(td_errors, expected_td_errors, atol=1e-5), (
        f"TD errors imply wrong bootstrap discount.\n"
        f"  Expected (γᴺ={GAMMA**N_STEP_RETURNS:.6f}): {expected_td_errors[0]:.6f}\n"
        f"  Got:                                        {td_errors[0]:.6f}\n"
        f"  Value if γ¹ were used: "
        f"{abs(0.3 - (0.0 + GAMMA * 0.5)):.6f}"
    )


# ---------------------------------------------------------------------------
# Test 2: Terminal transitions zero out the bootstrap
# ---------------------------------------------------------------------------

def test_terminal_transitions_have_no_bootstrap(compute_loss_fn):
    """
    When done=1.0, the bootstrap must be zeroed out.
    td_errors must equal |q_current - rewards| exactly.
    """
    batch, online_net, target_net, expected_td_errors = _make_inputs(
        batch_size=8,
        rewards_val=1.0,    # win reward
        dones_val=1.0,      # terminal
        q_current_val=0.3,
        q_next_val=0.9,     # should be irrelevant when done=1
    )

    _, _, td_errors, _, _, _ = compute_loss_fn(
        batch, online_net, target_net, torch.device("cpu")
    )

    # With done=1, target = rewards = 1.0; td_error = |0.3 - 1.0| = 0.7
    assert np.allclose(td_errors, expected_td_errors, atol=1e-5), (
        "Terminal transitions should zero out bootstrap. "
        f"Expected td_error={expected_td_errors[0]:.6f}, got {td_errors[0]:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3: Regression guard — GAMMA**N != GAMMA for N > 1
# ---------------------------------------------------------------------------

def test_n_step_discount_differs_from_single_step():
    """
    Sanity check: GAMMA ** N_STEP_RETURNS must be strictly less than GAMMA
    for N > 1. If this fails, tests 1 and 2 would pass vacuously even if
    the bug were re-introduced (N=1 case would be indistinguishable).
    """
    assert N_STEP_RETURNS > 1, (
        f"N_STEP_RETURNS={N_STEP_RETURNS}; N-step tests are only meaningful for N > 1"
    )
    assert GAMMA ** N_STEP_RETURNS < GAMMA, (
        f"Expected γᴺ < γ for N={N_STEP_RETURNS}, γ={GAMMA}, "
        f"but got γᴺ={GAMMA**N_STEP_RETURNS}"
    )
