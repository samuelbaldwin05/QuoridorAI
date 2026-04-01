"""
Tests for n-step return accumulation logic from scripts/train_dqn.py.

The n-step loop is extracted verbatim into _run_nstep_loop() so these tests
have no dependency on wandb, argparse, or torch. All spatial/scalar fields use
minimal numpy arrays — only reward, done, and action values are exercised.

Covers:
  - Discounted reward sum is computed as Σ γⁱ rᵢ
  - Bootstrap uses the newest transition's next_state and done flag
  - Terminal flush pushes all remaining transitions with done=True
  - Decreasing-lookback reward sums on flush are correct
  - oldest["action"] and newest["done"] are wired correctly
"""

from collections import deque

import numpy as np
import pytest

from config import GAMMA, N_STEP_RETURNS


# ---------------------------------------------------------------------------
# Loop extraction
# ---------------------------------------------------------------------------

class _MockBuffer:
    """Lightweight stand-in for PrioritizedReplayBuffer used in the loop."""
    def __init__(self):
        self.pushed = []

    def push(self, transition: dict) -> None:
        self.pushed.append(transition)


def _run_nstep_loop(
    transitions: list[dict],
    gamma: float = GAMMA,
    n: int = N_STEP_RETURNS,
) -> list[dict]:
    """
    Verbatim copy of the n-step accumulation block from train_dqn.py,
    parameterised by gamma and n for unit-testing without importing the script.

    Returns the list of transitions pushed to the replay buffer.
    """
    buffer = _MockBuffer()
    n_step_buffer: deque = deque()

    for t in transitions:
        spatial         = t["spatial"]
        scalars         = t["scalars"]
        action          = t["action"]
        reward          = t["reward"]
        next_spatial    = t["next_spatial"]
        next_scalars    = t["next_scalars"]
        done            = t["done"]
        next_legal_mask = t["next_legal_mask"]

        n_step_buffer.append({
            "spatial":         spatial,
            "scalars":         scalars,
            "action":          action,
            "reward":          reward,
            "next_spatial":    next_spatial,
            "next_scalars":    next_scalars,
            "done":            done,
            "next_legal_mask": next_legal_mask,
        })

        if len(n_step_buffer) >= n:
            n_step_reward = sum(
                gamma ** i * n_step_buffer[i]["reward"]
                for i in range(n)
            )
            oldest = n_step_buffer[0]
            newest = n_step_buffer[-1]
            buffer.push({
                "spatial":         oldest["spatial"],
                "scalars":         oldest["scalars"],
                "action":          oldest["action"],
                "reward":          n_step_reward,
                "next_spatial":    newest["next_spatial"],
                "next_scalars":    newest["next_scalars"],
                "done":            newest["done"],
                "next_legal_mask": newest["next_legal_mask"],
            })
            n_step_buffer.popleft()

        if done:
            while n_step_buffer:
                n_len = len(n_step_buffer)
                n_step_reward = sum(
                    gamma ** i * n_step_buffer[i]["reward"]
                    for i in range(n_len)
                )
                oldest = n_step_buffer[0]
                newest = n_step_buffer[-1]
                buffer.push({
                    "spatial":         oldest["spatial"],
                    "scalars":         oldest["scalars"],
                    "action":          oldest["action"],
                    "reward":          n_step_reward,
                    "next_spatial":    newest["next_spatial"],
                    "next_scalars":    newest["next_scalars"],
                    "done":            newest["done"],
                    "next_legal_mask": newest["next_legal_mask"],
                })
                n_step_buffer.popleft()

    return buffer.pushed


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_ZERO = np.zeros(1, dtype=np.float32)


def _step(reward: float = 0.0, done: bool = False, action: int = 0) -> dict:
    """Minimal transition dict — only reward, done, and action vary."""
    return {
        "spatial":         _ZERO,
        "scalars":         _ZERO,
        "action":          action,
        "reward":          reward,
        "next_spatial":    _ZERO,
        "next_scalars":    _ZERO,
        "done":            done,
        "next_legal_mask": _ZERO,
    }


# ---------------------------------------------------------------------------
# Reward accumulation
# ---------------------------------------------------------------------------

def test_nstep_reward_correctly_discounted():
    """
    First pushed transition's reward equals Σ GAMMA^i * rewards[i]
    for i in 0..N_STEP_RETURNS-1.
    """
    rewards = [1.0, 0.5, 0.25, 0.125, 0.0625]
    assert len(rewards) == N_STEP_RETURNS, "Test requires exactly N_STEP_RETURNS transitions"

    transitions = [_step(r) for r in rewards]
    pushed = _run_nstep_loop(transitions)

    expected = sum(GAMMA ** i * rewards[i] for i in range(N_STEP_RETURNS))
    assert pushed[0]["reward"] == pytest.approx(expected, abs=1e-5), (
        f"First pushed reward should be discounted sum {expected:.6f}"
    )


def test_nstep_reward_uses_correct_gamma_exponents():
    """
    Verify the exponents are i=0,1,2,... by using a simple all-ones reward.
    Expected: 1 + γ + γ² = (1-γ³)/(1-γ).
    """
    n = 3
    gamma = 0.99
    rewards = [1.0, 1.0, 1.0]
    transitions = [_step(r) for r in rewards]
    pushed = _run_nstep_loop(transitions, gamma=gamma, n=n)

    expected = 1.0 + gamma + gamma ** 2  # ≈ 2.9701
    assert pushed[0]["reward"] == pytest.approx(expected, abs=1e-4), (
        f"All-ones reward with n=3 should sum to ~{expected:.4f}"
    )


# ---------------------------------------------------------------------------
# Terminal flush
# ---------------------------------------------------------------------------

def test_terminal_flush_pushes_all_remaining():
    """
    When done=True and the main loop never fired (episode shorter than N),
    the flush must push exactly len(buffer) transitions.
    """
    # 3 normal steps + 1 terminal step; N=5 so main loop never fires
    transitions = [_step(0.0)] * 3 + [_step(1.0, done=True)]
    pushed = _run_nstep_loop(transitions)

    assert len(pushed) == 4, (
        f"Flush should push all 4 buffered transitions; pushed {len(pushed)}"
    )


def test_terminal_flush_all_done_true():
    """
    Every transition pushed during the flush must have done=True
    because the newest entry in the buffer is the terminal transition.
    """
    transitions = [_step(0.0)] * 3 + [_step(1.0, done=True)]
    pushed = _run_nstep_loop(transitions)

    assert all(p["done"] for p in pushed), (
        "All flushed transitions should carry done=True from the terminal entry"
    )


def test_terminal_flush_decreasing_lookback_rewards():
    """
    Flush rewards should match decreasing-lookback discounted returns.

    Transitions: [r=0, r=0, r=0, r=1(done)], N=5.
    Flush order and expected reward:
      iteration 1 (4 in buffer): 0 + 0 + 0 + γ³·1 = γ³
      iteration 2 (3 in buffer): 0 + 0 + γ²·1     = γ²
      iteration 3 (2 in buffer): 0 + γ·1           = γ
      iteration 4 (1 in buffer): 1·1               = 1.0
    """
    transitions = [_step(0.0), _step(0.0), _step(0.0), _step(1.0, done=True)]
    pushed = _run_nstep_loop(transitions)

    expected = [GAMMA ** 3, GAMMA ** 2, GAMMA ** 1, 1.0]
    for i, (p, exp) in enumerate(zip(pushed, expected)):
        assert p["reward"] == pytest.approx(exp, abs=1e-5), (
            f"Flush iteration {i}: expected reward {exp:.6f}, got {p['reward']:.6f}"
        )


# ---------------------------------------------------------------------------
# Oldest/newest wiring
# ---------------------------------------------------------------------------

def test_nstep_uses_oldest_action():
    """
    The pushed transition's action must come from the OLDEST entry in the
    n-step window (the one whose state we are computing a target for).
    """
    # N+1 distinct actions: 0,1,2,3,4,5 with N=5
    transitions = [_step(action=i) for i in range(N_STEP_RETURNS + 1)]
    pushed = _run_nstep_loop(transitions)

    assert pushed[0]["action"] == 0, (
        f"First pushed action should be 0 (oldest); got {pushed[0]['action']}"
    )
    assert pushed[1]["action"] == 1, (
        f"Second pushed action should be 1; got {pushed[1]['action']}"
    )


def test_nstep_uses_newest_done():
    """
    The pushed transition's done flag must come from the NEWEST entry in the
    n-step window (the furthest-ahead state used for bootstrapping).
    """
    # Window of N: all done=False until the last one
    transitions = [_step(done=False)] * (N_STEP_RETURNS - 1) + [_step(done=True)]
    pushed = _run_nstep_loop(transitions)

    # The main loop fires once when buffer hits N (the terminal is the newest)
    assert pushed[0]["done"] is True, (
        "done flag should come from the newest (terminal) entry in the window"
    )
