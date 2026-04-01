"""
Tests for HeuristicBot.choose_action() legality (agents/bot.py).

Every action returned by the bot must be legal in the given game state.
Tests cover random games, edge cases (no walls, adjacent opponent, near-goal),
and the property test runs 100 full games to expose rare illegal moves.
"""

import random

import numpy as np
import pytest

from agents.bot import HeuristicBot
from quoridor.game import QuoridorState
from config import INITIAL_WALLS_PER_PLAYER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh(p0=(8, 4), p1=(0, 4), turn=0, walls=INITIAL_WALLS_PER_PLAYER):
    """Build a QuoridorState with custom positions and wall counts."""
    s = QuoridorState()
    s.pos[0] = list(p0)
    s.pos[1] = list(p1)
    s.turn = turn
    s.walls_left[:] = walls
    return s


def _is_legal(game: QuoridorState, action: tuple) -> bool:
    """Return True iff action is legal in the given state."""
    if action[0] == "move":
        return (action[1], action[2]) in game.get_valid_moves()
    if action[0] == "fence":
        return game._fence_ok(action[1], action[2], action[3])
    return False


def _apply(game: QuoridorState, action: tuple) -> None:
    """Apply an action to the game state (raises ValueError if illegal)."""
    if action[0] == "move":
        game.move_to(action[1], action[2])
    else:
        game.place_fence(action[1], action[2], action[3])


# ---------------------------------------------------------------------------
# Property test: 100 random games
# ---------------------------------------------------------------------------

def test_bot_always_legal_across_random_games():
    """
    Every action chosen by HeuristicBot must be legal across 100 complete
    games. Uses alternating bot play for both P0 and P1.

    Seeds both random and numpy so failures are reproducible in CI.
    """
    random.seed(42)
    np.random.seed(42)

    bot0 = HeuristicBot()
    bot1 = HeuristicBot()

    for game_idx in range(100):
        game = QuoridorState()
        bot0.reset()
        bot1.reset()

        for step in range(200):   # safety cap — no Quoridor game exceeds ~150 moves
            if game.done:
                break

            bot = bot0 if game.turn == 0 else bot1
            action = bot.choose_action(game)

            assert _is_legal(game, action), (
                f"Game {game_idx}, step {step}, turn={game.turn}: "
                f"illegal action {action} in state {game}"
            )

            _apply(game, action)


# ---------------------------------------------------------------------------
# Edge case: no walls remaining
# ---------------------------------------------------------------------------

def test_bot_legal_with_no_walls_left():
    """With walls_left=[0,0] the bot must return a move action, not a fence."""
    game = _fresh(p0=(5, 4), p1=(3, 4), walls=0)
    bot = HeuristicBot()
    bot.move_count = 2   # skip opening book

    action = bot.choose_action(game)
    assert action[0] == "move", (
        f"Bot should return a move when no walls are left; got {action}"
    )
    assert _is_legal(game, action), f"Action {action} is illegal"


# ---------------------------------------------------------------------------
# Edge case: opponent directly adjacent
# ---------------------------------------------------------------------------

def test_bot_legal_with_adjacent_opponent():
    """
    With P1 directly ahead of P0, valid moves include jumps or diagonals.
    Run 10 times to cover any randomness in action selection.
    """
    game = _fresh(p0=(4, 4), p1=(3, 4), turn=0)
    bot = HeuristicBot()
    bot.move_count = 2   # skip opening book

    for _ in range(10):
        game2 = _fresh(p0=(4, 4), p1=(3, 4), turn=0)
        bot.reset()
        bot.move_count = 2
        action = bot.choose_action(game2)
        assert _is_legal(game2, action), (
            f"Illegal action with adjacent opponent: {action}"
        )


# ---------------------------------------------------------------------------
# Edge case: win in one
# ---------------------------------------------------------------------------

def test_bot_detects_win_in_one():
    """
    When P0 is one step from goal (row 0), the bot should pick the winning move.
    move_count=2 ensures the opening book is bypassed.
    """
    game = _fresh(p0=(1, 4), p1=(7, 4), turn=0)
    bot = HeuristicBot()
    bot.move_count = 2

    action = bot.choose_action(game)
    assert action == ("move", 0, 4), (
        f"Bot should win immediately from (1,4); got {action}"
    )


# ---------------------------------------------------------------------------
# Edge case: opponent near their goal (fence heuristics activate)
# ---------------------------------------------------------------------------

def test_bot_legal_opponent_near_goal():
    """
    When opponent is close to their goal, strategic fence heuristics may fire.
    Verify legality across 5 calls to exercise path diversity.
    """
    for _ in range(5):
        game = _fresh(p0=(5, 4), p1=(7, 4), turn=0)
        bot = HeuristicBot()
        bot.move_count = 2   # skip opening book
        action = bot.choose_action(game)
        assert _is_legal(game, action), (
            f"Illegal action when opponent is near their goal: {action}"
        )
