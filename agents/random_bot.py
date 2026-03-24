"""
random_bot.py — Random opponent for curriculum training.

Used as the first-stage opponent before introducing HeuristicBot.
A fully random opponent is beatable, which ensures the replay buffer
accumulates some +1 (win) experiences early in training. Without any
wins, the agent only ever sees -1 rewards and Q-values collapse.

Interface matches HeuristicBot exactly so it is a drop-in replacement
inside QuoridorEnv.

Behaviour:
  - 30% chance to place a random legal wall (when walls remain)
  - 70% chance to make a random legal pawn move
  - Always falls back to a pawn move if no legal wall is found quickly
"""

import random

from quoridor.game import QuoridorState, FENCE_GRID

# Probability of attempting a wall placement on any given turn.
# Set low enough that pawn play still dominates, but walls appear
# often enough for the agent to learn to deal with them.
_WALL_PROB = 0.30

# Maximum random (row, col, orientation) samples before giving up
# and falling back to a pawn move. Avoids slow loops in dense boards.
_MAX_WALL_TRIES = 20


class RandomBot:
    """
    Uniformly random Quoridor opponent.

    On each turn, randomly decides to place a wall or move a pawn.
    All choices are sampled uniformly from the set of legal actions.
    """

    def __init__(self) -> None:
        pass  # no internal state needed

    def reset(self) -> None:
        """Called by QuoridorEnv at the start of each episode."""
        pass  # nothing to reset

    def choose_action(self, game: QuoridorState) -> tuple:
        """
        Choose a random legal action for the current player.

        Args:
            game: Current QuoridorState (bot is always P1 inside QuoridorEnv).

        Returns:
            ("move", row, col) or ("fence", row, col, orientation)
        """
        player = game.turn

        # Attempt a random wall placement with probability _WALL_PROB
        if game.walls_left[player] > 0 and random.random() < _WALL_PROB:
            wall = self._random_wall(game)
            if wall is not None:
                return wall

        # Fall back to a random pawn move
        return self._random_move(game)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _random_move(self, game: QuoridorState) -> tuple:
        """Return a uniformly random legal pawn move."""
        destinations = game.get_valid_moves()
        row, col = random.choice(destinations)
        return ("move", row, col)

    def _random_wall(self, game: QuoridorState) -> tuple | None:
        """
        Sample random (row, col, orientation) positions until a legal
        wall placement is found, or return None after _MAX_WALL_TRIES.
        """
        orientations = ("h", "v")
        for _ in range(_MAX_WALL_TRIES):
            row         = random.randrange(FENCE_GRID)
            col         = random.randrange(FENCE_GRID)
            orientation = random.choice(orientations)
            if game._fence_ok(row, col, orientation):
                return ("fence", row, col, orientation)
        return None
