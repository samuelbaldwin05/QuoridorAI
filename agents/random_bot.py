"""
Random opponent for curriculum training (phase 1).

Beatable enough that the agent gets some early wins in the replay buffer.
Drop-in replacement for HeuristicBot in QuoridorEnv.
"""

import random

from quoridor.game import QuoridorState, FENCE_GRID

_WALL_PROB = 0.30
_MAX_WALL_TRIES = 20


class RandomBot:
    """Uniformly random Quoridor opponent."""

    def __init__(self):
        pass

    def reset(self):
        pass

    def choose_action(self, game: QuoridorState) -> tuple:
        """Pick a random legal action. 30% chance wall, 70% pawn move."""
        player = game.turn

        if game.walls_left[player] > 0 and random.random() < _WALL_PROB:
            wall = self._random_wall(game)
            if wall is not None:
                return wall

        return self._random_move(game)

    def _random_move(self, game):
        destinations = game.get_valid_moves()
        row, col = random.choice(destinations)
        return ("move", row, col)

    def _random_wall(self, game):
        orientations = ("h", "v")
        for _ in range(_MAX_WALL_TRIES):
            row = random.randrange(FENCE_GRID)
            col = random.randrange(FENCE_GRID)
            orientation = random.choice(orientations)
            if game._fence_ok(row, col, orientation):
                return ("fence", row, col, orientation)
        return None
