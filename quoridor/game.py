import numpy as np

BOARD_SIZE = 9
FENCE_GRID = BOARD_SIZE - 1 # 8x8 grid


class QuoridorState:
    """
    Core game state. Tracks player positions, fences on the board,
    remaining fence counts, and whose turn it is.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Initialize or reinitialize to the starting position."""
        # Player 0: starts bottom center (row 8), goal is top (row 0)
        # Player 1: starts top center (row 0), goal is bottom (row 8)
        self.pos = np.array([[8, 4], [0, 4]], dtype=np.int8)
        self.goals = np.array([0, 8], dtype=np.int8)
        self.walls_left = np.array([10, 10], dtype=np.int8)

        # fence grids: True where a fence has been placed
        self.h_walls = np.zeros((FENCE_GRID, FENCE_GRID), dtype=np.bool_)
        self.v_walls = np.zeros((FENCE_GRID, FENCE_GRID), dtype=np.bool_)

        self.turn = 0  # player 0 goes first
        self.done = False
        self.winner = -1

    def clone(self):
        """Fast copy of the full game state. Avoids deepcopy overhead."""
        s = QuoridorState.__new__(QuoridorState)
        s.pos = self.pos.copy()
        s.goals = self.goals  # immutable, no copy needed
        s.walls_left = self.walls_left.copy()
        s.h_walls = self.h_walls.copy()
        s.v_walls = self.v_walls.copy()
        s.turn = self.turn
        s.done = self.done
        s.winner = self.winner
        return s

    def switch_turn(self):
        """Advance to the next player's turn."""
        self.turn = 1 - self.turn

    def check_win(self):
        """Check if the current player has reached their goal row."""
        if self.pos[self.turn, 0] == self.goals[self.turn]:
            self.done = True
            self.winner = self.turn
            return True
        return False

    def __repr__(self):
        p = "P0" if self.turn == 0 else "P1"
        return f"QuoridorState(turn={p}, walls={list(self.walls_left)}, done={self.done})"