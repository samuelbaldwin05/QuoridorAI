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


    # ── movement ──────────────────────────────────────────────────

    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

    def get_valid_moves(self):
        """
        Return list of (row, col) positions the current player can move to.
        Handles normal steps, straight jumps over opponent, and diagonal
        jumps when straight jump is blocked.
        """
        row, col = int(self.pos[self.turn, 0]), int(self.pos[self.turn, 1])
        opp_r, opp_c = int(self.pos[self.turn ^ 1, 0]), int(self.pos[self.turn ^ 1, 1])
        moves = []

        for dr, dc in self.DIRECTIONS:
            nr, nc = row + dr, col + dc

            if not self._in_bounds(nr, nc):
                continue
            if self._blocked(row, col, nr, nc):
                continue

            # square is empty — normal step
            if nr != opp_r or nc != opp_c:
                moves.append((nr, nc))
                continue

            # opponent is in the way — try straight jump
            jr, jc = nr + dr, nc + dc
            if (self._in_bounds(jr, jc)
                    and not self._blocked(nr, nc, jr, jc)):
                moves.append((jr, jc))
                continue

            # straight jump blocked — try diagonal jumps
            for ddr, ddc in self.DIRECTIONS:
                # skip the direction we came from and the straight-ahead
                if (ddr, ddc) == (dr, dc) or (ddr, ddc) == (-dr, -dc):
                    continue
                diag_r, diag_c = nr + ddr, nc + ddc
                if (self._in_bounds(diag_r, diag_c)
                        and not self._blocked(nr, nc, diag_r, diag_c)
                        and (diag_r != row or diag_c != col)):
                    moves.append((diag_r, diag_c))

        return moves

    def move_to(self, row, col):
        """
        Move the current player to (row, col). Validates the move,
        updates position, checks for win, and switches turn.

        Returns True if this move won the game.
        Raises ValueError if the move is illegal.
        """
        if (row, col) not in self.get_valid_moves():
            raise ValueError(f"illegal move to ({row}, {col})")

        self.pos[self.turn] = [row, col]

        if self.check_win():
            return True

        self.switch_turn()
        return False

    # ── wall collision ──────────────────────────────────────────────

    def _in_bounds(self, r, c):
        return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE

    def _blocked(self, row1, col1, row2, col2):
        """True if a fence blocks the step between two adjacent cells."""
        if row2 > row1:  # stepping down
            return ((col1 > 0 and self.h_walls[row1, col1 - 1])
                    or (col1 < FENCE_GRID and self.h_walls[row1, col1]))
        if row2 < row1:  # stepping up is just the reverse
            return self._blocked(row2, col2, row1, col1)
        if col2 > col1:  # stepping right
            return ((row1 > 0 and self.v_walls[row1 - 1, col1])
                    or (row1 < FENCE_GRID and self.v_walls[row1, col1]))
        # stepping left is the reverse
        return self._blocked(row2, col2, row1, col1)

    def __repr__(self):
        p = "P0" if self.turn == 0 else "P1"
        return f"QuoridorState(turn={p}, walls={list(self.walls_left)}, done={self.done})"