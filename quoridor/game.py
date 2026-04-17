import numpy as np
from collections import deque

from config import BOARD_SIZE, FENCE_GRID, INITIAL_WALLS_PER_PLAYER, BFS_NORM_FACTOR


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
        self.walls_left = np.array([INITIAL_WALLS_PER_PLAYER, INITIAL_WALLS_PER_PLAYER], dtype=np.int8)

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


    # MOVEMENT

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

    # FENCE PLACEMENT

    def place_fence(self, row, col, orientation):
        """
        Place a fence at grid position (row, col) with given orientation
        ('h' for horizontal, 'v' for vertical). Validates placement,
        decrements wall count, and switches turn.

        Raises ValueError if the placement is illegal.
        """
        if not self._fence_ok(row, col, orientation):
            raise ValueError(
                f"illegal fence: ({row}, {col}, {orientation})"
            )

        grid = self.h_walls if orientation == "h" else self.v_walls
        grid[row, col] = True
        self.walls_left[self.turn] -= 1
        self.switch_turn()

    def _fence_ok(self, row, col, orientation):
        """Check whether a fence placement is legal."""
        # must have fences remaining
        if self.walls_left[self.turn] <= 0:
            return False

        # must be within the 8x8 fence grid
        if not (0 <= row < FENCE_GRID and 0 <= col < FENCE_GRID):
            return False

        if orientation == "h":
            grid = self.h_walls
        else:
            grid = self.v_walls

        # no duplicate
        if grid[row, col]:
            return False

        # no overlap with same-orientation neighbor (they share a post)
        if orientation == "h":
            if (col > 0 and self.h_walls[row, col - 1]):
                return False
            if (col < FENCE_GRID - 1 and self.h_walls[row, col + 1]):
                return False
        else:
            if (row > 0 and self.v_walls[row - 1, col]):
                return False
            if (row < FENCE_GRID - 1 and self.v_walls[row + 1, col]):
                return False

        # no crossing with perpendicular fence at same post
        if orientation == "h":
            other = self.v_walls
        else:
            other = self.h_walls
        if other[row, col]:
            return False

        # must not cut off either player's path to their goal
        grid[row, col] = True
        paths_ok = self._has_path(0) and self._has_path(1)
        grid[row, col] = False
        return paths_ok

    def shortest_path(self, player):
        """BFS shortest distance from a player's position to their goal row.
        Returns int distance, or float('inf') if no path exists.
        """
        start = (int(self.pos[player, 0]), int(self.pos[player, 1]))
        goal = int(self.goals[player])

        if start[0] == goal:
            return 0

        visited = set()
        visited.add(start)
        queue = deque([(start[0], start[1], 0)])

        while queue:
            cr, cc, dist = queue.popleft()
            for dr, dc in self.DIRECTIONS:
                nr, nc = cr + dr, cc + dc
                if (self._in_bounds(nr, nc)
                        and (nr, nc) not in visited
                        and not self._blocked(cr, cc, nr, nc)):
                    if nr == goal:
                        return dist + 1
                    visited.add((nr, nc))
                    queue.append((nr, nc, dist + 1))

        return float("inf")

    def bfs_distance_map(self, player: int) -> np.ndarray:
        """
        Compute a full 9×9 BFS distance map for a player.

        Each cell (r, c) contains the minimum number of steps to reach the
        player's goal row from that position, given the current wall layout.
        Unreachable cells are assigned a distance of 1.0 (the normalised max).

        Uses reverse multi-source BFS: start simultaneously from every cell on
        the goal row (distance 0) and expand outward. One BFS pass fills the
        entire board, which is more efficient than running a separate forward BFS
        from each cell individually.

        The same _blocked() and _in_bounds() helpers used by shortest_path() are
        reused here, so wall semantics are guaranteed to be consistent.

        Returns
        -------
        dist_map : np.ndarray, shape (9, 9), dtype float32
            Values in [0, 1], normalised by BFS_NORM_FACTOR (= BOARD_SIZE * 2 = 18).
        """
        goal_row = int(self.goals[player])
        dist = np.full((BOARD_SIZE, BOARD_SIZE), BFS_NORM_FACTOR, dtype=np.float32)

        # Seed the BFS with all cells on the goal row at distance 0.
        queue: deque = deque()
        for c in range(BOARD_SIZE):
            dist[goal_row, c] = 0.0
            queue.append((goal_row, c))

        # Expand outward. We move BACKWARDS through the board (from goal to
        # source), so the direction of travel is reversed relative to how a
        # player would actually walk. The wall blocking check must therefore
        # be applied as if stepping FROM the neighbour TO the current cell,
        # because _blocked(r1,c1,r2,c2) checks the wall between r1,c1→r2,c2.
        while queue:
            cr, cc = queue.popleft()
            for dr, dc in self.DIRECTIONS:
                nr, nc = cr + dr, cc + dc
                if not self._in_bounds(nr, nc):
                    continue
                # Check whether a player at (nr, nc) can step to (cr, cc).
                # _blocked is symmetric for orthogonal steps, so this is correct.
                if self._blocked(nr, nc, cr, cc):
                    continue
                new_dist = dist[cr, cc] + 1.0
                if new_dist < dist[nr, nc]:
                    dist[nr, nc] = new_dist
                    queue.append((nr, nc))

        # Normalise to [0, 1]. Cells that remained at BFS_NORM_FACTOR normalise
        # to 1.0, which correctly signals "far from goal / possibly cut off".
        return dist / BFS_NORM_FACTOR

    def _has_path(self, player):
        """Whether a player can reach their goal row."""
        return self.shortest_path(player) < float("inf")
    
    def fences_count(self):
        """Total number of fences on the board."""
        return int(self.h_walls.sum() + self.v_walls.sum())
    
    # WALL COLISIONS

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

    # OBSERVATION AND LEGAL MASK

    def get_observation(self, use_bfs: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Encode the current state as a (spatial, scalars) pair for the neural network.

        Parameters
        ----------
        use_bfs : bool, default False
            When True, two additional BFS distance map channels are appended:
              ch4 : current player's BFS distance map (normalised, goal row = 0)
              ch5 : opponent's BFS distance map (normalised, goal row = 0)
            Returns spatial shape (6, 9, 9) instead of (4, 9, 9).
            The same perspective flip applied to pawn/wall channels is applied
            to the BFS maps so the observation is always from the current player's POV.

        Returns
        -------
        spatial : np.ndarray, shape (4, 9, 9) or (6, 9, 9), dtype float32
            Board channels (see channel layout below).
        scalars : np.ndarray, shape (2,), dtype float32
            Normalised wall counts: [current_player_walls / 10, opponent_walls / 10].

        Channel layout (base 4 channels)
        ---------------------------------
        ch0 : current player pawn position (single 1.0)
        ch1 : opponent pawn position (single 1.0)
        ch2 : horizontal wall grid (8×8 in top-left of 9×9, rest 0)
        ch3 : vertical wall grid  (8×8 in top-left of 9×9, rest 0)

        Additional channels when use_bfs=True
        --------------------------------------
        ch4 : current player BFS distance map — each cell holds normalised steps
              to goal from that position; goal row = 0.0, further = higher value.
        ch5 : opponent BFS distance map (same encoding, same perspective flip).

        Perspective normalisation
        -------------------------
        The board is always shown from the current player's POV with their goal
        at row 0.  Player 0 already moves toward row 0, so no flip is needed.
        Player 1's turn flips the board vertically so their pawn appears at the
        bottom and the goal stays at the top.  This makes the observation
        semantically identical from either player's perspective.
        """
        current = self.turn
        opponent = 1 - current
        flip = (current == 1)

        n_channels = 6 if use_bfs else 4
        spatial = np.zeros((n_channels, 9, 9), dtype=np.float32)

        cur_r, cur_c = int(self.pos[current, 0]), int(self.pos[current, 1])
        opp_r, opp_c = int(self.pos[opponent, 0]), int(self.pos[opponent, 1])
        if flip:
            # Flip row coordinates: row i → row 8-i
            cur_r = 8 - cur_r
            opp_r = 8 - opp_r

        spatial[0, cur_r, cur_c] = 1.0  # current player pawn
        spatial[1, opp_r, opp_c] = 1.0  # opponent pawn

        h = self.h_walls.astype(np.float32)  # (8, 8)
        v = self.v_walls.astype(np.float32)  # (8, 8)
        if flip:
            # Flip wall grids vertically so they align with the flipped pawn positions
            h = np.flipud(h)
            v = np.flipud(v)
        spatial[2, :8, :8] = h  # embed 8×8 in top-left of 9×9
        spatial[3, :8, :8] = v

        if use_bfs:
            # Compute full-board BFS distance maps for both players.
            # bfs_distance_map() uses the actual game state (walls, positions) so
            # it correctly reflects the current strategic situation for each player.
            cur_dist = self.bfs_distance_map(current)   # (9, 9)
            opp_dist = self.bfs_distance_map(opponent)  # (9, 9)
            if flip:
                # Apply the same vertical flip as pawn/wall channels so ch4/ch5
                # are always oriented from the current player's perspective.
                cur_dist = np.flipud(cur_dist)
                opp_dist = np.flipud(opp_dist)
            spatial[4] = cur_dist  # ch4: my distance to goal
            spatial[5] = opp_dist  # ch5: opponent's distance to goal

        scalars = np.array([
            self.walls_left[current] / INITIAL_WALLS_PER_PLAYER,
            self.walls_left[opponent] / INITIAL_WALLS_PER_PLAYER,
        ], dtype=np.float32)  # (2,)

        return spatial, scalars

    def get_legal_mask(self) -> np.ndarray:
        """
        Build a boolean mask over the full 137-action space.

        Returns
        -------
        mask : np.ndarray, shape (137,), dtype bool
            mask[i] is True iff action i is currently legal for the active player.

        Pawn moves (indices 0–7) are derived from get_valid_moves() by collapsing
        each destination to its direction sign, then looking up DELTA_TO_INDEX.
        This handles both 1-step and 2-step straight jumps (same sign) and
        diagonal jumps (exact ±1 diagonal sign).

        Wall actions (indices 8–135) are True iff _fence_ok() passes.
        Pass (index 136) is always False — pass is never legal in standard Quoridor.
        """
        # Local import avoids a circular dependency at module load time.
        # action_encoding imports nothing from game.py, so this is safe.
        from quoridor.action_encoding import (
            NUM_ACTIONS, FENCE_GRID as _FG, DELTA_TO_INDEX,
            H_WALL_OFFSET, V_WALL_OFFSET,
        )
        mask = np.zeros(NUM_ACTIONS, dtype=bool)

        cur_r, cur_c = int(self.pos[self.turn, 0]), int(self.pos[self.turn, 1])
        for dest_r, dest_c in self.get_valid_moves():
            dr, dc = dest_r - cur_r, dest_c - cur_c
            # Collapse to direction sign so 1-step and 2-step straight jumps
            # both map to the same cardinal action index.
            normalized = (int(np.sign(dr)), int(np.sign(dc)))
            if normalized in DELTA_TO_INDEX:
                mask[DELTA_TO_INDEX[normalized]] = True

        for r in range(_FG):
            for c in range(_FG):
                if self._fence_ok(r, c, "h"):
                    mask[H_WALL_OFFSET + r * _FG + c] = True
                if self._fence_ok(r, c, "v"):
                    mask[V_WALL_OFFSET + r * _FG + c] = True

        # index 136 (pass) stays False — pass is always illegal
        return mask

    def __repr__(self):
        p = "P0" if self.turn == 0 else "P1"
        return f"QuoridorState(turn={p}, walls={list(self.walls_left)}, done={self.done})"
    
    def __str__(self):
        from quoridor.display import render
        return render(self)