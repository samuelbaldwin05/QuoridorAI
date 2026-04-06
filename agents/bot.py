import random
from collections import deque
from heapq import heappush, heappop

import numpy as np

from quoridor.game import QuoridorState, BOARD_SIZE, FENCE_GRID
from quoridor.action_encoding import index_to_action

"""
Rule-based Quoridor agent with heuristic evaluation.

Uses a priority chain:
    1. Opening book (weighted random lateral moves)
    2. Win-in-one detection
    3. High-impact fence (greedy one-ply lookahead, advantage >= 3)
    4. Strategic fence (direct blocking when opponent is close)
    5. Side fence (vertical blocking near opponent)
    6. Shortest path movement via uniform cost search with
       fence proximity penalties as tiebreakers
"""

# fence proximity penalties for UCS tiebreaking —
# bias toward open space on equidistant paths
FENCE_PENALTIES = {0: 0.10, 1: 0.05, 2: 0.03, 3: 0.01}

OPENING_PATTERNS = [
    {"moves": ["left", "left"], "weight": 2},
    {"moves": ["down", "left"], "weight": 2},
    {"moves": ["down", "right"], "weight": 2},
    {"moves": ["right", "right"], "weight": 2},
    {"moves": ["left", "down", "left"], "weight": 2},
    {"moves": ["right", "down", "right"], "weight": 2},
    {"moves": ["left", "left", "left"], "weight": 1},
    {"moves": ["right", "right", "right"], "weight": 1},
]

DIRECTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


class HeuristicBot:
    """Rule-based agent using a priority chain of heuristics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.move_count = 0
        self.opening_pattern = None
        self.opening_step = 0

    def choose_action(self, game):
        """
        Pick an action for the current player. Returns either:
            ("move", row, col)
            ("fence", row, col, orientation)
        """
        me = game.turn
        opp = me ^ 1

        # 1. opening book (first 1-2 moves, 75% chance)
        if self.move_count <= 1 and random.random() < 0.75:
            action = self._opening_move(game)
            if action:
                self.move_count += 1
                return action

        valid_moves = game.get_valid_moves()
        self.move_count += 1

        # 2. win in one
        goal = int(game.goals[me])
        for r, c in valid_moves:
            if r == goal:
                return ("move", r, c)

        # 3. high-impact fence (advantage increase >= 3)
        if game.walls_left[me] > 0:
            fence = self._find_high_impact_fence(game)
            if fence:
                return fence

        # 4/5. strategic + side fences when opponent is close
        opp_dist = game.shortest_path(opp)
        opp_row_dist = abs(int(game.pos[opp, 0]) - int(game.goals[opp]))

        if game.walls_left[me] > 0 and opp_dist <= 3 and opp_row_dist <= 4:
            fence = self._find_strategic_fence(game)
            if fence:
                return fence
            fence = self._find_side_fence(game)
            if fence:
                return fence

        # 6. uniform cost search toward goal
        best = self._ucs_best_move(game)
        if best:
            return ("move", best[0], best[1])

        # fallback
        r, c = valid_moves[0]
        return ("move", r, c)

    # ── opening book ────────────────────────────────────────────

    def _opening_move(self, game):
        if self.opening_pattern is None:
            total = sum(p["weight"] for p in OPENING_PATTERNS)
            roll = random.random() * total
            for p in OPENING_PATTERNS:
                roll -= p["weight"]
                if roll <= 0:
                    self.opening_pattern = p["moves"]
                    break
            else:
                self.opening_pattern = OPENING_PATTERNS[0]["moves"]
            self.opening_step = 0

        if self.opening_step >= len(self.opening_pattern):
            return None

        direction = self.opening_pattern[self.opening_step]
        dr, dc = DIRECTION_DELTAS[direction]
        r = int(game.pos[game.turn, 0]) + dr
        c = int(game.pos[game.turn, 1]) + dc

        if (r, c) in game.get_valid_moves():
            self.opening_step += 1
            return ("move", r, c)

        # desired move blocked, abandon opening
        self.opening_pattern = None
        return None

    # ── fence search ────────────────────────────────────────────

    def _advantage(self, game):
        """Opponent distance minus our distance. Higher is better for us."""
        me, opp = game.turn, game.turn ^ 1
        return game.shortest_path(opp) - game.shortest_path(me)

    def _find_high_impact_fence(self, game):
        """Greedy one-ply lookahead over fences that cross the opponent's shortest path.

        Why path-guided candidates instead of all valid fences:
            _all_valid_fences() checks all 128 positions with _fence_ok() (2 BFS each),
            then evaluates each valid fence with _advantage() (2 shortest_path each).
            On a mid-game board with ~100 valid fences, that costs ~200+ shortest_path
            calls per bot move.

            A fence only increases the opponent's distance if it crosses an edge they
            actually use. A path of ~10 edges yields ~18 raw candidates → ~10 valid
            after _fence_ok() → ~10 shortest_path calls total (~20x fewer).

        Why cache me_dist:
            Blocking fences almost never lengthen our own path. Precomputing me_dist
            once avoids one redundant shortest_path call per candidate.
        """
        me  = game.turn
        opp = me ^ 1

        opp_path = self._bfs_path(game, opp)
        if len(opp_path) <= 1:
            return None  # opponent already at goal or no path exists

        candidates = self._fences_blocking_path(opp_path)
        if not candidates:
            return None

        # Cache our distance — reused each iteration since blocking fences
        # almost never lengthen our own shortest path.
        me_dist     = game.shortest_path(me)
        opp_dist    = game.shortest_path(opp)
        current_adv = opp_dist - me_dist

        best_fence = None
        best_gain  = 0

        random.shuffle(candidates)

        for row, col, orientation in candidates:
            if not game._fence_ok(row, col, orientation):
                continue

            grid = game.h_walls if orientation == "h" else game.v_walls
            grid[row, col] = True

            # Only recompute opponent distance — our distance treated as stable.
            new_opp_dist = game.shortest_path(opp)
            gain = (new_opp_dist - me_dist) - current_adv

            grid[row, col] = False

            if gain >= 3 and gain > best_gain:
                best_gain  = gain
                best_fence = ("fence", row, col, orientation)

        return best_fence

    def _find_strategic_fence(self, game):
        """Check direct blocking fences in front of opponent."""
        return self._best_fence_from(game, self._direct_blocking_fences(game))

    def _find_side_fence(self, game):
        """Check vertical fences beside opponent."""
        return self._best_fence_from(game, self._side_blocking_fences(game))

    def _best_fence_from(self, game, candidates):
        """From a list of candidate fences, return the one giving best
        advantage increase, or None if none help.
        """
        current_adv = self._advantage(game)
        random.shuffle(candidates)

        best_fence = None
        best_gain = 0

        for row, col, orientation in candidates:
            if not game._fence_ok(row, col, orientation):
                continue

            grid = game.h_walls if orientation == "h" else game.v_walls
            grid[row, col] = True

            gain = self._advantage(game) - current_adv

            grid[row, col] = False

            if gain > best_gain:
                best_gain = gain
                best_fence = ("fence", row, col, orientation)

        return best_fence

    def _direct_blocking_fences(self, game):
        """Horizontal fences directly in front of the opponent."""
        opp = game.turn ^ 1
        opp_r, opp_c = int(game.pos[opp, 0]), int(game.pos[opp, 1])
        goal_dir = -1 if game.goals[opp] < opp_r else 1

        if goal_dir == -1:
            fence_row = opp_r - 1
        else:
            fence_row = opp_r

        fences = []
        for dc in [-1, 0]:
            col = max(0, min(FENCE_GRID - 1, opp_c + dc))
            fences.append((fence_row, col, "h"))
        return fences

    def _side_blocking_fences(self, game):
        """Vertical fences beside and near the opponent."""
        opp = game.turn ^ 1
        opp_r, opp_c = int(game.pos[opp, 0]), int(game.pos[opp, 1])
        fences = []

        for dc in [-1, 0]:
            col = opp_c + dc
            if 0 <= col < FENCE_GRID:
                fences.append((opp_r, col, "v"))
                if opp_r > 0:
                    fences.append((opp_r - 1, col, "v"))

        return fences

    def _all_valid_fences(self, game):
        """Every legal fence placement on the board."""
        fences = []
        for row in range(FENCE_GRID):
            for col in range(FENCE_GRID):
                for orientation in ("h", "v"):
                    if game._fence_ok(row, col, orientation):
                        fences.append((row, col, orientation))
        return fences

    # ── uniform cost search ─────────────────────────────────────

    def _ucs_best_move(self, game):
        """Find the best move by running UCS from each valid move to goal,
        using fence proximity penalties as edge weight tiebreakers.
        """
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        # run UCS from current position, return the first step on the optimal path
        path = self._ucs_path(game)
        if len(path) > 1:
            # path[0] is current pos, path[1] is the next move
            target = path[1]
            if target in valid_moves:
                return target

        # fallback: pick the valid move with smallest BFS distance
        me = game.turn
        goal = int(game.goals[me])
        best, best_dist = None, float("inf")
        for r, c in valid_moves:
            # temporarily move, measure, undo
            old = game.pos[me].copy()
            game.pos[me] = [r, c]
            d = game.shortest_path(me)
            game.pos[me] = old
            if d < best_dist:
                best_dist = d
                best = (r, c)
        return best

    def _ucs_path(self, game):
        """UCS from current player's position to their goal row.
        Edge weights = 1.0 + fence proximity penalty on the target square.
        Returns list of (row, col) from start to goal.
        """
        me = game.turn
        start = (int(game.pos[me, 0]), int(game.pos[me, 1]))
        goal = int(game.goals[me])

        # Pre-compute a fence-distance grid once before the search.
        # Without this, _fence_penalty (an O(FENCE_GRID²) scan) is called
        # for every neighbor of every expanded node — up to ~324 times per
        # UCS call. Building the grid once reduces that to a single O(256+81×n)
        # pass, giving ~80x fewer operations when walls are on the board.
        has_fences = game.fences_count() > 0
        if has_fences:
            fence_dist_grid = self._build_fence_distance_grid(game)

        # priority queue: (cost, row, col)
        pq = [(0.0, start[0], start[1])]
        costs = {start: 0.0}
        came_from = {start: None}

        while pq:
            cost, cr, cc = heappop(pq)

            if cr == goal:
                # reconstruct path
                path = []
                node = (cr, cc)
                while node is not None:
                    path.append(node)
                    node = came_from[node]
                path.reverse()
                return path

            if cost > costs.get((cr, cc), float("inf")):
                continue

            for dr, dc in game.DIRECTIONS:
                nr, nc = cr + dr, cc + dc
                if not game._in_bounds(nr, nc):
                    continue
                if game._blocked(cr, cc, nr, nc):
                    continue

                # O(1) grid lookup replaces the O(FENCE_GRID²) per-call scan
                if has_fences:
                    penalty = FENCE_PENALTIES.get(fence_dist_grid[nr][nc], 0.0)
                else:
                    penalty = 0.0
                edge_cost = 1.0 + penalty
                new_cost = cost + edge_cost

                if new_cost < costs.get((nr, nc), float("inf")):
                    costs[(nr, nc)] = new_cost
                    came_from[(nr, nc)] = (cr, cc)
                    heappush(pq, (new_cost, nr, nc))

        return [start]  # no path found

    def _bfs_path(self, game, player: int) -> list[tuple[int, int]]:
        """BFS shortest path for player from their position to their goal row.

        Returns a list of (row, col) cells from start to the first goal-row cell
        reached (inclusive). Returns [start] if already at goal or no path exists.

        Used by _find_high_impact_fence() to identify which fence positions could
        actually block the opponent — fences not crossing this path cannot increase
        their distance and are skipped.
        """
        start = (int(game.pos[player, 0]), int(game.pos[player, 1]))
        goal  = int(game.goals[player])

        if start[0] == goal:
            return [start]

        visited   = {start}
        came_from = {start: None}
        queue     = deque([start])

        while queue:
            cr, cc = queue.popleft()
            for dr, dc in game.DIRECTIONS:
                nr, nc = cr + dr, cc + dc
                if (game._in_bounds(nr, nc)
                        and (nr, nc) not in visited
                        and not game._blocked(cr, cc, nr, nc)):
                    visited.add((nr, nc))
                    came_from[(nr, nc)] = (cr, cc)
                    if nr == goal:
                        path, node = [], (nr, nc)
                        while node is not None:
                            path.append(node)
                            node = came_from[node]
                        path.reverse()
                        return path
                    queue.append((nr, nc))

        return [start]  # no path (should not occur in a valid game state)

    def _fences_blocking_path(self, path: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
        """Return fence candidates that block at least one edge in path.

        For each consecutive cell pair we identify which fence positions could
        block that step. This follows the same logic as game._blocked():
            vertical move   (same col) → blocked by h-walls at fence_row = min(r1,r2),
                                         fence_cols c1-1 and c1
            horizontal move (same row) → blocked by v-walls at fence_col = min(c1,c2),
                                         fence_rows r1-1 and r1

        Returns a deduplicated list of (row, col, orientation) within FENCE_GRID bounds.
        Candidates may still fail _fence_ok() — filter before placing.
        """
        seen: set[tuple[int, int, str]] = set()
        candidates: list[tuple[int, int, str]] = []

        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]

            if c1 == c2:  # vertical move (up or down) → h-wall blocks it
                fence_row = min(r1, r2)
                for fc in (c1 - 1, c1):
                    if 0 <= fence_row < FENCE_GRID and 0 <= fc < FENCE_GRID:
                        key = (fence_row, fc, "h")
                        if key not in seen:
                            seen.add(key)
                            candidates.append(key)
            else:          # horizontal move (left or right) → v-wall blocks it
                fence_col = min(c1, c2)
                for fr in (r1 - 1, r1):
                    if 0 <= fr < FENCE_GRID and 0 <= fence_col < FENCE_GRID:
                        key = (fr, fence_col, "v")
                        if key not in seen:
                            seen.add(key)
                            candidates.append(key)

        return candidates

    def _build_fence_distance_grid(self, game) -> list[list[int]]:
        """Pre-compute a BOARD_SIZE×BOARD_SIZE grid of Manhattan distances
        from each square to the nearest fence-affected square.

        A fence at grid position (fr, fc) touches the four board squares at
        its corners. We collect all such squares from placed walls, then for
        each of the 81 board cells compute the minimum Manhattan distance to
        any of them. Called once per _ucs_path invocation.
        """
        # Collect unique board squares touched by any placed fence
        fence_squares: set[tuple[int, int]] = set()
        for fr in range(FENCE_GRID):
            for fc in range(FENCE_GRID):
                if game.h_walls[fr, fc] or game.v_walls[fr, fc]:
                    for sq in self._fence_squares(fr, fc):
                        fence_squares.add(sq)

        # For each board cell, find min Manhattan distance to any fence square
        grid: list[list[int]] = []
        for r in range(BOARD_SIZE):
            row: list[int] = []
            for c in range(BOARD_SIZE):
                min_dist = min(
                    abs(r - ar) + abs(c - ac) for ar, ac in fence_squares
                )
                row.append(min_dist)
            grid.append(row)

        return grid

    def _fence_penalty(self, game, row, col):
        """Penalty for a square based on distance to nearest fence.
        Encourages paths through open space when distances are equal.
        """
        if game.fences_count() == 0:
            return 0.0

        min_dist = self._min_fence_distance(game, row, col)
        return FENCE_PENALTIES.get(min_dist, 0.0)

    def _min_fence_distance(self, game, row, col):
        """Manhattan distance from (row, col) to nearest fence-affected square."""
        best = float("inf")

        for fr in range(FENCE_GRID):
            for fc in range(FENCE_GRID):
                if game.h_walls[fr, fc]:
                    for ar, ac in self._fence_squares(fr, fc):
                        best = min(best, abs(row - ar) + abs(col - ac))
                if game.v_walls[fr, fc]:
                    for ar, ac in self._fence_squares(fr, fc):
                        best = min(best, abs(row - ar) + abs(col - ac))

        return best

    @staticmethod
    def _fence_squares(fr, fc) -> list[tuple[int, int]]:
        """The four board squares a fence at grid position (fr, fc) touches."""
        return [
            (fr, fc), (fr, fc + 1),
            (fr + 1, fc), (fr + 1, fc + 1),
        ]


# ---------------------------------------------------------------------------
# ε-greedy curriculum wrapper
# ---------------------------------------------------------------------------

class EpsilonHeuristicBot:
    """
    ε-greedy wrapper around HeuristicBot for curriculum training.

    With probability epsilon, takes a uniformly random legal action.
    Otherwise delegates to the wrapped HeuristicBot unchanged.

    This gives the PPO agent a stream of winnable games early in training when
    it would otherwise never beat a full-strength HeuristicBot — bootstrapping
    the positive reward signal that PPO needs to learn. As the agent improves,
    epsilon is annealed toward 0 by the training loop, gradually restoring
    full-strength opposition.

    The wrapped HeuristicBot is not modified — EpsilonHeuristicBot is a pure
    proxy. Swapping the epsilon attribute is sufficient to update the difficulty
    without rebuilding the environment.

    Parameters
    ----------
    epsilon : float
        Initial probability of taking a random action in [0, 1].
        Typically starts at EPSILON_CURRICULUM_START (0.3) and decays to 0
        as the agent's rolling win rate approaches EPSILON_CURRICULUM_THRESHOLD.
    """

    def __init__(self, epsilon: float = 0.3) -> None:
        self._heuristic = HeuristicBot()
        self.epsilon    = epsilon  # mutable — updated in-place by training loop

    def reset(self) -> None:
        """Delegate to wrapped HeuristicBot."""
        self._heuristic.reset()

    def choose_action(self, game) -> tuple:
        """
        Choose an action for the current player.

        With probability epsilon: uniformly random legal action.
        Otherwise: HeuristicBot's action.

        The random fallback uses the legal mask to sample only valid actions,
        then resolves direction deltas for pawn moves to absolute coordinates —
        the same logic used by env._apply_index() and PPOBot._decode().

        Returns
        -------
        tuple
            Either ("move", row, col) or ("fence", row, col, orientation).
        """
        if random.random() < self.epsilon:
            return self._random_legal_action(game)
        return self._heuristic.choose_action(game)

    def _random_legal_action(self, game) -> tuple:
        """Sample uniformly from all legal actions and return an absolute action tuple."""
        legal_mask    = game.get_legal_mask()
        legal_indices = np.where(legal_mask)[0]
        idx           = int(np.random.choice(legal_indices))
        action        = index_to_action(idx)

        if action[0] != "move":
            return action  # fence/pass — already in absolute form

        # Resolve direction delta → absolute board coordinates.
        # index_to_action returns ("move", dr, dc); the env and all bots expect
        # ("move", dest_r, dest_c) with absolute coordinates. Scan get_valid_moves()
        # for any destination whose direction sign matches the decoded delta.
        dr, dc = action[1], action[2]
        cur_r  = int(game.pos[game.turn, 0])
        cur_c  = int(game.pos[game.turn, 1])

        for dest_r, dest_c in game.get_valid_moves():
            if (np.sign(dest_r - cur_r) == np.sign(dr)
                    and np.sign(dest_c - cur_c) == np.sign(dc)):
                return ("move", dest_r, dest_c)

        # Fallback: should not occur if the legal mask is correct.
        dest_r, dest_c = game.get_valid_moves()[0]
        return ("move", dest_r, dest_c)