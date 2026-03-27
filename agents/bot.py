import random
from heapq import heappush, heappop

from quoridor.game import QuoridorState, BOARD_SIZE, FENCE_GRID

"""
Quoridor bot hierarchy for DQN curriculum training.

BaseBot - win-in-one, UCS path, fence candidate generators
RandomFenceBot - UCS + random fence from blocking candidates
HeuristicBot - full priority chain with advantage evaluation
"""

# fence proximity penalties for UCS tiebreaking
FENCE_PENALTIES = {0: 0.10, 1: 0.05, 2: 0.03, 3: 0.01}

DIRECTION_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

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


class BaseBot:
    """Shared logic: win detection, UCS pathfinding, fence candidates."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.move_count = 0

    # win detection

    def _find_winning_move(self, game):
        goal = int(game.goals[game.turn])
        for r, c in game.get_valid_moves():
            if r == goal:
                return ("move", r, c)
        return None

    # advantage

    def _advantage(self, game):
        """Opponent distance minus our distance. Higher = better for us."""
        me, opp = game.turn, game.turn ^ 1
        return game.shortest_path(opp) - game.shortest_path(me)

    # fence candidates

    def _direct_blocking_fences(self, game):
        """Horizontal fences directly in front of the opponent."""
        opp = game.turn ^ 1
        opp_r, opp_c = int(game.pos[opp, 0]), int(game.pos[opp, 1])
        goal_dir = -1 if game.goals[opp] < opp_r else 1
        fence_row = opp_r - 1 if goal_dir == -1 else opp_r

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

    # UCS pathfinding

    def _ucs_best_move(self, game):
        """Best move via UCS, with BFS fallback."""
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return None

        path = self._ucs_path(game)
        if len(path) > 1:
            target = path[1]
            if target in valid_moves:
                return target

        me = game.turn
        goal = int(game.goals[me])
        best, best_dist = None, float("inf")
        for r, c in valid_moves:
            old = game.pos[me].copy()
            game.pos[me] = [r, c]
            d = game.shortest_path(me)
            game.pos[me] = old
            if d < best_dist:
                best_dist = d
                best = (r, c)
        return best

    def _ucs_path(self, game):
        """UCS from current position to goal row with fence proximity tiebreaking."""
        me = game.turn
        start = (int(game.pos[me, 0]), int(game.pos[me, 1]))
        goal = int(game.goals[me])

        pq = [(0.0, start[0], start[1])]
        costs = {start: 0.0}
        came_from = {start: None}

        while pq:
            cost, cr, cc = heappop(pq)

            if cr == goal:
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

                edge_cost = 1.0 + self._fence_penalty(game, nr, nc)
                new_cost = cost + edge_cost

                if new_cost < costs.get((nr, nc), float("inf")):
                    costs[(nr, nc)] = new_cost
                    came_from[(nr, nc)] = (cr, cc)
                    heappush(pq, (new_cost, nr, nc))

        return [start]

    def _fence_penalty(self, game, row, col):
        if game.fences_count() == 0:
            return 0.0
        min_dist = self._min_fence_distance(game, row, col)
        return FENCE_PENALTIES.get(min_dist, 0.0)

    def _min_fence_distance(self, game, row, col):
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
    def _fence_squares(fr, fc):
        return [(fr, fc), (fr, fc + 1), (fr + 1, fc), (fr + 1, fc + 1)]


class RandomFenceBot(BaseBot):
    """UCS movement + random fence placement from blocking candidates.

    On each turn:
        1. Win-in-one
        2. With probability `fence_chance`, attempt a random blocking fence
           (direct first, then side). No advantage evaluation.
        3. UCS movement
    """

    def __init__(self, fence_chance=0.35):
        super().__init__()
        self.fence_chance = fence_chance

    def choose_action(self, game):
        self.move_count += 1

        # 1. win in one
        win = self._find_winning_move(game)
        if win:
            return win

        # 2. random blocking fence
        if game.walls_left[game.turn] > 0 and random.random() < self.fence_chance:
            fence = self._random_blocking_fence(game)
            if fence:
                return fence

        # 3. UCS movement
        best = self._ucs_best_move(game)
        if best:
            return ("move", best[0], best[1])

        r, c = game.get_valid_moves()[0]
        return ("move", r, c)

    def _random_blocking_fence(self, game):
        """Try a random direct-blocking fence, fall back to side-blocking."""
        candidates = self._direct_blocking_fences(game)
        random.shuffle(candidates)
        for row, col, orientation in candidates:
            if game._fence_ok(row, col, orientation):
                return ("fence", row, col, orientation)

        candidates = self._side_blocking_fences(game)
        random.shuffle(candidates)
        for row, col, orientation in candidates:
            if game._fence_ok(row, col, orientation):
                return ("fence", row, col, orientation)

        return None


class HeuristicBot(BaseBot):
    """Full priority chain: opening book, win-in-one, high-impact fence,
    strategic fence, side fence, UCS movement.
    """

    def reset(self):
        super().reset()
        self.opening_pattern = None
        self.opening_step = 0

    def choose_action(self, game):
        me = game.turn
        opp = me ^ 1

        # 1. opening book
        if self.move_count <= 1 and random.random() < 0.75:
            action = self._opening_move(game)
            if action:
                self.move_count += 1
                return action

        self.move_count += 1

        # 2. win in one
        win = self._find_winning_move(game)
        if win:
            return win

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

        # 6. UCS movement
        best = self._ucs_best_move(game)
        if best:
            return ("move", best[0], best[1])

        r, c = game.get_valid_moves()[0]
        return ("move", r, c)

    # opening book

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

        self.opening_pattern = None
        return None

    # fence evaluation

    def _find_high_impact_fence(self, game):
        """Greedy one-ply: returns fence with largest advantage gain, if >= 3."""
        current_adv = self._advantage(game)
        candidates = self._all_valid_fences(game)
        random.shuffle(candidates)

        best_fence = None
        best_gain = 0

        for row, col, orientation in candidates:
            grid = game.h_walls if orientation == "h" else game.v_walls
            grid[row, col] = True
            gain = self._advantage(game) - current_adv
            grid[row, col] = False

            if gain >= 3 and gain > best_gain:
                best_gain = gain
                best_fence = ("fence", row, col, orientation)

        return best_fence

    def _find_strategic_fence(self, game):
        return self._best_fence_from(game, self._direct_blocking_fences(game))

    def _find_side_fence(self, game):
        return self._best_fence_from(game, self._side_blocking_fences(game))

    def _best_fence_from(self, game, candidates):
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