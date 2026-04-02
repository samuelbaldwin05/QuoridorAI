"""
Action encoding for the Quoridor DQN agent.

Maps between integer action indices (0-136) and game action tuples.
Single source of truth for the action space layout:

  0-7:     pawn moves (cardinal + diagonal directions)
  8-71:    horizontal walls (8 + row*8 + col)
  72-135:  vertical walls (72 + row*8 + col)
  136:     pass (reserved, always illegal)

Move actions use direction deltas, not absolute positions. A single
"move up" index covers both 1-step and 2-step jumps (same direction sign).
The env wrapper resolves which destination to actually move to.
"""

NUM_ACTIONS: int = 137
MOVE_ACTION_COUNT: int = 8
H_WALL_OFFSET: int = 8
V_WALL_OFFSET: int = 72
PASS_ACTION: int = 136
FENCE_GRID: int = 8

# Fundamental constants: import from config so this module stays in sync
# if board dimensions or the action space ever change.
from config import NUM_ACTIONS, FENCE_GRID

MOVE_ACTION_COUNT: int = 8      # pawn move actions (cardinal + diagonal)
# Derived offsets — computed from the fundamental constants above so they
# never drift out of sync with NUM_ACTIONS or FENCE_GRID.
H_WALL_OFFSET: int = MOVE_ACTION_COUNT                          # first h-wall index = 8
V_WALL_OFFSET: int = H_WALL_OFFSET + FENCE_GRID * FENCE_GRID   # first v-wall index = 72
PASS_ACTION: int   = V_WALL_OFFSET + FENCE_GRID * FENCE_GRID   # pass index = 136

# Maps pawn action index (0–7) to (row_delta, col_delta).
# Cardinal entries (0–3) encode *direction*: the env wrapper matches any
# destination that is in this direction relative to the current pawn position.
# Diagonal entries (4–7) encode exact one-step diagonal offsets; these only
# arise during jump situations when a straight jump is wall-blocked.
PAWN_DELTAS: dict[int, tuple[int, int]] = {
    0: (-1,  0),   # up
    1: (+1,  0),   # down
    2: ( 0, -1),   # left
    3: ( 0, +1),   # right
    4: (-1, -1),   # NW diagonal
    5: (-1, +1),   # NE diagonal
    6: (+1, -1),   # SW diagonal
    7: (+1, +1),   # SE diagonal
}

# reverse map: (dr, dc) -> index
DELTA_TO_INDEX: dict[tuple[int, int], int] = {v: k for k, v in PAWN_DELTAS.items()}


def action_to_index(action: tuple) -> int:
    """Convert a game action tuple to its integer index."""
    action_type = action[0]

    if action_type == "move":
        _, dr, dc = action
        delta = (dr, dc)
        if delta not in DELTA_TO_INDEX:
            raise ValueError(f"Unknown move delta {delta}")
        return DELTA_TO_INDEX[delta]

    elif action_type == "fence":
        _, row, col, orientation = action
        if not (0 <= row < FENCE_GRID and 0 <= col < FENCE_GRID):
            raise ValueError(f"Fence position ({row}, {col}) out of bounds")
        if orientation == "h":
            return H_WALL_OFFSET + row * FENCE_GRID + col
        elif orientation == "v":
            return V_WALL_OFFSET + row * FENCE_GRID + col
        else:
            raise ValueError(f"Unknown fence orientation '{orientation}'")

    elif action_type == "pass":
        return PASS_ACTION

    else:
        raise ValueError(f"Unknown action type '{action_type}'")


def index_to_action(index: int) -> tuple:
    """Convert an integer action index back to a game action tuple."""
    if not (0 <= index < NUM_ACTIONS):
        raise ValueError(f"Action index {index} out of range [0, {NUM_ACTIONS - 1}]")

    if index < H_WALL_OFFSET:
        dr, dc = PAWN_DELTAS[index]
        return ("move", dr, dc)
    elif index < V_WALL_OFFSET:
        row, col = divmod(index - H_WALL_OFFSET, FENCE_GRID)
        return ("fence", row, col, "h")
    elif index < PASS_ACTION:
        row, col = divmod(index - V_WALL_OFFSET, FENCE_GRID)
        return ("fence", row, col, "v")
    else:
        return ("pass",)
