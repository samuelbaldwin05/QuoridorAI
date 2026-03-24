"""
Action encoding for the Quoridor DQN agent.

This module defines the canonical bijection between integer action indices (0–136)
and game actions. Every component that touches actions — the Gym env wrapper, the
legal mask, the replay buffer, and the DQN output head — must use this module as
the single source of truth for action encoding.

---------------------------------------------------------------------
Full action space layout (137 actions total)
---------------------------------------------------------------------

  Index    Type            Detail
  -------  --------------  -----------------------------------------------
  0        Move up         direction delta (-1,  0); covers 1-step and jump
  1        Move down       direction delta (+1,  0)
  2        Move left       direction delta ( 0, -1)
  3        Move right      direction delta ( 0, +1)
  4        Diagonal NW     direction delta (-1, -1)  ← diagonal jump only (occasionally masked)
  5        Diagonal NE     direction delta (-1, +1)
  6        Diagonal SW     direction delta (+1, -1)
  7        Diagonal SE     direction delta (+1, +1)
  8 – 71   Horizontal wall 8  + row*8 + col  (row, col ∈ 0..7)
  72 – 135 Vertical wall   72 + row*8 + col  (row, col ∈ 0..7)
  136      Pass            Reserved; always masked as illegal

---------------------------------------------------------------------
Why 8 pawn actions instead of 4
---------------------------------------------------------------------

Standard Quoridor allows "jump" moves: when the two pawns are adjacent,
the current player may jump over the opponent. If a wall blocks the straight
jump, diagonal jumps to either side become legal instead. `get_valid_moves()`
can therefore return diagonal destinations that are NOT reachable by any
combination of the four cardinal moves. Encoding only 4 directions would
make those destinations unrepresentable.

The 8-action design uses direction *deltas*, not absolute destinations,
so a single "move up" index covers both a 1-step advance (-1, 0) and a
2-step straight jump (-2, 0). The environment wrapper disambiguates by
scanning `get_valid_moves()` for any destination in the requested direction.

---------------------------------------------------------------------
Why cardinal deltas encode both 1-step and straight-jump moves
---------------------------------------------------------------------

From the pawn's current row r, "move up" could land on row r-1 (normal)
or row r-2 (straight jump over an adjacent opponent). Both destinations
share the same direction sign (-row). Encoding the *direction* rather
than the *offset* keeps the action space compact and avoids needing
separate indices for "up 1" and "up 2".

---------------------------------------------------------------------
Usage example
---------------------------------------------------------------------

    from quoridor.action_encoding import action_to_index, index_to_action

    idx = action_to_index(("fence", 3, 5, "v"))  # → 97
    act = index_to_action(97)                     # → ("fence", 3, 5, "v")

    move_idx = action_to_index(("move", -1, 0))  # → 0  (up)
    pass_idx = action_to_index(("pass",))         # → 136

IMPORTANT: move actions use direction *deltas*, not absolute board positions.
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_ACTIONS: int = 137          # total size of the action space
MOVE_ACTION_COUNT: int = 8      # pawn move actions (cardinal + diagonal)
H_WALL_OFFSET: int = 8          # index of the first horizontal wall action
V_WALL_OFFSET: int = 72         # index of the first vertical wall action
PASS_ACTION: int = 136          # reserved pass action; always masked illegal
FENCE_GRID: int = 8             # fence placement grid is 8×8

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
    4: (-1, -1),   # NW diagonal jump
    5: (-1, +1),   # NE diagonal jump
    6: (+1, -1),   # SW diagonal jump
    7: (+1, +1),   # SE diagonal jump
}

# Reverse map: (row_delta, col_delta) → pawn action index.
# Built automatically from PAWN_DELTAS so the two dicts never diverge.
DELTA_TO_INDEX: dict[tuple[int, int], int] = {v: k for k, v in PAWN_DELTAS.items()}


# ---------------------------------------------------------------------------
# Encoding functions
# ---------------------------------------------------------------------------

def action_to_index(action: tuple) -> int:
    """Convert a game action tuple to its integer index in [0, NUM_ACTIONS).

    Parameters
    ----------
    action:
        One of:
          - ``("move", dr, dc)``          where (dr, dc) is a direction delta
          - ``("fence", row, col, "h")``   horizontal wall placement
          - ``("fence", row, col, "v")``   vertical wall placement
          - ``("pass",)``                  reserved pass move

    Returns
        The unique integer index for this action.

    ValueError
        If the action type is unrecognised, the delta has no assigned index,
        or a fence coordinate is out of the 8×8 placement grid.
    """
    action_type = action[0]

    if action_type == "move":
        # Unpack the direction delta and look it up.
        # We do NOT accept arbitrary deltas — only those in PAWN_DELTAS.
        _, dr, dc = action
        delta = (dr, dc)
        if delta not in DELTA_TO_INDEX:
            raise ValueError(
                f"Unknown move delta {delta}. "
                f"Valid deltas: {list(DELTA_TO_INDEX.keys())}"
            )
        return DELTA_TO_INDEX[delta]

    elif action_type == "fence":
        _, row, col, orientation = action
        # Fence grid is 8×8 (indices 0..7 on each axis).
        if not (0 <= row < FENCE_GRID and 0 <= col < FENCE_GRID):
            raise ValueError(
                f"Fence position ({row}, {col}) out of bounds. "
                f"Both row and col must be in [0, {FENCE_GRID - 1}]."
            )
        if orientation == "h":
            return H_WALL_OFFSET + row * FENCE_GRID + col
        elif orientation == "v":
            return V_WALL_OFFSET + row * FENCE_GRID + col
        else:
            raise ValueError(
                f"Unknown fence orientation '{orientation}'. Expected 'h' or 'v'."
            )

    elif action_type == "pass":
        return PASS_ACTION

    else:
        raise ValueError(
            f"Unknown action type '{action_type}'. "
            f"Expected 'move', 'fence', or 'pass'."
        )


def index_to_action(index: int) -> tuple:
    """Convert an integer action index back to its game action tuple.

    Parameters
    ----------
    index:
        An integer in [0, NUM_ACTIONS).

    Returns
    -------
    tuple
        The game action corresponding to this index. Format matches the
        input format of ``action_to_index``.

    Raises
    ------
    ValueError
        If ``index`` is outside [0, NUM_ACTIONS).
    """
    if not (0 <= index < NUM_ACTIONS):
        raise ValueError(
            f"Action index {index} out of range. "
            f"Valid range: [0, {NUM_ACTIONS - 1}]."
        )

    if index < H_WALL_OFFSET:
        # Pawn move (indices 0–7)
        dr, dc = PAWN_DELTAS[index]
        return ("move", dr, dc)

    elif index < V_WALL_OFFSET:
        # Horizontal wall (indices 8–71)
        row, col = divmod(index - H_WALL_OFFSET, FENCE_GRID)
        return ("fence", row, col, "h")

    elif index < PASS_ACTION:
        # Vertical wall (indices 72–135)
        row, col = divmod(index - V_WALL_OFFSET, FENCE_GRID)
        return ("fence", row, col, "v")

    else:
        # Pass (index 136)
        return ("pass",)
