def render(state):
    """
    Return a string showing the board, players (1/2), and fences.
    Fences shown as ─── (horizontal) and │ (vertical).
    """
    lines = []

    # header
    lines.append(
        f"Turn: P{state.turn}  |  "
        f"Walls: P0={state.walls_left[0]} P1={state.walls_left[1]}"
    )
    lines.append("")

    # column labels
    lines.append("    " + "   ".join(chr(ord("a") + i) for i in range(9)))

    for row in range(9):
        # --- cell row ---
        parts = []
        for col in range(9):
            if state.pos[0, 0] == row and state.pos[0, 1] == col:
                parts.append(" 1 ")
            elif state.pos[1, 0] == row and state.pos[1, 1] == col:
                parts.append(" 2 ")
            else:
                parts.append(" . ")

            # vertical fence between this cell and the next
            if col < 8:
                if state._blocked(row, col, row, col + 1):
                    parts.append("│")
                else:
                    parts.append(" ")

        lines.append(f" {row + 1}  {''.join(parts)}")

        # --- fence row between this row and the next ---
        if row < 8:
            parts = []
            for col in range(9):
                if state._blocked(row, col, row + 1, col):
                    parts.append("───")
                else:
                    parts.append("   ")

                if col < 8:
                    parts.append(" ")

            lines.append(f"    {''.join(parts)}")

    return "\n".join(lines)