"""Two-player Quoridor in the terminal."""

from quoridor.game import QuoridorState
from quoridor.display import render


def parse_input(text):
    """
    Parse player input. Accepts:
        e5      -> move to column e, row 5
        h a3    -> horizontal fence at column a, row 3
        v c2    -> vertical fence at column c, row 2
        quit    -> exit
    """
    text = text.strip().lower()

    if text == "quit":
        return ("quit",)

    parts = text.split()

    # fence placement: "h a3" or "v a3"
    if len(parts) == 2 and parts[0] in ("h", "v"):
        orientation = parts[0]
        coord = parts[1]
        if len(coord) >= 2 and coord[0].isalpha() and coord[1:].isdigit():
            col = ord(coord[0]) - ord("a")
            row = int(coord[1:]) - 1
            return ("fence", row, col, orientation)

    # movement: "e5"
    if len(parts) == 1:
        coord = parts[0]
        if len(coord) >= 2 and coord[0].isalpha() and coord[1:].isdigit():
            col = ord(coord[0]) - ord("a")
            row = int(coord[1:]) - 1
            return ("move", row, col)

    return None


def main():
    game = QuoridorState()

    print("=== Quoridor (2 Player) ===")
    print("Move: type a cell like 'e5'")
    print("Fence: type 'h a3' or 'v c2'")
    print("Type 'quit' to exit")
    print()

    while not game.done:
        print(render(game))
        print()

        valid_moves = game.get_valid_moves()
        prompt = f"P{game.turn}> "

        try:
            raw = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return

        parsed = parse_input(raw)

        if parsed is None:
            print("Invalid input. Examples: 'e5', 'h a3', 'v c2'")
            continue

        if parsed[0] == "quit":
            print("Goodbye!")
            return

        try:
            if parsed[0] == "move":
                _, row, col = parsed
                if (row, col) not in valid_moves:
                    print(f"Can't move there. Valid: {[(chr(c+97)+str(r+1)) for r,c in valid_moves]}")
                    continue
                won = game.move_to(row, col)

            elif parsed[0] == "fence":
                _, row, col, orientation = parsed
                game.place_fence(row, col, orientation)
                won = False

        except ValueError as e:
            print(f"Illegal: {e}")
            continue

        if won:
            print(render(game))
            print(f"\nPlayer {game.winner} wins!")
            return

    print(render(game))
    print(f"\nPlayer {game.winner} wins!")


if __name__ == "__main__":
    main()