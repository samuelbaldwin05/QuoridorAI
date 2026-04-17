"""
play.py — Play Quoridor against a bot in the terminal.

Usage:
    python -m scripts.play                                          # 2 human players
    python -m scripts.play --opponent heuristic                     # vs HeuristicBot
    python -m scripts.play --opponent ppo --checkpoint checkpoints/best_ppo.pt
"""

import argparse

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


def human_turn(game):
    """Prompt the human for a move. Returns True if the move won the game."""
    valid_moves = game.get_valid_moves()
    prompt = f"P{game.turn} (you)> "

    while True:
        try:
            raw = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            raise SystemExit

        parsed = parse_input(raw)

        if parsed is None:
            print("Invalid input. Examples: 'e5', 'h a3', 'v c2'")
            continue

        if parsed[0] == "quit":
            print("Goodbye!")
            raise SystemExit

        try:
            if parsed[0] == "move":
                _, row, col = parsed
                if (row, col) not in valid_moves:
                    print(f"Can't move there. Valid: {[(chr(c+97)+str(r+1)) for r,c in valid_moves]}")
                    continue
                return game.move_to(row, col)

            elif parsed[0] == "fence":
                _, row, col, orientation = parsed
                game.place_fence(row, col, orientation)
                return False

        except ValueError as e:
            print(f"Illegal: {e}")
            continue


def bot_turn(game, bot):
    """Let the bot take a turn. Returns True if the move won the game."""
    action = bot.choose_action(game)
    if action[0] == "move":
        _, r, c = action
        print(f"P{game.turn} (bot): move to {chr(c+97)}{r+1}")
        return game.move_to(r, c)
    else:
        _, r, c, ori = action
        print(f"P{game.turn} (bot): {ori} fence at {chr(c+97)}{r+1}")
        game.place_fence(r, c, ori)
        return False


def main():
    parser = argparse.ArgumentParser(description="Play Quoridor in the terminal.")
    parser.add_argument(
        "--opponent", default=None,
        choices=["heuristic", "ppo"],
        help="Bot opponent. Omit for 2-player human mode.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="PPO checkpoint path (required with --opponent ppo).",
    )
    parser.add_argument(
        "--model", default="bfs_resnet",
        choices=["baseline", "resnet", "bfs", "bfs_resnet"],
        help="Model architecture for PPO opponent.",
    )
    parser.add_argument(
        "--play-as", default=0, type=int, choices=[0, 1],
        dest="play_as",
        help="Which player you control: 0 (goes first) or 1 (goes second). Default: 0.",
    )
    args = parser.parse_args()

    bot = None
    if args.opponent == "heuristic":
        from agents.bot import HeuristicBot
        bot = HeuristicBot()
    elif args.opponent == "ppo":
        if not args.checkpoint:
            parser.error("--opponent ppo requires --checkpoint")
        from agents.ppo_bot import PPOBot
        bot = PPOBot(
            checkpoint_path=args.checkpoint,
            model_type=args.model,
            greedy=False,
        )

    game = QuoridorState()
    if bot:
        bot.reset()

    print("=== Quoridor ===")
    print("Move: type a cell like 'e5'")
    print("Fence: type 'h a3' or 'v c2'")
    print("Type 'quit' to exit")
    if bot:
        print(f"You are P{args.play_as}. Bot is P{1 - args.play_as}.")
    print()

    while not game.done:
        print(render(game))
        print()

        if bot and game.turn != args.play_as:
            won = bot_turn(game, bot)
        else:
            won = human_turn(game)

        if won:
            break

    print(render(game))
    if game.winner == args.play_as:
        print("\nYou win!")
    elif bot:
        print("\nBot wins!")
    else:
        print(f"\nPlayer {game.winner} wins!")


if __name__ == "__main__":
    main()
