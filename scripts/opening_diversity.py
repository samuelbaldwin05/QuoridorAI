"""
opening_diversity.py — Measure how diverse a bot's opening play is.

Runs N games and records the first K moves. If the bot has memorized a few
gambits, most games will share identical opening sequences. If it's playing
general strategy, openings should be diverse.

Reports:
  - Number of unique opening sequences out of N games
  - Most common openings and their frequency
  - Move-by-move entropy (how many distinct choices at each ply)

Usage:
    python -m scripts.opening_diversity --checkpoint checkpoints/best_mixed-v2.pt --games 100 --moves 6
"""

import argparse
from collections import Counter

from agents.bot import HeuristicBot
from agents.ppo_bot import PPOBot
from agents.random_bot import RandomBot
from quoridor.game import QuoridorState


MAX_PLIES = 300


def format_action(action: tuple) -> str:
    """Human-readable action string."""
    if action[0] == "move":
        _, r, c = action
        return f"{chr(c + 97)}{r + 1}"
    else:
        _, r, c, ori = action
        return f"{ori}{chr(c + 97)}{r + 1}"


def record_game(p0, p1, num_moves: int) -> list[str]:
    """Play a game and return the first num_moves actions as strings."""
    game = QuoridorState()
    p0.reset()
    p1.reset()
    actions = []
    plies = 0

    while not game.done and plies < MAX_PLIES and len(actions) < num_moves:
        bot = p0 if game.turn == 0 else p1
        action = bot.choose_action(game)
        who = "P0" if game.turn == 0 else "P1"
        actions.append(f"{who}:{format_action(action)}")

        if action[0] == "move":
            game.move_to(action[1], action[2])
        else:
            game.place_fence(action[1], action[2], action[3])
        plies += 1

    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure opening diversity.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="bfs_resnet",
                        choices=["baseline", "resnet", "bfs", "bfs_resnet"])
    parser.add_argument("--opponent", default="heuristic",
                        choices=["heuristic", "random", "ppo"])
    parser.add_argument("--opponent-checkpoint", default=None)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--moves", type=int, default=6,
                        help="Number of opening moves (plies) to record per game")
    args = parser.parse_args()

    agent = PPOBot(args.checkpoint, model_type=args.model, greedy=False)

    if args.opponent == "heuristic":
        opp = HeuristicBot()
    elif args.opponent == "random":
        opp = RandomBot()
    else:
        opp = PPOBot(args.opponent_checkpoint, model_type=args.model, greedy=False)

    # Collect openings
    openings = []
    for _ in range(args.games):
        seq = record_game(agent, opp, args.moves)
        openings.append(tuple(seq))

    # Analysis
    print(f"\n{'=' * 55}")
    print(f"Opening Diversity Analysis  ({args.games} games, first {args.moves} plies)")
    print(f"{'=' * 55}")

    unique = len(set(openings))
    print(f"\nUnique opening sequences: {unique}/{args.games} "
          f"({100 * unique / args.games:.0f}%)")

    # Most common full sequences
    seq_counts = Counter(openings).most_common(10)
    print(f"\nTop opening sequences:")
    for seq, count in seq_counts:
        pct = 100 * count / args.games
        moves_str = " → ".join(seq)
        print(f"  {count:3d}x ({pct:4.1f}%)  {moves_str}")

    # Per-ply diversity: how many distinct choices at each move?
    print(f"\nPer-ply diversity:")
    for ply in range(args.moves):
        moves_at_ply = [o[ply] for o in openings if len(o) > ply]
        distinct = len(set(moves_at_ply))
        most_common = Counter(moves_at_ply).most_common(3)
        top_str = ", ".join(f"{m} ({100*c/len(moves_at_ply):.0f}%)" for m, c in most_common)
        print(f"  Ply {ply + 1}: {distinct} distinct moves  |  Top: {top_str}")


if __name__ == "__main__":
    main()
