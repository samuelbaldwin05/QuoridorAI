"""
eval_bots.py — Batch evaluation: run N games between two bots and report win rate.

Usage:
    python -m scripts.eval_bots --p0 ppo --p0-checkpoint checkpoints/best_self-play-mixed-v1.pt --p1 heuristic --games 100
    python -m scripts.eval_bots --p0 ppo --p0-checkpoint checkpoints/best_strong_shaping.pt --p1 ppo --p1-checkpoint checkpoints/best_pool_v1.pt --games 50
"""

import argparse

from agents.bot import HeuristicBot
from agents.ppo_bot import PPOBot
from agents.random_bot import RandomBot
from quoridor.game import QuoridorState


MAX_PLIES = 300


def make_bot(kind: str, checkpoint: str | None, model: str, greedy: bool):
    if kind == "ppo":
        if not checkpoint:
            raise ValueError("--p0/p1-checkpoint required for ppo bot")
        return PPOBot(checkpoint_path=checkpoint, model_type=model, greedy=greedy)
    elif kind == "heuristic":
        return HeuristicBot()
    elif kind == "random":
        return RandomBot()
    else:
        raise ValueError(f"Unknown bot kind: {kind}")


def play_game(p0, p1) -> tuple[int | None, int]:
    """Play one game. Returns (winner, ply_count)."""
    game = QuoridorState()
    p0.reset()
    p1.reset()
    plies = 0

    while not game.done and plies < MAX_PLIES:
        bot = p0 if game.turn == 0 else p1
        action = bot.choose_action(game)
        if action[0] == "move":
            game.move_to(action[1], action[2])
        else:
            game.place_fence(action[1], action[2], action[3])
        plies += 1

    return (game.winner if game.done else None), plies


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch bot vs bot evaluation.")
    parser.add_argument("--p0", required=True, choices=["ppo", "heuristic", "random"])
    parser.add_argument("--p0-checkpoint", default=None)
    parser.add_argument("--p0-model", default="bfs_resnet", choices=["baseline", "resnet", "bfs", "bfs_resnet"])
    parser.add_argument("--p0-greedy", action="store_true", default=False)
    parser.add_argument("--p1", required=True, choices=["ppo", "heuristic", "random"])
    parser.add_argument("--p1-checkpoint", default=None)
    parser.add_argument("--p1-model", default="bfs_resnet", choices=["baseline", "resnet", "bfs", "bfs_resnet"])
    parser.add_argument("--p1-greedy", action="store_true", default=False)
    parser.add_argument("--games", type=int, default=100)
    args = parser.parse_args()

    p0 = make_bot(args.p0, args.p0_checkpoint, args.p0_model, args.p0_greedy)
    p1 = make_bot(args.p1, args.p1_checkpoint, args.p1_model, args.p1_greedy)

    p0_label = f"{args.p0}" + (f" ({args.p0_checkpoint})" if args.p0_checkpoint else "")
    p1_label = f"{args.p1}" + (f" ({args.p1_checkpoint})" if args.p1_checkpoint else "")
    print(f"P0: {p0_label}")
    print(f"P1: {p1_label}")
    print(f"Games: {args.games}\n")

    p0_wins = 0
    p1_wins = 0
    draws = 0
    total_plies = 0

    for i in range(args.games):
        winner, plies = play_game(p0, p1)
        total_plies += plies
        if winner == 0:
            p0_wins += 1
        elif winner == 1:
            p1_wins += 1
        else:
            draws += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1:3d}/{args.games}]  P0: {p0_wins}  P1: {p1_wins}  Draws: {draws}")

    print(f"\n{'─' * 50}")
    print(f"P0 wins: {p0_wins}/{args.games}  ({100*p0_wins/args.games:.1f}%)")
    print(f"P1 wins: {p1_wins}/{args.games}  ({100*p1_wins/args.games:.1f}%)")
    print(f"Draws:   {draws}/{args.games}  ({100*draws/args.games:.1f}%)")
    print(f"Avg plies/game: {total_plies/args.games:.1f}")


if __name__ == "__main__":
    main()
