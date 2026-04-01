"""
watch_ppo.py — Watch a trained PPO agent play Quoridor in the terminal.

Loads a checkpoint, plays one or more complete games against a chosen
opponent, and renders each board state with a configurable delay. Useful
for quickly eyeballing whether the agent plays sensibly — does it advance
its pawn? Does it place walls that actually obstruct the opponent?

Usage:
    python -m scripts.watch_ppo --checkpoint checkpoints/ppo_best.pt
    python -m scripts.watch_ppo --checkpoint checkpoints/ppo_best.pt --model bfs_resnet
    python -m scripts.watch_ppo --checkpoint checkpoints/ppo_best.pt --games 3 --delay 0.5
    python -m scripts.watch_ppo --checkpoint checkpoints/ppo_best.pt --opponent random --no-greedy
"""

import argparse
import time

from agents.bot import HeuristicBot
from agents.ppo_bot import PPOBot
from agents.random_bot import RandomBot
from quoridor.display import render
from quoridor.game import QuoridorState


def play_game(
    agent:    PPOBot,
    opponent,
    delay:    float = 0.3,
    quiet:    bool  = False,
) -> tuple[int, int]:
    """
    Play one complete game: PPO agent as P0, opponent as P1.

    Returns
    -------
    (winner, move_count) : tuple[int, int]
        winner is 0 (agent) or 1 (opponent).
        move_count is the total number of half-moves (plies) in the game.
    """
    game = QuoridorState()
    agent.reset()
    opponent.reset()
    move_count = 0

    while not game.done:
        if not quiet:
            # \033[2J clears the screen; \033[H moves cursor to top-left.
            print("\033[2J\033[H", end="")
            print(render(game))
            who = "Agent (P0)" if game.turn == 0 else "Opponent (P1)"
            print(f"\nPly {move_count}  |  {who} to move")
            time.sleep(delay)

        if game.turn == 0:
            action = agent.choose_action(game)
        else:
            action = opponent.choose_action(game)

        if action[0] == "move":
            game.move_to(action[1], action[2])
        else:
            game.place_fence(action[1], action[2], action[3])

        move_count += 1

    if not quiet:
        print("\033[2J\033[H", end="")
        print(render(game))
        winner_name = "Agent (P0)" if game.winner == 0 else "Opponent (P1)"
        print(f"\n{winner_name} wins in {move_count} plies!")
        time.sleep(delay * 3)

    return game.winner, move_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch a trained PPO agent play Quoridor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to .pt checkpoint (e.g. checkpoints/ppo_best.pt)",
    )
    parser.add_argument(
        "--model", default="bfs_resnet",
        choices=["baseline", "resnet", "bfs", "bfs_resnet"],
        help="Model architecture — must match the checkpoint",
    )
    parser.add_argument(
        "--opponent", default="heuristic",
        choices=["heuristic", "random"],
        help="Opponent the agent faces",
    )
    parser.add_argument(
        "--games", type=int, default=5,
        help="Number of games to play",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Seconds to pause between plies",
    )
    parser.add_argument(
        "--no-greedy", dest="greedy", action="store_false",
        help="Sample from the action distribution instead of taking argmax",
    )
    parser.set_defaults(greedy=True)
    args = parser.parse_args()

    agent = PPOBot(
        checkpoint_path=args.checkpoint,
        model_type=args.model,
        greedy=args.greedy,
    )

    if args.opponent == "heuristic":
        opponent = HeuristicBot()
    else:
        opponent = RandomBot()

    mode = "greedy" if args.greedy else "stochastic"
    print(f"Agent    : PPO ({args.model}, {mode}) ← {args.checkpoint}")
    print(f"Opponent : {args.opponent}")
    print(f"Games    : {args.games}  |  Delay: {args.delay}s")
    print()
    input("Press Enter to start...\n")

    wins       = 0
    total_plies = 0

    for i in range(args.games):
        winner, plies = play_game(agent, opponent, delay=args.delay)
        if winner == 0:
            wins += 1
        total_plies += plies

        result = "WIN" if winner == 0 else "loss"
        print(f"  Game {i + 1:2d}/{args.games}  [{result}]  {plies} plies  "
              f"— Agent wins so far: {wins}/{i + 1}")

        if i < args.games - 1:
            input("\nPress Enter for next game...\n")

    avg_plies = total_plies / args.games
    print(f"\n{'─' * 40}")
    print(f"Final  : {wins}/{args.games}  ({100 * wins / args.games:.0f}% win rate)")
    print(f"Avg    : {avg_plies:.1f} plies/game")


if __name__ == "__main__":
    main()
