"""
eval_epsilon_sweep.py — Win rate vs opponent epsilon sweep.

Evaluates a PPO checkpoint against EpsilonHeuristicBot at a range of epsilon
values. Prints a table of win rates so you can see where the agent's capability
breaks down as the opponent approaches full strength (epsilon=0).

Usage:
    python -m scripts.eval_epsilon_sweep --checkpoint checkpoints/ppo_final_500224.pt
    python -m scripts.eval_epsilon_sweep --checkpoint checkpoints/ppo_final_500224.pt --games 100
    python -m scripts.eval_epsilon_sweep --checkpoint checkpoints/ppo_final_500224.pt \
        --epsilons 0.0 0.05 0.1 0.2 0.3
"""

import argparse

from agents.bot import EpsilonHeuristicBot
from agents.ppo_bot import PPOBot
from quoridor.game import QuoridorState

DEFAULT_EPSILONS = [0.3, 0.2, 0.15, 0.1, 0.09, 0.05, 0.03, 0.0]


def play_games(agent: PPOBot, epsilon: float, n_games: int) -> tuple[int, float]:
    """Play n_games and return (wins, avg_plies)."""
    opponent = EpsilonHeuristicBot(epsilon=epsilon)
    wins = 0
    total_plies = 0

    for _ in range(n_games):
        game = QuoridorState()
        agent.reset()
        opponent.reset()
        plies = 0

        while not game.done:
            if game.turn == 0:
                action = agent.choose_action(game)
            else:
                action = opponent.choose_action(game)

            if action[0] == "move":
                game.move_to(action[1], action[2])
            else:
                game.place_fence(action[1], action[2], action[3])
            plies += 1

        if game.winner == 0:
            wins += 1
        total_plies += plies

    return wins, total_plies / n_games


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Win rate vs opponent epsilon sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--model", default="bfs_resnet",
        choices=["baseline", "resnet", "bfs", "bfs_resnet"],
    )
    parser.add_argument("--games", type=int, default=50,
                        help="Games per epsilon value")
    parser.add_argument("--epsilons", type=float, nargs="+",
                        default=DEFAULT_EPSILONS,
                        help="Opponent epsilon values to test")
    parser.add_argument("--no-greedy", dest="greedy", action="store_false")
    parser.set_defaults(greedy=True)
    args = parser.parse_args()

    agent = PPOBot(args.checkpoint, model_type=args.model, greedy=args.greedy)
    mode = "greedy" if args.greedy else "stochastic"

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Model      : {args.model} ({mode})")
    print(f"Games/eps  : {args.games}")
    print()
    print(f"{'Epsilon':>8}  {'Win Rate':>9}  {'Wins':>6}  {'Avg Plies':>10}")
    print("─" * 42)

    for eps in sorted(args.epsilons, reverse=True):
        wins, avg_plies = play_games(agent, eps, args.games)
        win_pct = 100 * wins / args.games
        bar = "█" * int(win_pct / 5)  # 1 block per 5%
        label = "← training" if abs(eps - 0.09) < 0.001 else ("← full strength" if eps == 0.0 else "")
        print(f"  {eps:>6.2f}  {win_pct:>8.1f}%  {wins:>4}/{args.games}  "
              f"{avg_plies:>9.1f}  {bar} {label}")

    print()


if __name__ == "__main__":
    main()
