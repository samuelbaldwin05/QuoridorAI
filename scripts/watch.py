"""
watch.py — Watch a bot-vs-bot game with a configurable delay between moves.

Usage:
    python -m scripts.watch                           # HeuristicBot vs HeuristicBot
    python -m scripts.watch --delay 1.0              # slower (1 second per move)
    python -m scripts.watch --delay 0                # instant
    python -m scripts.watch --p1 random              # Heuristic vs Random
    python -m scripts.watch --p0 ppo --checkpoint checkpoints/best_ppo.pt --model bfs_resnet
"""

import argparse
import os
import time

from quoridor.game import QuoridorState
from quoridor.display import render
from agents.bot import HeuristicBot
from agents.random_bot import RandomBot


def _clear() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def _build_bot(name: str, args, player_idx: int):
    """Instantiate a bot by name."""
    if name == "heuristic":
        return HeuristicBot()
    if name == "random":
        return RandomBot()
    if name == "ppo":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when using --p0/--p1 ppo")
        if not args.model:
            raise ValueError("--model is required when using --p0/--p1 ppo")
        import torch
        from scripts.train_ppo import load_checkpoint, _MODEL_REGISTRY, _model_forward, _flip_legal_mask
        from quoridor.action_encoding import index_to_action, FENCE_GRID
        import numpy as np

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        ModelClass, use_bfs = _MODEL_REGISTRY[args.model]
        model = ModelClass().to(device)
        ckpt  = load_checkpoint(args.checkpoint, device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        class _PPOWatcher:
            """Minimal PPO bot for watch.py — greedy (argmax) policy."""
            def reset(self): pass
            def choose_action(self, game):
                flip = (game.turn == 1)
                spatial, scalars = game.get_observation(use_bfs=use_bfs)
                actual_mask = game.get_legal_mask()
                legal_mask  = _flip_legal_mask(actual_mask) if flip else actual_mask

                s  = torch.tensor(spatial,    dtype=torch.float32).unsqueeze(0).to(device)
                sc = torch.tensor(scalars,    dtype=torch.float32).unsqueeze(0).to(device)
                m  = torch.tensor(legal_mask, dtype=torch.bool   ).unsqueeze(0).to(device)

                with torch.no_grad():
                    dist, _, _ = _model_forward(model, s, sc, m)
                    action_idx = int(dist.probs.argmax(dim=1).item())

                action = index_to_action(action_idx)

                if action[0] == "move":
                    _, dr, dc = action
                    if flip:
                        dr = -dr
                    cur_r = int(game.pos[game.turn, 0])
                    cur_c = int(game.pos[game.turn, 1])
                    for dest_r, dest_c in game.get_valid_moves():
                        if (np.sign(dest_r - cur_r) == np.sign(dr)
                                and np.sign(dest_c - cur_c) == np.sign(dc)):
                            return ("move", dest_r, dest_c)
                    raise ValueError(f"PPO watcher: no valid destination for action {action_idx}")

                if action[0] == "fence":
                    _, r, c, ori = action
                    if flip:
                        r = FENCE_GRID - 1 - r
                    return ("fence", r, c, ori)

                return action

        return _PPOWatcher()

    raise ValueError(f"Unknown bot '{name}'. Choose: heuristic, random, ppo")


def run(args) -> None:
    import numpy as np
    game = QuoridorState()
    bots = [
        _build_bot(args.p0, args, player_idx=0),
        _build_bot(args.p1, args, player_idx=1),
    ]
    bot_names = [args.p0.capitalize(), args.p1.capitalize()]
    move_log: list[str] = []
    move_num = 0

    while not game.done:
        _clear()
        print(f"=== {bot_names[0]} (P0/bottom) vs {bot_names[1]} (P1/top) ===")
        print(f"Move {move_num}")
        print(render(game))

        # Last few moves
        if move_log:
            print("\nRecent moves:")
            for entry in move_log[-6:]:
                print(f"  {entry}")

        print(f"\n[delay={args.delay}s — Ctrl+C to quit]")
        time.sleep(args.delay)

        bot = bots[game.turn]
        action = bot.choose_action(game)

        if action[0] == "move":
            col_label = chr(ord("a") + int(action[2]))
            row_label  = str(int(action[1]) + 1)
            desc = f"P{game.turn} ({bot_names[game.turn]}) moves → {col_label}{row_label}"
            game.move_to(action[1], action[2])
        elif action[0] == "fence":
            col_label = chr(ord("a") + int(action[2]))
            row_label  = str(int(action[1]) + 1)
            desc = f"P{game.turn} ({bot_names[game.turn]}) places {action[3]} fence at {col_label}{row_label}"
            game.place_fence(action[1], action[2], action[3])
        else:
            desc = f"P{game.turn} passes"

        move_log.append(f"#{move_num:>3}  {desc}")
        move_num += 1

    # Final board
    _clear()
    print(f"=== {bot_names[0]} (P0) vs {bot_names[1]} (P1) ===")
    print(f"Game over after {move_num} moves")
    print(render(game))
    print(f"\n{'='*40}")
    print(f"Winner: P{game.winner} ({bot_names[game.winner]})")
    print(f"\nFull move log:")
    for entry in move_log:
        print(f"  {entry}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch a bot-vs-bot Quoridor game.")
    parser.add_argument("--p0", type=str, default="heuristic",
                        choices=["heuristic", "random", "ppo"],
                        help="Bot for P0/bottom (default: heuristic).")
    parser.add_argument("--p1", type=str, default="heuristic",
                        choices=["heuristic", "random", "ppo"],
                        help="Bot for P1/top (default: heuristic).")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Seconds to pause between moves (default: 0.5).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to PPO checkpoint (required when --p0/--p1 ppo).")
    parser.add_argument("--model", type=str, default=None,
                        choices=["baseline", "resnet", "bfs", "bfs_resnet"],
                        help="PPO model architecture (required when --p0/--p1 ppo).")
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nStopped.")
