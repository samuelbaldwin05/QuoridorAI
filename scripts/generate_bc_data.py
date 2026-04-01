"""
generate_bc_data.py — Generate behavioral cloning training data from HeuristicBot games.

Runs BC_GAMES games of HeuristicBot vs HeuristicBot and records every
(state, action) pair from both players' perspectives. The resulting dataset
is saved to data/bc_data.npz (or data/bc_data_bfs.npz with --bfs) for use
by pretrain_bc.py.

Why use HeuristicBot self-play as expert data?
    The DQN agent can't learn to beat HeuristicBot from RL alone because
    sparse rewards provide no gradient signal when the agent never wins.
    Supervised pretraining on HeuristicBot actions bootstraps the policy to
    "what a competent Quoridor player does", giving the RL fine-tuning phase
    a starting point from which it can occasionally win and generate real signal.

Both P0 and P1 turns are recorded using current-player-normalized observations
(get_observation() always shows the board from the active player's POV). The
network never needs to know which player is acting — both produce identical-
format observations with "my pawn at bottom, goal at row 0".

Output: data/bc_data.npz (default) or data/bc_data_bfs.npz (--bfs) with keys:
    spatial     — (N, 4, 9, 9)  float32   board channels (4-ch default)
                  (N, 6, 9, 9)  float32   with --bfs (adds BFS distance maps)
    scalars     — (N, 2)        float32   wall counts (normalized)
    legal_masks — (N, 137)      bool      legal action mask
    actions     — (N,)          int64     HeuristicBot's chosen action index

Usage:
    python scripts/generate_bc_data.py               # 4-channel dataset
    python scripts/generate_bc_data.py --bfs          # 6-channel dataset for bfs/bfs_resnet models
    python scripts/generate_bc_data.py --games 500   # fewer games, faster
"""

import argparse
import os

import numpy as np

from agents.bot import HeuristicBot
from config import BC_GAMES, SEED
from quoridor.action_encoding import action_to_index
from quoridor.game import QuoridorState


def bot_action_to_index(game: QuoridorState, bot_action: tuple) -> int:
    """
    Convert a HeuristicBot action tuple to an action index.

    HeuristicBot.choose_action() returns absolute board positions for move
    actions: ("move", dest_r, dest_c). The action encoding uses direction
    deltas, not positions. We compute the direction sign to recover the delta.

    For fence actions ("fence", row, col, orientation) the encoding is direct.
    """
    if bot_action[0] == "move":
        _, dest_r, dest_c = bot_action
        cur_r = int(game.pos[game.turn, 0])
        cur_c = int(game.pos[game.turn, 1])
        dr = int(np.sign(dest_r - cur_r))
        dc = int(np.sign(dest_c - cur_c))
        return action_to_index(("move", dr, dc))
    else:  # "fence"
        _, row, col, orientation = bot_action
        return action_to_index(("fence", row, col, orientation))


def generate(n_games: int, use_bfs: bool = False) -> dict:
    """
    Run n_games of HeuristicBot vs HeuristicBot and collect all transitions.

    Args:
        n_games:  Number of complete games to simulate.
        use_bfs:  If True, observations include BFS distance map channels
                  (ch4=my dist, ch5=opp dist), giving spatial shape (N, 6, 9, 9).
                  If False, spatial shape is (N, 4, 9, 9).

    Returns a dict of numpy arrays ready to be saved with np.savez.
    """
    np.random.seed(SEED)

    all_spatial:      list[np.ndarray] = []
    all_scalars:      list[np.ndarray] = []
    all_legal_masks:  list[np.ndarray] = []
    all_actions:      list[int]        = []

    # Two bots — one per player — so each has its own opening-book state.
    bots = [HeuristicBot(), HeuristicBot()]

    for game_idx in range(n_games):
        game = QuoridorState()
        bots[0].reset()
        bots[1].reset()

        while not game.done:
            player = game.turn  # 0 or 1

            # Observe from the current player's perspective (board flipped if P1).
            # use_bfs=True adds ch4 (my BFS distance map) and ch5 (opp BFS distance map).
            spatial, scalars = game.get_observation(use_bfs=use_bfs)
            legal_mask = game.get_legal_mask()

            # Ask the appropriate bot for its action
            bot_action = bots[player].choose_action(game)
            action_idx = bot_action_to_index(game, bot_action)

            # Record this (state, action) pair
            all_spatial.append(spatial)
            all_scalars.append(scalars)
            all_legal_masks.append(legal_mask)
            all_actions.append(action_idx)

            # Apply the action to advance the game
            if bot_action[0] == "move":
                _, r, c = bot_action
                game.move_to(r, c)
            else:
                _, r, c, orientation = bot_action
                game.place_fence(r, c, orientation)

        if (game_idx + 1) % 100 == 0:
            n_samples = len(all_actions)
            print(f"  Game {game_idx + 1:>5}/{n_games}  |  samples so far: {n_samples:>7,}")

    return {
        "spatial":     np.array(all_spatial,     dtype=np.float32),
        "scalars":     np.array(all_scalars,     dtype=np.float32),
        "legal_masks": np.array(all_legal_masks, dtype=bool),
        "actions":     np.array(all_actions,     dtype=np.int64),
    }


def main(n_games: int, use_bfs: bool = False) -> None:
    os.makedirs("data", exist_ok=True)
    out_path = "data/bc_data_bfs.npz" if use_bfs else "data/bc_data.npz"

    print(f"Generating {n_games} HeuristicBot vs HeuristicBot games (use_bfs={use_bfs})...")
    data = generate(n_games, use_bfs=use_bfs)

    n_samples = data["actions"].shape[0]
    print(f"\nTotal samples collected: {n_samples:,}")
    print(f"  spatial shape:     {data['spatial'].shape}")
    print(f"  scalars shape:     {data['scalars'].shape}")
    print(f"  legal_masks shape: {data['legal_masks'].shape}")
    print(f"  actions shape:     {data['actions'].shape}")

    np.savez_compressed(out_path, **data)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BC training data.")
    parser.add_argument(
        "--games",
        type=int,
        default=BC_GAMES,
        help=f"Number of games to generate (default: {BC_GAMES}).",
    )
    parser.add_argument(
        "--bfs",
        action="store_true",
        default=False,
        help=(
            "Include BFS distance map channels (ch4/ch5) in observations. "
            "Required for bfs and bfs_resnet PPO model variants. "
            "Output is saved to data/bc_data_bfs.npz instead of data/bc_data.npz."
        ),
    )
    args = parser.parse_args()
    main(args.games, use_bfs=args.bfs)
