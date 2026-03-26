"""
DQN training loop for the Quoridor agent.

Collects transitions via epsilon-greedy, trains with Double DQN + Huber loss,
evaluates win rate periodically. Supports curriculum training (RandomBot first,
then HeuristicBot with warm-started weights).

Usage:
    python scripts/train_dqn.py --opponent random --no-wandb
    python scripts/train_dqn.py --opponent heuristic --checkpoint checkpoints/best.pt
"""

import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from agents.dqn_model import DQNModel
from agents.replay_buffer import ReplayBuffer
from config import (
    BATCH_SIZE, EPSILON_DECAY_HEURISTIC, EPSILON_DECAY_RANDOM,
    EPSILON_END, EPSILON_START_HEURISTIC, EPSILON_START_RANDOM,
    EVAL_EPISODES, EVAL_FREQ, GAMMA, GRADIENT_CLIP_NORM,
    LEARNING_RATE, MAX_STEPS, NUM_ACTIONS, REPLAY_BUFFER_SIZE,
    SEED, TARGET_UPDATE_FREQ, WIN_RATE_TARGET,
)
from quoridor.env import QuoridorEnv


def epsilon_greedy(model, spatial, scalars, legal_mask, epsilon, device):
    """Pick action: random with prob epsilon, greedy otherwise."""
    if random.random() < epsilon:
        legal_indices = np.where(legal_mask)[0]
        return int(np.random.choice(legal_indices))

    model.eval()
    with torch.no_grad():
        spatial_t    = torch.tensor(spatial).unsqueeze(0).to(device)
        scalars_t    = torch.tensor(scalars).unsqueeze(0).to(device)
        legal_mask_t = torch.tensor(legal_mask).unsqueeze(0).to(device)
        q = model(spatial_t, scalars_t, legal_mask_t)
    return int(q.argmax(dim=1).item())


def compute_td_loss(batch, online_net, target_net, device):
    """Double DQN TD loss with Huber. Returns (loss, mean_q)."""
    spatial          = torch.tensor(batch["spatial"]).to(device)
    scalars          = torch.tensor(batch["scalars"]).to(device)
    actions          = torch.tensor(batch["actions"]).to(device)
    rewards          = torch.tensor(batch["rewards"]).to(device)
    next_spatial     = torch.tensor(batch["next_spatial"]).to(device)
    next_scalars     = torch.tensor(batch["next_scalars"]).to(device)
    dones            = torch.tensor(batch["dones"]).float().to(device)
    next_legal_mask  = torch.tensor(batch["next_legal_mask"]).to(device)

    B = spatial.shape[0]

    # Q(s, a) for the action we actually took
    all_true_mask = torch.ones(B, NUM_ACTIONS, dtype=torch.bool, device=device)
    q_all     = online_net(spatial, scalars, all_true_mask)
    q_current = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Double DQN target: online picks action, target evaluates it
    with torch.no_grad():
        q_next_online     = online_net(next_spatial, next_scalars, next_legal_mask)
        best_next_actions = q_next_online.argmax(dim=1)
        q_next_target     = target_net(next_spatial, next_scalars, next_legal_mask)
        q_next_max = q_next_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)
        target = rewards + GAMMA * q_next_max * (1.0 - dones)

    loss = F.huber_loss(q_current, target, delta=1.0)
    mean_q = q_current.detach().mean().item()
    return loss, mean_q


def evaluate(model, n_episodes, device, bot):
    """Run n_episodes with greedy policy. Returns (win_rate, mean_ep_length)."""
    eval_env = QuoridorEnv(bot=bot)
    model.eval()
    wins = 0
    total_length = 0

    with torch.no_grad():
        for _ in range(n_episodes):
            spatial, scalars = eval_env.reset()
            legal_mask = eval_env.get_legal_mask()
            done = False
            ep_length = 0

            while not done:
                spatial_t = torch.tensor(spatial).unsqueeze(0).to(device)
                scalars_t = torch.tensor(scalars).unsqueeze(0).to(device)
                mask_t    = torch.tensor(legal_mask).unsqueeze(0).to(device)
                q         = model(spatial_t, scalars_t, mask_t)
                action    = int(q.argmax(dim=1).item())

                spatial, scalars, reward, done, info = eval_env.step(action)
                legal_mask = info["legal_mask"]
                ep_length += 1

            if reward > 0:
                wins += 1
            total_length += ep_length

    model.train()
    return wins / n_episodes, total_length / n_episodes


def main(use_wandb, args):
    # device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # opponent setup
    if args.opponent == "random":
        from agents.random_bot import RandomBot
        bot = RandomBot()
        epsilon_start = EPSILON_START_RANDOM
        epsilon_decay = EPSILON_DECAY_RANDOM
    else:
        from agents.bot import HeuristicBot
        bot = HeuristicBot()
        epsilon_start = EPSILON_START_HEURISTIC
        epsilon_decay = EPSILON_DECAY_HEURISTIC
    print(f"Opponent: {args.opponent}")

    env        = QuoridorEnv(bot=bot)
    online_net = DQNModel().to(device)
    target_net = copy.deepcopy(online_net)
    target_net.eval()

    # optional warm-start from checkpoint
    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint, map_location=device)
        online_net.load_state_dict(state_dict)
        target_net.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {args.checkpoint}")

    optimizer = torch.optim.Adam(online_net.parameters(), lr=LEARNING_RATE)
    buffer    = ReplayBuffer()

    os.makedirs("checkpoints", exist_ok=True)

    # wandb
    if use_wandb:
        import wandb
        wandb.init(
            project="quoridor-dqn",
            config={
                "device": str(device), "opponent": args.opponent,
                "checkpoint": args.checkpoint, "lr": LEARNING_RATE,
                "gamma": GAMMA, "batch_size": BATCH_SIZE,
                "buffer_size": REPLAY_BUFFER_SIZE,
                "target_update_freq": TARGET_UPDATE_FREQ,
                "grad_clip": GRADIENT_CLIP_NORM,
                "epsilon_start": epsilon_start, "epsilon_end": EPSILON_END,
                "epsilon_decay": epsilon_decay, "max_steps": MAX_STEPS,
                "eval_freq": EVAL_FREQ, "eval_episodes": EVAL_EPISODES,
                "seed": SEED,
            },
        )

    epsilon       = epsilon_start
    total_steps   = 0
    best_win_rate = 0.0
    graduated     = False

    print(f"Starting training... buffer fills after {BATCH_SIZE} steps")

    while total_steps < MAX_STEPS and not graduated:
        spatial, scalars = env.reset()
        legal_mask = env.get_legal_mask()
        done = False

        while not done and total_steps < MAX_STEPS:
            action = epsilon_greedy(online_net, spatial, scalars, legal_mask, epsilon, device)

            next_spatial, next_scalars, reward, done, info = env.step(action)
            next_legal_mask = info["legal_mask"]

            buffer.push({
                "spatial": spatial, "scalars": scalars,
                "action": action, "reward": reward,
                "next_spatial": next_spatial, "next_scalars": next_scalars,
                "done": done, "next_legal_mask": next_legal_mask,
            })

            # train once buffer has enough data
            if len(buffer) >= BATCH_SIZE:
                online_net.train()
                batch = buffer.sample(BATCH_SIZE)
                loss, mean_q = compute_td_loss(batch, online_net, target_net, device)

                optimizer.zero_grad()
                loss.backward()
                grad_norm = clip_grad_norm_(online_net.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()

                if use_wandb:
                    import wandb
                    wandb.log({
                        "train/loss": loss.item(), "train/epsilon": epsilon,
                        "train/mean_q_value": mean_q, "train/grad_norm": grad_norm.item(),
                    }, step=total_steps)

            # hard target update
            if total_steps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(online_net.state_dict())
                target_net.eval()

            # eval
            if total_steps > 0 and total_steps % EVAL_FREQ == 0:
                win_rate, mean_length = evaluate(online_net, EVAL_EPISODES, device, bot)
                print(
                    f"Step {total_steps:>7} | "
                    f"win_rate={win_rate:.2%} | "
                    f"eps={epsilon:.3f} | "
                    f"mean_ep_len={mean_length:.1f}"
                )

                if use_wandb:
                    import wandb
                    wandb.log({
                        "eval/win_rate": win_rate,
                        "eval/mean_episode_length": mean_length,
                    }, step=total_steps)

                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    torch.save(online_net.state_dict(), "checkpoints/best.pt")
                    print(f"  New best ({win_rate:.2%}) - saved checkpoint")

                if win_rate >= WIN_RATE_TARGET:
                    print(f"\nGraduated at step {total_steps}! Win rate: {win_rate:.2%}")
                    graduated = True
                    break

            # advance
            spatial, scalars, legal_mask = next_spatial, next_scalars, next_legal_mask
            total_steps += 1
            epsilon = max(EPSILON_END, epsilon * epsilon_decay)

    final_path = f"checkpoints/final_{total_steps}.pt"
    torch.save(online_net.state_dict(), final_path)
    print(f"\nDone. Final checkpoint: {final_path}")
    print(f"Best win rate: {best_win_rate:.2%}")

    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent for Quoridor")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--opponent", choices=["random", "heuristic"], default="random",
                        help="Training opponent: random (phase 1) or heuristic (phase 2)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pt checkpoint for warm-start")
    args = parser.parse_args()
    main(use_wandb=not args.no_wandb, args=args)
