"""
Experiment runner for testing different model architectures and reward configs.

Trains a model for a fixed number of steps, then evaluates with detailed
behavioral analysis: win rate, game length, wall usage, distance progression,
and strategy classification.

Usage:
    python scripts/experiment.py --config baseline
    python scripts/experiment.py --config deeper
    python scripts/experiment.py --config dueling
    python scripts/experiment.py --config reward_v2
"""

import argparse
import copy
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.dqn_model import DQNModel
from agents.replay_buffer import ReplayBuffer
from agents.bot import HeuristicBot
from quoridor.env import QuoridorEnv
from quoridor.action_encoding import index_to_action
from config import (
    BOARD_SIZE, NUM_ACTIONS, NUM_CHANNELS, NUM_SCALARS,
    BATCH_SIZE, GAMMA, GRADIENT_CLIP_NORM,
    REPLAY_BUFFER_SIZE, SEED,
)


# ─── Alternative Model Architectures ───────────────────────────────────────

class DeeperDQN(nn.Module):
    """3 conv layers with residual connection + LeakyReLU. Deeper feature extraction."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )
        # Residual: conv2 output + conv3 output (both 64 channels)

        _conv_out = 64 * BOARD_SIZE * BOARD_SIZE  # 5184
        _fc_in = _conv_out + NUM_SCALARS  # 5186

        self.fc = nn.Sequential(
            nn.Linear(_fc_in, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, NUM_ACTIONS),
        )

    def forward(self, spatial, scalars, legal_mask):
        x = self.conv1(spatial)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        x = x2 + x3  # residual connection
        x = x.flatten(1)
        fused = torch.cat([x, scalars], dim=1)
        q = self.fc(fused)
        q = q.masked_fill(~legal_mask, -1e9)
        return q


class DuelingDQN(nn.Module):
    """Dueling architecture: separate value and advantage streams.

    The value stream estimates V(s) - how good is this state overall.
    The advantage stream estimates A(s,a) - how much better is each action
    compared to the average action in this state.

    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))

    This helps the network learn which states are valuable independent of
    the specific action taken, which is useful when many actions have similar
    values (e.g., most wall placements are equally unhelpful).
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
        )

        _conv_out = 64 * BOARD_SIZE * BOARD_SIZE + NUM_SCALARS  # 5186

        # Value stream: state -> scalar value
        self.value_stream = nn.Sequential(
            nn.Linear(_conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Advantage stream: state -> per-action advantage
        self.advantage_stream = nn.Sequential(
            nn.Linear(_conv_out, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_ACTIONS),
        )

    def forward(self, spatial, scalars, legal_mask):
        x = self.conv(spatial)
        fused = torch.cat([x, scalars], dim=1)

        value = self.value_stream(fused)            # (B, 1)
        advantage = self.advantage_stream(fused)     # (B, 137)

        # Q = V + (A - mean(A))
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        q = q.masked_fill(~legal_mask, -1e9)
        return q


# ─── Experiment Configurations ──────────────────────────────────────────────

CONFIGS = {
    "baseline": {
        "desc": "Original 2-conv DQN with current reward settings",
        "model_class": DQNModel,
        "lr": 5e-5,
        "target_update_freq": 200,
        "step_penalty": -0.01,
        "reward_shaping": True,
        "tempo_weight": 0.05,
        "path_width_weight": 0.02,
        "epsilon_start": 0.3,
        "epsilon_decay": 0.99998,
        "epsilon_end": 0.05,
        "max_steps": 100_000,
        "checkpoint": "checkpoints/best.pt",
    },
    "deeper": {
        "desc": "3-conv residual network with LeakyReLU, deeper FC layers",
        "model_class": DeeperDQN,
        "lr": 3e-5,
        "target_update_freq": 300,
        "step_penalty": -0.01,
        "reward_shaping": True,
        "tempo_weight": 0.05,
        "path_width_weight": 0.02,
        "epsilon_start": 0.5,
        "epsilon_decay": 0.99998,
        "epsilon_end": 0.05,
        "max_steps": 100_000,
        "checkpoint": None,  # train from scratch
    },
    "dueling": {
        "desc": "Dueling DQN, no shaping (pure win/loss + step penalty), from scratch",
        "model_class": DuelingDQN,
        "lr": 1e-4,
        "target_update_freq": 500,
        "step_penalty": -0.02,
        "reward_shaping": False,
        "tempo_weight": 0.0,
        "path_width_weight": 0.0,
        "epsilon_start": 0.9,
        "epsilon_decay": 0.99997,
        "epsilon_end": 0.05,
        "max_steps": 100_000,
        "checkpoint": None,
    },
    "distance": {
        "desc": "Direct distance rewards: +0.1 per step closer, penalty for walls that don't help",
        "model_class": DuelingDQN,
        "lr": 1e-4,
        "target_update_freq": 500,
        "step_penalty": -0.02,
        "reward_shaping": False,
        "tempo_weight": 0.0,
        "path_width_weight": 0.0,
        "distance_reward": True,
        "epsilon_start": 0.9,
        "epsilon_decay": 0.99997,
        "epsilon_end": 0.05,
        "max_steps": 100_000,
        "checkpoint": None,
    },
    "biased_explore": {
        "desc": "Dueling DQN + distance rewards + 70% move bias in exploration (fixes action space imbalance)",
        "model_class": DuelingDQN,
        "lr": 1e-4,
        "target_update_freq": 500,
        "step_penalty": -0.02,
        "reward_shaping": False,
        "tempo_weight": 0.0,
        "path_width_weight": 0.0,
        "distance_reward": True,
        "move_bias": 0.7,
        "epsilon_start": 0.9,
        "epsilon_decay": 0.99997,
        "epsilon_end": 0.05,
        "max_steps": 100_000,
        "checkpoint": None,
    },
    "reward_v2": {
        "desc": "Stronger tempo shaping + higher step penalty to force goal-seeking",
        "model_class": DQNModel,
        "lr": 5e-5,
        "target_update_freq": 200,
        "step_penalty": -0.02,
        "reward_shaping": True,
        "tempo_weight": 0.10,
        "path_width_weight": 0.03,
        "epsilon_start": 0.3,
        "epsilon_decay": 0.99998,
        "epsilon_end": 0.05,
        "max_steps": 100_000,
        "checkpoint": "checkpoints/best.pt",
    },
}


# ─── Training Logic ─────────────────────────────────────────────────────────

def epsilon_greedy(model, spatial, scalars, legal_mask, epsilon, device,
                    move_bias=0.0):
    """Epsilon-greedy with optional move bias during exploration.

    move_bias: probability of picking a move action (indices 0-7) during
    random exploration, if any are legal. 0.0 = uniform random (original).
    0.7 = 70% chance of picking a move when exploring.
    """
    if random.random() < epsilon:
        legal_indices = np.where(legal_mask)[0]
        if move_bias > 0:
            move_indices = legal_indices[legal_indices < 8]
            wall_indices = legal_indices[legal_indices >= 8]
            if len(move_indices) > 0 and random.random() < move_bias:
                return int(np.random.choice(move_indices))
            elif len(wall_indices) > 0:
                return int(np.random.choice(wall_indices))
        return int(np.random.choice(legal_indices))
    model.eval()
    with torch.no_grad():
        s = torch.tensor(spatial).unsqueeze(0).to(device)
        sc = torch.tensor(scalars).unsqueeze(0).to(device)
        m = torch.tensor(legal_mask).unsqueeze(0).to(device)
        q = model(s, sc, m)
    return int(q.argmax(dim=1).item())


def compute_td_loss(batch, online_net, target_net, device):
    spatial = torch.tensor(batch["spatial"]).to(device)
    scalars = torch.tensor(batch["scalars"]).to(device)
    actions = torch.tensor(batch["actions"]).to(device)
    rewards = torch.tensor(batch["rewards"]).to(device)
    next_spatial = torch.tensor(batch["next_spatial"]).to(device)
    next_scalars = torch.tensor(batch["next_scalars"]).to(device)
    dones = torch.tensor(batch["dones"]).float().to(device)
    next_legal_mask = torch.tensor(batch["next_legal_mask"]).to(device)

    B = spatial.shape[0]
    all_true = torch.ones(B, NUM_ACTIONS, dtype=torch.bool, device=device)
    q_all = online_net(spatial, scalars, all_true)
    q_current = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        q_next_online = online_net(next_spatial, next_scalars, next_legal_mask)
        best_next = q_next_online.argmax(dim=1)
        q_next_target = target_net(next_spatial, next_scalars, next_legal_mask)
        q_next_max = q_next_target.gather(1, best_next.unsqueeze(1)).squeeze(1)
        target = rewards + GAMMA * q_next_max * (1.0 - dones)

    loss = F.huber_loss(q_current, target, delta=1.0)
    mean_q = q_current.detach().mean().item()
    return loss, mean_q


# ─── Behavioral Analysis ────────────────────────────────────────────────────

def analyze_behavior(model, device, n_episodes=50):
    """Play n_episodes with greedy policy and collect detailed statistics.

    This is how we figure out what the bot learned:
    - We track every action it takes (move vs wall placement)
    - We track its distance to goal over time to see if it's making progress
    - We classify each game's strategy based on action ratios
    - We look at wall usage patterns to see if it's using walls strategically
    """
    bot = HeuristicBot()
    env = QuoridorEnv(bot=bot, reward_shaping=False)
    model.eval()

    stats = {
        "wins": 0, "losses": 0,
        "game_lengths": [],
        "move_counts": [],       # pawn moves per game
        "wall_counts": [],       # wall placements per game
        "walls_left_at_end": [], # walls remaining when game ends
        "distance_at_start": [],
        "distance_at_end": [],
        "strategy_labels": [],   # classified strategy per game
        "win_distances": [],     # final distance when winning
        "loss_distances": [],    # final distance when losing
        "action_type_sequence": [],  # for one sample game
    }

    with torch.no_grad():
        for ep in range(n_episodes):
            spatial, scalars = env.reset()
            legal_mask = env.get_legal_mask()
            done = False

            moves, walls = 0, 0
            start_dist = env.state.shortest_path(0)
            action_seq = []

            while not done:
                s = torch.tensor(spatial).unsqueeze(0).to(device)
                sc = torch.tensor(scalars).unsqueeze(0).to(device)
                m = torch.tensor(legal_mask).unsqueeze(0).to(device)
                q = model(s, sc, m)
                action = int(q.argmax(dim=1).item())

                # classify action
                decoded = index_to_action(action)
                if decoded[0] == "move":
                    moves += 1
                    action_seq.append("M")
                else:
                    walls += 1
                    action_seq.append("W")

                spatial, scalars, reward, done, info = env.step(action)
                legal_mask = info["legal_mask"]

            end_dist = env.state.shortest_path(0)
            game_len = moves + walls

            stats["game_lengths"].append(game_len)
            stats["move_counts"].append(moves)
            stats["wall_counts"].append(walls)
            stats["walls_left_at_end"].append(int(env.state.walls_left[0]))
            stats["distance_at_start"].append(start_dist)
            stats["distance_at_end"].append(end_dist)

            if reward > 0:
                stats["wins"] += 1
                stats["win_distances"].append(end_dist)
            else:
                stats["losses"] += 1
                stats["loss_distances"].append(end_dist)

            # classify strategy
            total = moves + walls
            wall_ratio = walls / total if total > 0 else 0
            if wall_ratio > 0.5:
                strategy = "wall-heavy"
            elif wall_ratio > 0.25:
                strategy = "balanced"
            elif wall_ratio > 0.05:
                strategy = "move-focused"
            else:
                strategy = "pure-runner"
            stats["strategy_labels"].append(strategy)

            if ep == 0:
                stats["action_type_sequence"] = action_seq

    return stats


def print_analysis(stats, config_name, config_desc):
    """Print a human-readable analysis of the model's behavior."""
    n = stats["wins"] + stats["losses"]
    win_rate = stats["wins"] / n if n > 0 else 0

    print("\n" + "=" * 70)
    print(f"  EXPERIMENT: {config_name}")
    print(f"  {config_desc}")
    print("=" * 70)

    print(f"\n  Win Rate: {win_rate:.1%} ({stats['wins']}/{n})")
    print(f"  Avg Game Length: {np.mean(stats['game_lengths']):.1f} steps")
    print(f"  Avg Moves/Game: {np.mean(stats['move_counts']):.1f}")
    print(f"  Avg Walls/Game: {np.mean(stats['wall_counts']):.1f}")
    print(f"  Avg Walls Left at End: {np.mean(stats['walls_left_at_end']):.1f}/10")

    print(f"\n  Distance to Goal:")
    print(f"    At start: {np.mean(stats['distance_at_start']):.1f}")
    print(f"    At end:   {np.mean(stats['distance_at_end']):.1f}")
    if stats["win_distances"]:
        print(f"    (Wins end at dist 0 by definition)")
    if stats["loss_distances"]:
        print(f"    When losing, avg dist from goal: {np.mean(stats['loss_distances']):.1f}")

    # Strategy distribution
    from collections import Counter
    strat_counts = Counter(stats["strategy_labels"])
    print(f"\n  Strategy Distribution:")
    for strat, count in strat_counts.most_common():
        print(f"    {strat}: {count}/{n} ({count/n:.0%})")

    # Sample game action sequence
    if stats["action_type_sequence"]:
        seq = "".join(stats["action_type_sequence"][:60])
        if len(stats["action_type_sequence"]) > 60:
            seq += "..."
        print(f"\n  Sample Game Actions (M=move, W=wall):")
        print(f"    {seq}")

    print()


# ─── Custom Env with configurable rewards ───────────────────────────────────

class ConfigurableEnv:
    """QuoridorEnv wrapper with experiment-specific reward parameters."""

    def __init__(self, bot, step_penalty, reward_shaping, tempo_weight, path_width_weight,
                 distance_reward=False):
        self.state = __import__("quoridor.game", fromlist=["QuoridorState"]).QuoridorState()
        self.bot = bot
        self.step_penalty = step_penalty
        self.use_shaping = reward_shaping
        self.tempo_weight = tempo_weight
        self.path_width_weight = path_width_weight
        self.distance_reward = distance_reward
        self._prev_my_dist = None
        self._prev_opp_dist = None

    def reset(self):
        self.state.reset()
        self.bot.reset()
        self._prev_my_dist = self.state.shortest_path(0)
        self._prev_opp_dist = self.state.shortest_path(1)
        return self.state.get_observation()

    def step(self, action):
        from quoridor.action_encoding import index_to_action
        _TERMINAL_MASK = np.zeros(NUM_ACTIONS, dtype=bool)

        legal = self.state.get_legal_mask()
        if not legal[action]:
            raise ValueError(f"Action {action} is illegal")

        if self.use_shaping:
            phi_s = self._potential()

        # agent moves
        won = self._apply(action)
        if won:
            spatial, scalars = self.state.get_observation()
            return spatial, scalars, +1.0, True, {"legal_mask": _TERMINAL_MASK.copy()}

        # bot responds
        bot_action = self.bot.choose_action(self.state)
        self._dispatch_bot(bot_action)

        if self.state.done:
            spatial, scalars = self.state.get_observation()
            return spatial, scalars, -1.0, True, {"legal_mask": _TERMINAL_MASK.copy()}

        reward = self.step_penalty

        if self.use_shaping:
            phi_s_prime = self._potential()
            reward += GAMMA * phi_s_prime - phi_s

        # Distance-based reward: directly reward getting closer to goal
        # and penalize opponent getting closer to their goal
        if self.distance_reward:
            my_dist = self.state.shortest_path(0)
            opp_dist = self.state.shortest_path(1)

            # +0.1 for each step closer we got, -0.1 for each step further
            my_progress = (self._prev_my_dist - my_dist) * 0.1
            # +0.05 for slowing opponent, -0.05 for opponent advancing
            opp_progress = (opp_dist - self._prev_opp_dist) * 0.05

            reward += my_progress + opp_progress
            self._prev_my_dist = my_dist
            self._prev_opp_dist = opp_dist

        spatial, scalars = self.state.get_observation()
        return spatial, scalars, reward, False, {"legal_mask": self.state.get_legal_mask()}

    def get_legal_mask(self):
        return self.state.get_legal_mask()

    def _potential(self):
        my_dist, my_reach = self.state.shortest_path_and_reach(0)
        opp_dist, _ = self.state.shortest_path_and_reach(1)
        if my_dist == float("inf"):
            tempo = -20.0
        elif opp_dist == float("inf"):
            tempo = 20.0
        else:
            tempo = float(opp_dist - my_dist)
        return (self.tempo_weight * tempo
                + self.path_width_weight * (my_reach / BOARD_SIZE))

    def _apply(self, idx):
        from quoridor.action_encoding import index_to_action
        action = index_to_action(idx)
        if action[0] == "move":
            dr, dc = action[1], action[2]
            cur_r = int(self.state.pos[self.state.turn, 0])
            cur_c = int(self.state.pos[self.state.turn, 1])
            for dest_r, dest_c in self.state.get_valid_moves():
                if (np.sign(dest_r - cur_r) == np.sign(dr)
                        and np.sign(dest_c - cur_c) == np.sign(dc)):
                    return self.state.move_to(dest_r, dest_c)
            raise ValueError(f"No valid destination for ({dr}, {dc})")
        else:
            self.state.place_fence(action[1], action[2], action[3])
            return False

    def _dispatch_bot(self, bot_action):
        if bot_action[0] == "move":
            _, r, c = bot_action
            self.state.move_to(r, c)
        else:
            _, r, c, o = bot_action
            self.state.place_fence(r, c, o)


# ─── Main Training + Eval Loop ──────────────────────────────────────────────

def run_experiment(config_name, eval_only=False):
    cfg = CONFIGS[config_name]
    print(f"\n{'='*70}")
    print(f"  Config: {config_name} — {cfg['desc']}")
    print(f"{'='*70}")

    # device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Device: {device}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # model
    model_class = cfg["model_class"]
    online_net = model_class().to(device)
    param_count = sum(p.numel() for p in online_net.parameters())
    print(f"  Model: {model_class.__name__} ({param_count:,} params)")

    # load checkpoint if specified
    if cfg["checkpoint"] and os.path.exists(cfg["checkpoint"]):
        try:
            state_dict = torch.load(cfg["checkpoint"], map_location=device)
            online_net.load_state_dict(state_dict)
            print(f"  Loaded checkpoint: {cfg['checkpoint']}")
        except RuntimeError as e:
            print(f"  Checkpoint incompatible (different architecture), training from scratch")
            print(f"    ({e})")

    target_net = copy.deepcopy(online_net)
    target_net.eval()

    if eval_only:
        print("\n  [EVAL ONLY MODE]")
        stats = analyze_behavior(online_net, device, n_episodes=50)
        print_analysis(stats, config_name, cfg["desc"])
        return stats

    # training
    bot = HeuristicBot()
    env = ConfigurableEnv(
        bot=bot,
        step_penalty=cfg["step_penalty"],
        reward_shaping=cfg["reward_shaping"],
        tempo_weight=cfg["tempo_weight"],
        path_width_weight=cfg["path_width_weight"],
        distance_reward=cfg.get("distance_reward", False),
    )

    optimizer = torch.optim.Adam(online_net.parameters(), lr=cfg["lr"])
    buffer = ReplayBuffer()

    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/exp_{config_name}.pt"

    epsilon = cfg["epsilon_start"]
    total_steps = 0
    best_win_rate = 0.0
    eval_freq = 10_000
    eval_episodes = 30
    start_time = time.time()

    print(f"  Training for {cfg['max_steps']:,} steps...")
    print(f"  Eval every {eval_freq:,} steps ({eval_episodes} episodes)")
    print()

    while total_steps < cfg["max_steps"]:
        spatial, scalars = env.reset()
        legal_mask = env.get_legal_mask()
        done = False

        while not done and total_steps < cfg["max_steps"]:
            action = epsilon_greedy(online_net, spatial, scalars, legal_mask, epsilon, device,
                                    move_bias=cfg.get("move_bias", 0.0))
            next_spatial, next_scalars, reward, done, info = env.step(action)
            next_legal_mask = info["legal_mask"]

            buffer.push({
                "spatial": spatial, "scalars": scalars,
                "action": action, "reward": reward,
                "next_spatial": next_spatial, "next_scalars": next_scalars,
                "done": done, "next_legal_mask": next_legal_mask,
            })

            if len(buffer) >= BATCH_SIZE:
                online_net.train()
                batch = buffer.sample(BATCH_SIZE)
                loss, mean_q = compute_td_loss(batch, online_net, target_net, device)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(online_net.parameters(), GRADIENT_CLIP_NORM)
                optimizer.step()

            if total_steps % cfg["target_update_freq"] == 0:
                target_net.load_state_dict(online_net.state_dict())
                target_net.eval()

            if total_steps > 0 and total_steps % eval_freq == 0:
                eval_bot = HeuristicBot()
                eval_env = QuoridorEnv(bot=eval_bot, reward_shaping=False)
                online_net.eval()
                wins = 0
                total_len = 0

                with torch.no_grad():
                    for _ in range(eval_episodes):
                        sp, sc = eval_env.reset()
                        lm = eval_env.get_legal_mask()
                        d = False
                        ep_len = 0
                        while not d:
                            st = torch.tensor(sp).unsqueeze(0).to(device)
                            sct = torch.tensor(sc).unsqueeze(0).to(device)
                            mt = torch.tensor(lm).unsqueeze(0).to(device)
                            q = online_net(st, sct, mt)
                            a = int(q.argmax(dim=1).item())
                            sp, sc, r, d, inf = eval_env.step(a)
                            lm = inf["legal_mask"]
                            ep_len += 1
                        if r > 0:
                            wins += 1
                        total_len += ep_len

                win_rate = wins / eval_episodes
                mean_len = total_len / eval_episodes
                elapsed = time.time() - start_time
                print(
                    f"  Step {total_steps:>7,} | "
                    f"win={win_rate:.0%} | "
                    f"eps={epsilon:.3f} | "
                    f"len={mean_len:.0f} | "
                    f"time={elapsed:.0f}s"
                )

                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    torch.save(online_net.state_dict(), save_path)

                online_net.train()

            spatial, scalars, legal_mask = next_spatial, next_scalars, next_legal_mask
            total_steps += 1
            epsilon = max(cfg["epsilon_end"], epsilon * cfg["epsilon_decay"])

    # save final
    torch.save(online_net.state_dict(), save_path)
    print(f"\n  Training complete. Best win rate: {best_win_rate:.0%}")
    print(f"  Saved to: {save_path}")

    # detailed behavioral analysis
    print("\n  Running behavioral analysis...")
    stats = analyze_behavior(online_net, device, n_episodes=50)
    print_analysis(stats, config_name, cfg["desc"])
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, choices=list(CONFIGS.keys()))
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training, just evaluate existing checkpoint")
    args = parser.parse_args()
    run_experiment(args.config, eval_only=args.eval_only)
