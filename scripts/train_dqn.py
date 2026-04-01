"""
train_dqn.py — DQN training loop for the Quoridor agent.

This script ties together all Stage 2 components (DQNModel, ReplayBuffer,
config) with the Stage 1 environment (QuoridorEnv) to train a vanilla DQN
agent against the HeuristicBot.

Algorithm summary:
  1. Collect transitions via epsilon-greedy exploration
  2. Store them in a replay buffer
  3. Sample random mini-batches to compute TD loss
  4. Periodically hard-copy the online network to a frozen target network
  5. Evaluate win rate vs HeuristicBot every EVAL_FREQ steps
  6. Save the best checkpoint; stop when win rate >= WIN_RATE_TARGET

Usage:
    python scripts/train_dqn.py              # requires wandb login
    python scripts/train_dqn.py --no-wandb   # dry run, no W&B logging

Device selection (automatic):
    MPS  — Apple Silicon GPU (fastest on Mac, used if available)
    CPU  — fallback if MPS is not available
"""

import argparse
import copy
import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from agents.dqn_model import DQNModel
from agents.replay_buffer import PrioritizedReplayBuffer
from config import (
    BATCH_SIZE,
    EPSILON_DECAY_HEURISTIC,
    EPSILON_DECAY_RANDOM,
    EPSILON_DECAY_SELF_PLAY,
    EPSILON_END,
    EPSILON_START_HEURISTIC,
    EPSILON_START_RANDOM,
    EPSILON_START_SELF_PLAY,
    EVAL_EPISODES,
    EVAL_FREQ,
    GAMMA,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    LEARNING_RATE_HEURISTIC,
    MAX_STEPS,
    N_STEP_RETURNS,
    NUM_ACTIONS,
    OPPONENT_UPDATE_FREQ,
    PER_BETA_END,
    PER_BETA_START,
    SEED,
    TARGET_UPDATE_FREQ,
    WIN_RATE_TARGET,
)
from quoridor.action_encoding import H_WALL_OFFSET, PASS_ACTION
from quoridor.env import QuoridorEnv


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def epsilon_greedy(
    model: DQNModel,
    spatial: np.ndarray,
    scalars: np.ndarray,
    legal_mask: np.ndarray,
    epsilon: float,
    device: torch.device,
) -> int:
    """
    Select an action using an epsilon-greedy policy.

    With probability epsilon, choose uniformly at random from legal actions
    (exploration). Otherwise, choose the action with the highest Q-value
    (exploitation).

    Args:
        model:      Online DQN network.
        spatial:    Board observation, shape (4, 9, 9).
        scalars:    Wall-count features, shape (2,).
        legal_mask: Boolean array of legal actions, shape (137,).
        epsilon:    Current exploration probability.
        device:     Torch device (mps or cpu) to run inference on.

    Returns:
        Chosen action index in [0, NUM_ACTIONS).
    """
    if random.random() < epsilon:
        # Explore: sample uniformly from all legal actions
        legal_indices = np.where(legal_mask)[0]
        return int(np.random.choice(legal_indices))

    # Exploit: greedy argmax over Q-values
    model.eval()
    with torch.no_grad():
        # Add batch dimension and move to device: (4,9,9) → (1,4,9,9)
        spatial_t    = torch.tensor(spatial).unsqueeze(0).to(device)     # (1, 4, 9, 9)
        scalars_t    = torch.tensor(scalars).unsqueeze(0).to(device)     # (1, 2)
        legal_mask_t = torch.tensor(legal_mask).unsqueeze(0).to(device)  # (1, 137)
        q = model(spatial_t, scalars_t, legal_mask_t)                    # (1, 137)
    return int(q.argmax(dim=1).item())


# ---------------------------------------------------------------------------
# TD loss
# ---------------------------------------------------------------------------

def compute_td_loss(
    batch: dict,
    online_net: DQNModel,
    target_net: DQNModel,
    device: torch.device,
) -> tuple[torch.Tensor, float, np.ndarray, list, float, float]:
    """
    Compute the Double DQN TD loss for a PER mini-batch.

    TD target: r + gamma * Q_target(s', argmax_a' Q_online(s', a'))
    Loss:      IS-weighted Huber(Q_online(s, a), target)

    IS weights (batch["weights"]) correct the bias introduced by non-uniform
    PER sampling: high-priority transitions are down-weighted so the gradient
    update is unbiased in expectation when beta=1.0.

    Note on current-state masking: forward() always requires a legal_mask,
    but the replay buffer does not store the current state's mask (it only
    stores next_legal_mask for target computation). Since we index directly
    into the taken action via gather(), masking the current state's Q-values
    has no effect on the loss — so we pass an all-True mask.

    Args:
        batch:      Dict from PrioritizedReplayBuffer.sample() — includes
                    "weights" (B,) float32 IS corrections and "indices" list[int].
        online_net: Network being trained.
        target_net: Frozen copy used for TD targets.
        device:     Torch device (mps or cpu) to run computation on.

    Returns:
        (loss tensor, mean Q-value, per-sample |TD error| as np.ndarray, indices list,
         move_entropy, wall_entropy)

        move_entropy: mean Shannon entropy of the softmax distribution over the 8 pawn
            move actions, averaged across the batch.  High = spreads probability across
            all 8 moves; low = concentrating on one direction.
        wall_entropy: same for the 128 wall placement actions (h-walls + v-walls).
            Tracking these separately reveals whether the agent explores wall placement
            or converges to placing walls in the same spots every time.
    """
    # Convert NumPy arrays to torch tensors and move to device.
    spatial          = torch.tensor(batch["spatial"]).to(device)               # (B, 4, 9, 9)
    scalars          = torch.tensor(batch["scalars"]).to(device)               # (B, 2)
    actions          = torch.tensor(batch["actions"]).to(device)               # (B,) int64
    rewards          = torch.tensor(batch["rewards"]).to(device)               # (B,)
    next_spatial     = torch.tensor(batch["next_spatial"]).to(device)          # (B, 4, 9, 9)
    next_scalars     = torch.tensor(batch["next_scalars"]).to(device)          # (B, 2)
    dones            = torch.tensor(batch["dones"]).float().to(device)         # (B,)
    next_legal_mask  = torch.tensor(batch["next_legal_mask"]).to(device)       # (B, 137)
    # IS correction weights from PER — down-weight frequently sampled transitions
    weights          = torch.tensor(batch["weights"]).to(device)               # (B,) float32

    B = spatial.shape[0]

    # All-True mask for current state — masking doesn't affect gather() on taken action
    all_true_mask = torch.ones(B, NUM_ACTIONS, dtype=torch.bool, device=device)

    # Q(s, a) — online network, select Q-value for the action that was actually taken
    q_all     = online_net(spatial, scalars, all_true_mask)         # (B, 137)
    q_current = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)    # (B,)

    # TD target: N-step Double DQN.
    # rewards already contains the N-step partial return: Σ_{i=0}^{N-1} γⁱ rᵢ
    # The bootstrap term must therefore be discounted by γᴺ (not γ¹) so the
    # full target is: r_n + γᴺ * Q_target(s_{t+N}, argmax_a Q_online(s_{t+N}, a))
    # Using γ instead of γᴺ overestimates future value — a consistent upward bias
    # that accumulates over training and partially defeats Double DQN's correction.
    # Decoupling selection (online) from evaluation (target) eliminates maximization
    # bias; the observed Q>1.0 peak (~1.5) in the first run came from using GAMMA
    # rather than GAMMA**N here combined with single-network target computation.
    # target_net is always in eval() mode; no gradient needed for either net here.
    with torch.no_grad():
        # Step 1: online net picks the greedy action in the next state
        q_next_online     = online_net(next_spatial, next_scalars, next_legal_mask)  # (B, 137)
        best_next_actions = q_next_online.argmax(dim=1)                              # (B,)
        # Step 2: target net evaluates that specific action (stable, lagged weights)
        q_next_target = target_net(next_spatial, next_scalars, next_legal_mask)      # (B, 137)
        q_next_max    = q_next_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)  # (B,)
        # γᴺ discount for N-step bootstrap; zero out bootstrap for terminal transitions
        target = rewards + GAMMA ** N_STEP_RETURNS * q_next_max * (1.0 - dones)     # (B,)

    # Per-sample Huber loss, then IS-weight each element before averaging.
    # reduction='none' gives a (B,) tensor; multiplying by IS weights and taking
    # the mean is equivalent to the IS-corrected gradient update from PER paper.
    per_loss = F.huber_loss(q_current, target, reduction="none", delta=1.0)   # (B,)
    loss     = (weights * per_loss).mean()

    # Per-sample |TD error| returned so the caller can update priorities.
    # Detach from the graph — we only need the scalar values for the SumTree.
    td_errors = (q_current - target).abs().detach().cpu().numpy()  # (B,)
    mean_q    = q_current.detach().mean().item()

    # Per-subspace Shannon entropy, averaged over the batch.
    # Softmax is applied independently within each subspace so the two entropies
    # are not competing — each is normalised to its own action count.
    # This lets us diagnose whether the agent is exploring wall placement
    # separately from whether it is exploring movement directions.
    with torch.no_grad():
        move_q = q_all[:, :H_WALL_OFFSET].detach()        # (B, 8)   — pawn moves
        wall_q = q_all[:, H_WALL_OFFSET:PASS_ACTION].detach()  # (B, 128) — h+v walls

        move_probs = F.softmax(move_q, dim=1)              # (B, 8)
        wall_probs = F.softmax(wall_q, dim=1)              # (B, 128)

        # entropy = -sum(p * log(p)), clamped to avoid log(0)
        move_entropy = -(move_probs * move_probs.clamp(min=1e-9).log()).sum(dim=1).mean().item()
        wall_entropy = -(wall_probs * wall_probs.clamp(min=1e-9).log()).sum(dim=1).mean().item()

    return loss, mean_q, td_errors, batch["indices"], move_entropy, wall_entropy


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: DQNModel, n_episodes: int, device: torch.device, bot) -> tuple[float, float]:
    """
    Evaluate the agent's win rate against the given bot.

    Runs n_episodes full games with a greedy (epsilon=0) policy.

    Args:
        model:      DQN network to evaluate.
        n_episodes: Number of evaluation games to run.
        device:     Torch device (mps or cpu) to run inference on.
        bot:        Opponent bot instance (must match the training opponent).

    Returns:
        (win_rate, mean_episode_length)
    """
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

            if reward > 0:  # +1.0 means agent (P0) won
                wins += 1
            total_length += ep_length

    model.train()
    win_rate = wins / n_episodes
    mean_length = total_length / n_episodes
    return win_rate, mean_length


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main(use_wandb: bool, args) -> None:
    """
    Run the full DQN training loop.

    Args:
        use_wandb: If True, log metrics to Weights & Biases.
                   Set to False for dry runs before W&B is configured.
    """
    # --- Device selection ---
    # Use Apple Silicon GPU (MPS) if available; fall back to CPU.
    # MPS is significantly faster than CPU for conv/linear layers on Apple Silicon.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- Reproducibility ---
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    print(f"Seed: {SEED}")

    # --- Initialise components ---
    if args.opponent == "random":
        from agents.random_bot import RandomBot
        bot = RandomBot()
        epsilon_start = EPSILON_START_RANDOM
        epsilon_decay = EPSILON_DECAY_RANDOM
    elif args.opponent == "heuristic":
        from agents.bot import HeuristicBot
        bot = HeuristicBot()
        epsilon_start = EPSILON_START_HEURISTIC
        epsilon_decay = EPSILON_DECAY_HEURISTIC
    else:  # self-play
        from agents.dqn_bot import DQNBot
        # Opponent starts as a deepcopy of the online net (or loaded checkpoint).
        # Weights are periodically synced from online_net every OPPONENT_UPDATE_FREQ steps.
        # Initialised after online_net is built below; placeholder set here.
        bot = None
        epsilon_start = EPSILON_START_SELF_PLAY
        epsilon_decay = EPSILON_DECAY_SELF_PLAY
    print(f"Opponent: {args.opponent}")

    online_net = DQNModel().to(device)
    target_net = copy.deepcopy(online_net)   # already on device via deepcopy
    target_net.eval()  # target net is never trained directly; always frozen

    # --- Optional checkpoint warm-start ---
    # Use this when transitioning between curriculum phases (e.g. RandomBot → self-play).
    # Both online and target nets are loaded so the target starts in sync with the online net.
    if args.checkpoint is not None:
        state_dict = torch.load(args.checkpoint, map_location=device)
        online_net.load_state_dict(state_dict)
        target_net.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {args.checkpoint}")

    # --- Finish self-play bot initialisation (needs warm-started online_net weights) ---
    # The DQNBot opponent starts as a copy of the (possibly warm-started) online net.
    # Weights are periodically synced every OPPONENT_UPDATE_FREQ steps so the opponent
    # stays competitive without causing non-stationarity between syncs.
    if args.opponent == "self-play":
        from agents.dqn_bot import DQNBot
        opponent_model = copy.deepcopy(online_net)
        opponent_model.eval()
        bot = DQNBot(opponent_model, device)

    env = QuoridorEnv(bot=bot)

    # Evaluation always uses HeuristicBot regardless of training opponent.
    # The graduation criterion (>80% vs heuristic) must be measured consistently
    # even during self-play training where the training opponent is a DQN clone.
    from agents.bot import HeuristicBot as _HeuristicBot
    eval_bot = _HeuristicBot()

    # Use a lower learning rate for the heuristic phase to preserve
    # warm-started weights and make conservative updates when adapting
    # to the harder opponent distribution.
    lr = LEARNING_RATE_HEURISTIC if args.opponent == "heuristic" else LEARNING_RATE
    print(f"Learning rate: {lr}")
    optimizer = torch.optim.Adam(online_net.parameters(), lr=lr)
    buffer    = PrioritizedReplayBuffer()

    os.makedirs("checkpoints", exist_ok=True)

    # --- W&B setup ---
    if use_wandb:
        import wandb
        wandb.init(
            project="quoridor-dqn",
            config={
                "device":             str(device),
                "opponent":           args.opponent,
                "checkpoint":         args.checkpoint,
                "learning_rate":      LEARNING_RATE,
                "gamma":              GAMMA,
                "batch_size":         BATCH_SIZE,
                "replay_buffer_size": buffer._tree.capacity,
                "target_update_freq": TARGET_UPDATE_FREQ,
                "gradient_clip_norm": GRADIENT_CLIP_NORM,
                "epsilon_start":      epsilon_start,
                "epsilon_end":        EPSILON_END,
                "epsilon_decay":      epsilon_decay,
                "max_steps":          MAX_STEPS,
                "eval_freq":          EVAL_FREQ,
                "eval_episodes":      EVAL_EPISODES,
                "seed":               SEED,
            },
        )

    # --- Training state ---
    epsilon       = epsilon_start
    total_steps   = 0
    best_win_rate = 0.0
    graduated     = False
    beta          = PER_BETA_START  # IS correction exponent; annealed toward 1.0 over training

    # N-step return buffer: accumulates the last N_STEP_RETURNS transitions
    # before computing a bootstrapped TD target. The terminal reward propagates
    # N steps backward in a single replay update instead of requiring N separate
    # gradient passes — critical for short sparse-reward episodes.
    n_step_buffer: deque = deque()

    print("Starting training...")
    print(f"Buffer fills after {BATCH_SIZE} steps; first update then.")

    # --- Main loop ---
    # Wrapped in try/finally so a Ctrl+C or crash always writes a checkpoint.
    try:
        while total_steps < MAX_STEPS and not graduated:
            spatial, scalars = env.reset()
            legal_mask = env.get_legal_mask()
            done = False

            while not done and total_steps < MAX_STEPS:

                # 1. Select action
                action = epsilon_greedy(online_net, spatial, scalars, legal_mask, epsilon, device)

                # 2. Step environment
                next_spatial, next_scalars, reward, done, info = env.step(action)
                next_legal_mask = info["legal_mask"]

                # 3. Accumulate transition in n-step buffer, then push to replay.
                # Once the buffer has N_STEP_RETURNS entries, compute the n-step
                # discounted return for the oldest transition and store that in the
                # replay buffer. s_{t+n} and done_{t+n} come from the newest entry.
                n_step_buffer.append({
                    "spatial":         spatial,
                    "scalars":         scalars,
                    "action":          action,
                    "reward":          reward,
                    "next_spatial":    next_spatial,
                    "next_scalars":    next_scalars,
                    "done":            done,
                    "next_legal_mask": next_legal_mask,
                })

                if len(n_step_buffer) >= N_STEP_RETURNS:
                    # Compute n-step discounted return from the oldest transition
                    n_step_reward = sum(
                        GAMMA ** i * n_step_buffer[i]["reward"]
                        for i in range(N_STEP_RETURNS)
                    )
                    oldest = n_step_buffer[0]
                    newest = n_step_buffer[-1]
                    buffer.push({
                        "spatial":         oldest["spatial"],
                        "scalars":         oldest["scalars"],
                        "action":          oldest["action"],
                        "reward":          n_step_reward,
                        "next_spatial":    newest["next_spatial"],
                        "next_scalars":    newest["next_scalars"],
                        "done":            newest["done"],
                        "next_legal_mask": newest["next_legal_mask"],
                    })
                    n_step_buffer.popleft()

                # On episode end, flush remaining transitions with truncated returns
                if done:
                    while n_step_buffer:
                        n = len(n_step_buffer)
                        n_step_reward = sum(
                            GAMMA ** i * n_step_buffer[i]["reward"]
                            for i in range(n)
                        )
                        oldest = n_step_buffer[0]
                        newest = n_step_buffer[-1]
                        buffer.push({
                            "spatial":         oldest["spatial"],
                            "scalars":         oldest["scalars"],
                            "action":          oldest["action"],
                            "reward":          n_step_reward,
                            "next_spatial":    newest["next_spatial"],
                            "next_scalars":    newest["next_scalars"],
                            "done":            newest["done"],
                            "next_legal_mask": newest["next_legal_mask"],
                        })
                        n_step_buffer.popleft()

                # 4. Train (only once buffer has enough transitions)
                if len(buffer) >= BATCH_SIZE:
                    online_net.train()
                    batch = buffer.sample(BATCH_SIZE, beta)
                    loss, mean_q, td_errors, indices, move_entropy, wall_entropy = compute_td_loss(
                        batch, online_net, target_net, device
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    grad_norm = clip_grad_norm_(online_net.parameters(), GRADIENT_CLIP_NORM)
                    optimizer.step()

                    # Update PER priorities with fresh TD errors so the SumTree
                    # reflects how wrong the network currently is on each transition.
                    buffer.update_priorities(indices, td_errors)

                    if use_wandb:
                        import wandb
                        wandb.log({
                            "train/loss":              loss.item(),
                            "train/epsilon":           epsilon,
                            "train/beta":              beta,
                            "train/mean_q_value":      mean_q,
                            "train/grad_norm":         grad_norm.item(),
                            # Subspace entropies: max for 8 moves ≈ 2.08, max for 128 walls ≈ 4.85
                            "train/move_entropy":      move_entropy,
                            "train/wall_entropy":      wall_entropy,
                        }, step=total_steps)

                # 5. Hard-update target network
                if total_steps % TARGET_UPDATE_FREQ == 0:
                    target_net.load_state_dict(online_net.state_dict())
                    target_net.eval()  # load_state_dict does not change self.training; keep target frozen

                # 6. Periodic opponent sync (self-play only)
                # Freeze a fresh copy of the online net as the new opponent every
                # OPPONENT_UPDATE_FREQ steps. The gap between syncs acts as a
                # stationary window for DQN to train against a stable target.
                if args.opponent == "self-play" and total_steps % OPPONENT_UPDATE_FREQ == 0:
                    bot.update_weights(online_net.state_dict())

                # 7. Periodic evaluation — always vs HeuristicBot for consistent graduation metric
                if total_steps > 0 and total_steps % EVAL_FREQ == 0:
                    win_rate, mean_length = evaluate(online_net, EVAL_EPISODES, device, eval_bot)
                    print(
                        f"Step {total_steps:>7} | "
                        f"win_rate={win_rate:.2%} | "
                        f"eps={epsilon:.3f} | "
                        f"beta={beta:.3f} | "
                        f"mean_ep_len={mean_length:.1f}"
                    )

                    if use_wandb:
                        import wandb
                        wandb.log({
                            "eval/win_rate":            win_rate,
                            "eval/mean_episode_length": mean_length,
                        }, step=total_steps)

                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        torch.save(online_net.state_dict(), "checkpoints/best.pt")
                        print(f"  ✓ New best win rate ({win_rate:.2%}) — checkpoint saved.")

                    if win_rate >= WIN_RATE_TARGET:
                        print(f"\nGraduated at step {total_steps}! Win rate: {win_rate:.2%}")
                        graduated = True
                        break

                # 8. Advance to next state
                spatial, scalars, legal_mask = next_spatial, next_scalars, next_legal_mask
                total_steps += 1
                # Decay per step (not per episode) so the schedule is independent of episode length.
                # First run showed ~10 steps/episode vs the assumed 30, causing 3x faster decay.
                epsilon = max(EPSILON_END, epsilon * epsilon_decay)
                # Anneal beta from PER_BETA_START → 1.0 linearly over MAX_STEPS.
                # Full correction (beta=1) at the end of training ensures the gradient
                # is unbiased once the network has stabilised.
                beta = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * min(
                    1.0, total_steps / MAX_STEPS
                )

        # --- Normal completion checkpoint ---
        final_path = f"checkpoints/final_{total_steps}.pt"
        torch.save(online_net.state_dict(), final_path)
        print(f"\nTraining complete. Final checkpoint: {final_path}")
        print(f"Best win rate achieved: {best_win_rate:.2%}")

    finally:
        # Always runs on normal exit, Ctrl+C, or unhandled exception.
        # Saves current weights so no progress is lost when interrupting a run.
        emergency_path = f"checkpoints/emergency_{total_steps}.pt"
        torch.save(online_net.state_dict(), emergency_path)
        print(f"\nCheckpoint saved: {emergency_path}")
        if use_wandb:
            import wandb
            wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN agent for Quoridor.")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging (use for dry runs before wandb is configured).",
    )
    parser.add_argument(
        "--opponent",
        choices=["random", "heuristic", "self-play"],
        default="random",
        help=(
            "Opponent to train against. "
            "'random' for curriculum phase 1, "
            "'self-play' for phase 2 (frozen DQN clone, periodically updated), "
            "'heuristic' for direct heuristic training."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint to warm-start from (e.g. checkpoints/best.pt).",
    )
    args = parser.parse_args()
    main(use_wandb=not args.no_wandb, args=args)
