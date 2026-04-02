"""
train_ppo.py — PPO training loop for the Quoridor agent.

Why PPO over DQN for this problem?
    DQN failed because sparse rewards (+1/-1 only) combined with a strong
    heuristic opponent meant the agent almost never won, so Q-values collapsed
    to a flat -1 fixed point and gradient signal disappeared. PPO avoids this
    via entropy regularization: the policy is explicitly penalized for becoming
    too deterministic, so it keeps exploring even when every observed outcome is -1.

Algorithm (Schulman et al., 2017 — Proximal Policy Optimization):
    1. Collect ROLLOUT_STEPS transitions using the current policy π_θ.
       Store: spatial, scalars, legal_mask, action, log_prob, reward, done, value.
    2. Compute Generalized Advantage Estimates (GAE) for the full rollout.
       Returns = advantages + values (used as value-head targets).
    3. For PPO_EPOCHS gradient passes over the rollout in mini-batches:
         a. Recompute log_probs and values under the current (updated) π_θ.
         b. Ratio:        r_t = exp(log_prob_new - log_prob_old)
         c. Policy loss:  -E[min(r_t*A_t, clip(r_t, 1-ε, 1+ε)*A_t)]
         d. Value loss:   MSE(V_θ(s_t), returns_t)
         e. Entropy:      -entropy_coef * H[π_θ(·|s_t)]
         f. Total:        policy_loss + value_coef*value_loss + entropy_loss
         g. Gradient clip + Adam step.
    4. Discard rollout — PPO is on-policy, no replay buffer.

BC warm-start:
    Pass --checkpoint checkpoints/bc_pretrained.pt to load the CNN backbone
    from BC pretraining. The actor and value heads are randomly initialized.
    This gives PPO a 20% starting win rate against HeuristicBot (verified),
    vs 0% cold start.

Usage:
    python -m scripts.train_ppo                               # requires wandb login
    python -m scripts.train_ppo --no-wandb                   # dry run
    python -m scripts.train_ppo --checkpoint checkpoints/bc_pretrained.pt
    python -m scripts.train_ppo --no-wandb --max-steps 1024  # smoke test (2 updates)
"""

import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from agents.bot import HeuristicBot
from agents.random_bot import RandomBot
from agents.ppo_model import PPOModel
from agents.ppo_model_resnet import PPOModelResNet
from agents.ppo_model_bfs import PPOModelBFS
from agents.ppo_model_bfs_resnet import PPOModelBFSResNet
from config import (
    EVAL_EPISODES,
    EVAL_FREQ,
    FENCE_GRID,
    NUM_CHANNELS_BFS,
    OPPONENT_UPDATE_FREQ,
    PPO_CLIP_EPS,
    PPO_ENTROPY_COEF_END,
    PPO_ENTROPY_COEF_START,
    PPO_EPOCHS,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_GRAD_CLIP,
    PPO_LR,
    PPO_MAX_STEPS,
    PPO_MINI_BATCH_SIZE,
    PPO_VALUE_COEF,
    ROLLOUT_STEPS,
    SEED,
    WIN_RATE_TARGET,
)
from quoridor.action_encoding import index_to_action
from quoridor.env import QuoridorEnv


# ---------------------------------------------------------------------------
# Self-play opponent wrapper
# ---------------------------------------------------------------------------

class PPOBot:
    """Wraps a frozen PPOModel so it can serve as a QuoridorEnv opponent bot.

    Used during self-play: a frozen copy of the training model acts as P1.
    choose_action() samples from the frozen policy (not greedy) so the opponent
    maintains some exploration and doesn't become a fixed deterministic target.

    The use_bfs flag is inferred from the first conv layer's input channels so
    this wrapper works with both 4-channel and 6-channel model variants.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model  = model
        self.device = device
        # Detect whether this model expects 6-channel (BFS) or 4-channel input.
        first_weight = next(iter(model.parameters()))
        self.use_bfs = (first_weight.shape[1] == NUM_CHANNELS_BFS)

    def reset(self) -> None:
        pass  # PPO model is stateless across episodes

    def choose_action(self, game) -> tuple:
        spatial, scalars = game.get_observation(use_bfs=self.use_bfs)
        legal_mask = game.get_legal_mask()

        s  = torch.tensor(spatial,    dtype=torch.float32).unsqueeze(0).to(self.device)
        sc = torch.tensor(scalars,    dtype=torch.float32).unsqueeze(0).to(self.device)
        m  = torch.tensor(legal_mask, dtype=torch.bool   ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist, _ = self.model(s, sc, m)
            action_idx = int(dist.sample().item())

        action = index_to_action(action_idx)

        # The observation was returned from the current player's perspective:
        # when game.turn == 1 the board is flipped (rows 0↔8) before the model
        # sees it. That flip is applied inside game.get_observation(), so the
        # model's direction deltas and fence rows are in flipped space.
        # We must un-flip them here before resolving to actual board coordinates.
        flip = (game.turn == 1)

        # index_to_action returns ("move", dr, dc) with direction deltas, but
        # _dispatch_bot_action (like HeuristicBot) expects ("move", dest_r, dest_c)
        # with absolute board coordinates. Resolve the direction to a destination.
        if action[0] == "move":
            _, dr, dc = action
            # Vertical axis is flipped when flip=True (np.flipud): negate dr.
            # Columns are not affected by flipud, so dc is unchanged.
            if flip:
                dr = -dr
            cur_r = int(game.pos[game.turn, 0])
            cur_c = int(game.pos[game.turn, 1])
            for dest_r, dest_c in game.get_valid_moves():
                if (np.sign(dest_r - cur_r) == np.sign(dr)
                        and np.sign(dest_c - cur_c) == np.sign(dc)):
                    return ("move", dest_r, dest_c)
            # Fallback: move forward toward goal (should not occur with valid mask)
            return ("move", cur_r + dr, cur_c + dc)

        if action[0] == "fence":
            _, r, c, ori = action
            # Fence rows in flipped space map to actual row = FENCE_GRID - 1 - r.
            if flip:
                r = FENCE_GRID - 1 - r
            return ("fence", r, c, ori)

        return action  # pass action needs no conversion


# ---------------------------------------------------------------------------
# Model + environment factory
# ---------------------------------------------------------------------------

# Maps --model flag values to (ModelClass, use_bfs) pairs.
# BFS models must be paired with use_bfs=True so the env returns 6-channel
# observations. Non-BFS models use the standard 4-channel env.
_MODEL_REGISTRY: dict = {
    "baseline":   (PPOModel,          False),
    "resnet":     (PPOModelResNet,     False),
    "bfs":        (PPOModelBFS,        True),
    "bfs_resnet": (PPOModelBFSResNet,  True),
}


def build_model_and_env(
    model_type:          str,
    bot,
    device:              torch.device,
    use_reward_shaping:  bool = False,
) -> tuple:
    """
    Instantiate the correct model and environment pair for a given model type.

    Keeping this in one place ensures the model channel count, env BFS flag,
    and reward shaping flag are always in sync.

    Args:
        model_type:         One of "baseline", "resnet", "bfs", "bfs_resnet".
        bot:                Opponent bot instance passed to QuoridorEnv.
        device:             Torch device; model is moved to device before returning.
        use_reward_shaping: If True, training env uses path-length delta shaping.
                            Eval env is always unshapen so win rate is comparable
                            across runs with and without shaping.

    Returns:
        (model, train_env, eval_env)
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Choose from: {list(_MODEL_REGISTRY.keys())}"
        )
    ModelClass, use_bfs = _MODEL_REGISTRY[model_type]
    model    = ModelClass().to(device)
    env      = QuoridorEnv(bot=bot,            use_bfs=use_bfs, use_reward_shaping=use_reward_shaping)
    eval_env = QuoridorEnv(bot=HeuristicBot(), use_bfs=use_bfs, use_reward_shaping=False)
    return model, env, eval_env


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout(
    model: PPOModel,
    env: QuoridorEnv,
    rollout_steps: int,
    device: torch.device,
    spatial: np.ndarray,
    scalars: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray, bool]:
    """
    Run the current policy for rollout_steps environment steps.

    May span multiple episodes. Episode boundaries are tracked via the 'done'
    flag — GAE resets the advantage accumulation at each terminal step.

    Args:
        model:         PPOModel in eval mode during collection.
        env:           QuoridorEnv with bot opponent.
        rollout_steps: Number of env steps to collect (ROLLOUT_STEPS).
        device:        Torch device for inference.
        spatial:       Current observation spatial component (carried across calls).
        scalars:       Current observation scalars component (carried across calls).

    Returns:
        rollout:       Dict of tensors, each shape (T, ...).
        next_spatial:  Observation after the final step (for GAE bootstrap).
        next_scalars:  Scalars after the final step.
        last_done:     True if the rollout ended on a terminal step.
    """
    spatials:    list = []
    scalars_lst: list = []
    legal_masks: list = []
    actions:     list = []
    log_probs:   list = []
    rewards:     list = []
    dones:       list = []
    values:      list = []

    model.eval()

    for _ in range(rollout_steps):
        legal_mask = env.get_legal_mask()

        with torch.no_grad():
            spatial_t = torch.tensor(spatial).unsqueeze(0).to(device)      # (1, 4, 9, 9)
            scalars_t = torch.tensor(scalars).unsqueeze(0).to(device)      # (1, 2)
            mask_t    = torch.tensor(legal_mask).unsqueeze(0).to(device)   # (1, 137)
            dist, value = model(spatial_t, scalars_t, mask_t)
            action    = dist.sample()                                       # (1,)
            log_prob  = dist.log_prob(action)                               # (1,)

        action_int = int(action.item())

        next_spatial, next_scalars, reward, done, info = env.step(action_int)

        # Record current step
        spatials.append(spatial)
        scalars_lst.append(scalars)
        legal_masks.append(legal_mask)
        actions.append(action_int)
        log_probs.append(float(log_prob.item()))
        rewards.append(float(reward))
        dones.append(done)
        values.append(float(value.item()))

        if done:
            # Reset env; next iteration starts a new episode
            spatial, scalars = env.reset()
        else:
            spatial, scalars = next_spatial, next_scalars

    # Stack all lists into tensors — shape (T, ...) for each field
    rollout = {
        "spatial":    torch.tensor(np.stack(spatials),    dtype=torch.float32).to(device),  # (T, 4, 9, 9)
        "scalars":    torch.tensor(np.stack(scalars_lst), dtype=torch.float32).to(device),  # (T, 2)
        "legal_mask": torch.tensor(np.stack(legal_masks), dtype=torch.bool).to(device),     # (T, 137)
        "actions":    torch.tensor(actions,               dtype=torch.long).to(device),     # (T,)
        "log_probs":  torch.tensor(log_probs,             dtype=torch.float32).to(device),  # (T,)
        "rewards":    torch.tensor(rewards,               dtype=torch.float32).to(device),  # (T,)
        "dones":      torch.tensor(dones,                 dtype=torch.bool).to(device),     # (T,)
        "values":     torch.tensor(values,                dtype=torch.float32).to(device),  # (T,)
    }

    return rollout, spatial, scalars, dones[-1]


# ---------------------------------------------------------------------------
# Generalized Advantage Estimation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards:    torch.Tensor,  # (T,)
    values:     torch.Tensor,  # (T,)
    dones:      torch.Tensor,  # (T,) bool
    last_value: torch.Tensor,  # scalar — V(s_{T+1}); 0 if last step was terminal
    gamma:      float,
    lam:        float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimates (GAE, Schulman et al. 2016).

    GAE smoothly interpolates between Monte Carlo (λ=1, low bias, high variance)
    and TD(0) (λ=0, high bias, low variance). λ=0.95 sits close to MC, giving
    accurate credit assignment for Quoridor's long-horizon rewards.

    For each step t (walking backwards):
        delta_t = r_t + γ * V(s_{t+1}) * (1-done_t) - V(s_t)
        A_t     = delta_t + γ * λ * (1-done_t) * A_{t+1}

    The (1-done_t) term zeros out both the bootstrap value and the carry-over
    advantage at episode boundaries, so credit never bleeds between episodes.

    Returns:
        advantages : (T,) — not yet normalised (normalisation is per-mini-batch)
        returns    : (T,) — advantages + values; used as value-head targets
    """
    T          = rewards.shape[0]
    advantages = torch.zeros(T, device=rewards.device)
    gae        = 0.0

    for t in reversed(range(T)):
        not_done = 1.0 - float(dones[t])
        next_val = float(values[t + 1]) if t < T - 1 else float(last_value)
        delta    = float(rewards[t]) + gamma * next_val * not_done - float(values[t])
        gae      = delta + gamma * lam * not_done * gae
        advantages[t] = gae

    returns = advantages + values  # (T,) — value-head training targets
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    model:           PPOModel,
    optimizer:       torch.optim.Optimizer,
    rollout:         dict,
    advantages:      torch.Tensor,  # (T,)
    returns:         torch.Tensor,  # (T,)
    ppo_epochs:      int,
    mini_batch_size: int,
    clip_eps:        float,
    value_coef:      float,
    entropy_coef:    float,
    grad_clip:       float,
    device:          torch.device,
) -> dict:
    """
    Run PPO_EPOCHS gradient passes over the collected rollout.

    Each epoch shuffles the rollout and iterates mini-batches. Within each
    mini-batch, new log_probs and values are recomputed under the current
    (possibly updated) policy. The PPO clip objective prevents the policy
    from moving too far from the collection policy in a single update.

    Returns:
        metrics dict with: policy_loss, value_loss, entropy, approx_kl,
                           clip_fraction, grad_norm (means over last epoch)
    """
    T = rollout["actions"].shape[0]

    # Accumulate metrics over the final epoch for logging.
    # move_prob_mass / wall_prob_mass: average fraction of probability mass on
    # pawn-move actions (0–7) vs wall-placement actions (8–135). If wall_prob_mass
    # collapses to ~0 the policy has stopped considering walls entirely.
    # move_entropy / wall_entropy: conditional entropy within each action type,
    # computed by renormalising probs to that subset. Tracks diversity of choices
    # within each type independently of overall probability mass.
    metrics = {k: 0.0 for k in [
        "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm",
        "move_prob_mass", "wall_prob_mass", "move_entropy", "wall_entropy",
    ]}
    n_batches = 0

    model.train()

    for epoch in range(ppo_epochs):
        indices = torch.randperm(T, device=device)

        for start in range(0, T, mini_batch_size):
            mb_idx = indices[start : start + mini_batch_size]

            mb_spatial   = rollout["spatial"][mb_idx]    # (B, 4, 9, 9)
            mb_scalars   = rollout["scalars"][mb_idx]    # (B, 2)
            mb_mask      = rollout["legal_mask"][mb_idx] # (B, 137)
            mb_actions   = rollout["actions"][mb_idx]    # (B,)
            mb_old_logp  = rollout["log_probs"][mb_idx]  # (B,)
            mb_adv       = advantages[mb_idx]            # (B,) — normalised per-rollout in main()
            mb_returns   = returns[mb_idx]               # (B,)

            # Forward pass under CURRENT policy weights
            dist, value = model(mb_spatial, mb_scalars, mb_mask)
            new_log_probs = dist.log_prob(mb_actions)   # (B,)
            entropy       = dist.entropy()              # (B,)

            # PPO clip objective
            # ratio = π_new(a|s) / π_old(a|s); in log-space to avoid exp overflow
            ratio    = torch.exp(new_log_probs - mb_old_logp)          # (B,)
            surr1    = ratio * mb_adv                                   # (B,)
            surr2    = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss: MSE between critic prediction and GAE returns
            value_loss = F.mse_loss(value.squeeze(-1), mb_returns)

            # Entropy bonus: maximise entropy (subtract from loss)
            entropy_loss = -entropy_coef * entropy.mean()

            loss = policy_loss + value_coef * value_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Track diagnostics — only accumulate from the last epoch so the
            # metrics reflect the final policy state after all updates.
            if epoch == ppo_epochs - 1:
                with torch.no_grad():
                    # approx_kl: first-order approximation of KL(π_old || π_new)
                    # Should stay below 0.02–0.05; if higher, policy is changing too fast.
                    approx_kl     = ((ratio - 1) - (new_log_probs - mb_old_logp)).mean()
                    # clip_fraction: fraction of ratios outside [1-ε, 1+ε]
                    # Should be 10–30%; if > 50%, policy is too aggressive.
                    clip_fraction = ((ratio - 1).abs() > clip_eps).float().mean()

                    # Per-action-type diagnostics.
                    # Action layout: 0–7 = pawn moves, 8–135 = wall placements, 136 = pass.
                    # move_prob_mass / wall_prob_mass: fraction of total probability on each
                    # type. wall_prob_mass collapsing to ~0 signals wall-action collapse.
                    # move_entropy / wall_entropy: entropy within each type (renormalised).
                    # Tracks diversity of choices independent of probability mass.
                    probs = dist.probs                                         # (B, 137)
                    move_p = probs[:, :8]                                      # (B, 8)
                    wall_p = probs[:, 8:136]                                   # (B, 128)

                    move_prob_mass = move_p.sum(dim=1).mean()
                    wall_prob_mass = wall_p.sum(dim=1).mean()

                    move_p_norm = move_p / (move_p.sum(dim=1, keepdim=True) + 1e-8)
                    wall_p_norm = wall_p / (wall_p.sum(dim=1, keepdim=True) + 1e-8)
                    move_entropy = -(move_p_norm * (move_p_norm + 1e-8).log()).sum(dim=1).mean()
                    wall_entropy = -(wall_p_norm * (wall_p_norm + 1e-8).log()).sum(dim=1).mean()

                metrics["policy_loss"]    += policy_loss.item()
                metrics["value_loss"]     += value_loss.item()
                metrics["entropy"]        += entropy.mean().item()
                metrics["approx_kl"]      += approx_kl.item()
                metrics["clip_fraction"]  += clip_fraction.item()
                metrics["grad_norm"]      += grad_norm.item()
                metrics["move_prob_mass"] += move_prob_mass.item()
                metrics["wall_prob_mass"] += wall_prob_mass.item()
                metrics["move_entropy"]   += move_entropy.item()
                metrics["wall_entropy"]   += wall_entropy.item()
                n_batches += 1

    # Average over mini-batches in the final epoch
    if n_batches > 0:
        for k in metrics:
            metrics[k] /= n_batches

    return metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model:      PPOModel,
    n_episodes: int,
    device:     torch.device,
    eval_env:   QuoridorEnv,
) -> tuple[float, float]:
    """
    Evaluate the agent's greedy win rate using a pre-built eval environment.

    The env is passed in (rather than a bot) so the caller can control use_bfs.
    BFS models must evaluate against a use_bfs=True env — passing a bot here
    and constructing the env internally would silently use the wrong channel count.

    Uses model.eval() for consistent inference (important for LayerNorm models too,
    since eval() disables dropout if any is added later).

    Returns:
        (win_rate, mean_episode_length)
    """
    model.eval()
    wins         = 0
    total_length = 0

    with torch.no_grad():
        for _ in range(n_episodes):
            spatial, scalars = eval_env.reset()
            done     = False
            ep_length = 0

            while not done:
                legal_mask = eval_env.get_legal_mask()
                spatial_t  = torch.tensor(spatial).unsqueeze(0).to(device)
                scalars_t  = torch.tensor(scalars).unsqueeze(0).to(device)
                mask_t     = torch.tensor(legal_mask).unsqueeze(0).to(device)

                dist, _ = model(spatial_t, scalars_t, mask_t)
                action  = int(dist.probs.argmax(dim=1).item())  # greedy

                spatial, scalars, reward, done, _ = eval_env.step(action)
                ep_length += 1

            if reward > 0:
                wins += 1
            total_length += ep_length

    model.train()
    return wins / n_episodes, total_length / n_episodes


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main(use_wandb: bool, args) -> None:
    """Run the full PPO training loop."""

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # --- Reproducibility ---
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    print(f"Seed: {SEED}")

    # --- Model + environment ---
    # build_model_and_env() pairs the correct model class with the correct env
    # BFS flag so channel counts always match. Never construct them separately.
    bot               = HeuristicBot()
    model, env, eval_env_obj = build_model_and_env(args.model, bot, device, args.reward_shaping)
    print(f"Model: {args.model}  |  BFS env: {_MODEL_REGISTRY[args.model][1]}  |  "
          f"Opponent: {args.opponent}  |  Reward shaping: {args.reward_shaping}")

    # Self-play setup: replace training env's bot with a frozen copy of the model.
    # The frozen copy is updated every OPPONENT_UPDATE_FREQ steps so the opponent
    # gradually gets stronger as the agent improves — the core AlphaZero insight.
    frozen_model = None
    if args.opponent == "self_play":
        frozen_model = copy.deepcopy(model)
        frozen_model.eval()
        env.bot = PPOBot(frozen_model, device)
        print(f"Self-play: frozen opponent initialized. "
              f"Updates every {OPPONENT_UPDATE_FREQ} steps.")

    # Optional BC warm-start: load CNN backbone weights from bc_pretrained.pt.
    # strict=False silently skips keys whose shapes don't match (e.g. the first
    # conv layer when using 6-channel BFS models vs a 4-channel checkpoint).
    if args.checkpoint is not None:
        bc_state     = torch.load(args.checkpoint, map_location=device)
        backbone     = {k: v for k, v in bc_state.items() if k.startswith("conv.")}
        missing, _   = model.load_state_dict(backbone, strict=False)
        print(f"Loaded backbone from {args.checkpoint} ({len(backbone)} tensors).")
        print(f"  Randomly initialized: {len(missing)} tensors (actor + value heads).")

    # --- Optimizer ---
    # Single Adam over all parameters (backbone + both heads) — standard PPO practice.
    optimizer = torch.optim.Adam(model.parameters(), lr=PPO_LR)

    os.makedirs("checkpoints", exist_ok=True)

    # --- W&B ---
    if use_wandb:
        import wandb
        wandb.init(
            project="quoridor-ppo",
            name=args.run_name,   # None → wandb auto-names; set via --run-name for batch runs
            config={
                "model":              args.model,
                "opponent":           args.opponent,
                "reward_shaping":     args.reward_shaping,
                "device":             str(device),
                "checkpoint":         args.checkpoint,
                "rollout_steps":      ROLLOUT_STEPS,
                "ppo_epochs":         PPO_EPOCHS,
                "mini_batch_size":    PPO_MINI_BATCH_SIZE,
                "clip_eps":              PPO_CLIP_EPS,
                "gae_lambda":            args.lam,
                "gamma":                 args.gamma,
                "entropy_coef_start":    PPO_ENTROPY_COEF_START,
                "entropy_coef_end":      PPO_ENTROPY_COEF_END,
                "value_coef":            PPO_VALUE_COEF,
                "lr":                    PPO_LR,
                "grad_clip":             args.grad_clip,
                "run_name":              args.run_name,
                "max_steps":          args.max_steps,
                "eval_freq":          EVAL_FREQ,
                "eval_episodes":      EVAL_EPISODES,
                "seed":               SEED,
            },
        )

    # --- Training state ---
    total_steps   = 0
    best_win_rate = 0.0
    graduated     = False

    # Carry the current observation across rollout calls
    spatial, scalars = env.reset()

    print("Starting PPO training...")
    print(f"Each rollout: {ROLLOUT_STEPS} steps → {PPO_EPOCHS} epochs × "
          f"{ROLLOUT_STEPS // PPO_MINI_BATCH_SIZE} mini-batches")

    try:
        while total_steps < args.max_steps and not graduated:

            # 1. Collect rollout
            rollout, spatial, scalars, last_done = collect_rollout(
                model, env, ROLLOUT_STEPS, device, spatial, scalars
            )

            # 2. Bootstrap value for the step *after* the rollout ends.
            #    If the last step was terminal, the true bootstrap value is 0.
            if last_done:
                last_value = torch.tensor(0.0, device=device)
            else:
                with torch.no_grad():
                    model.eval()
                    s_t  = torch.tensor(spatial).unsqueeze(0).to(device)
                    sc_t = torch.tensor(scalars).unsqueeze(0).to(device)
                    last_value = model.get_value(s_t, sc_t).squeeze()   # scalar

            # 3. GAE — uses args.gamma/lam so CLI overrides work for experiments.
            advantages, returns = compute_gae(
                rollout["rewards"], rollout["values"], rollout["dones"],
                last_value, args.gamma, args.lam,
            )

            # Normalise advantages per-rollout (not per-minibatch).
            # With sparse ±1 rewards, many minibatches have near-constant advantages.
            # Per-minibatch normalisation would amplify that near-zero signal to
            # unit variance, creating artificial gradient signal from noise.
            # Per-rollout normalisation only amplifies real variance across the
            # full 512-step collection window.
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 4. Entropy coefficient — linearly decayed over training.
            # High early to prevent wall-action collapse; low late to allow
            # the policy to specialise once it has learned which actions matter.
            entropy_coef = PPO_ENTROPY_COEF_START + (
                PPO_ENTROPY_COEF_END - PPO_ENTROPY_COEF_START
            ) * min(1.0, total_steps / args.max_steps)

            # 5. PPO update — uses args.grad_clip so CLI override works.
            metrics = ppo_update(
                model, optimizer, rollout, advantages, returns,
                PPO_EPOCHS, PPO_MINI_BATCH_SIZE,
                PPO_CLIP_EPS, PPO_VALUE_COEF, entropy_coef,
                args.grad_clip, device,
            )

            prev_steps   = total_steps
            total_steps += ROLLOUT_STEPS

            # Self-play opponent sync: periodically copy online model → frozen opponent.
            # The modulo check fires once per OPPONENT_UPDATE_FREQ boundary crossing,
            # same pattern as the eval trigger above.
            if (args.opponent == "self_play"
                    and total_steps // OPPONENT_UPDATE_FREQ > prev_steps // OPPONENT_UPDATE_FREQ):
                frozen_model.load_state_dict(copy.deepcopy(model.state_dict()))
                frozen_model.eval()
                env.bot = PPOBot(frozen_model, device)
                print(f"[Self-play] Opponent synced at step {total_steps}")
                if use_wandb:
                    import wandb
                    wandb.log({"train/self_play_sync": total_steps}, step=total_steps)

            if use_wandb:
                import wandb
                wandb.log({
                    "train/policy_loss":   metrics["policy_loss"],
                    "train/value_loss":    metrics["value_loss"],
                    "train/entropy":       metrics["entropy"],
                    "train/approx_kl":     metrics["approx_kl"],
                    "train/clip_fraction": metrics["clip_fraction"],
                    "train/grad_norm":     metrics["grad_norm"],
                    # Entropy schedule — lets you see the current coefficient
                    # alongside entropy so you can tell if collapse is driven
                    # by the schedule or by the policy itself.
                    "train/entropy_coef":  entropy_coef,
                    # Per-action-type diagnostics — early warning for wall collapse.
                    # wall_prob_mass < 0.1 means the policy has stopped considering walls.
                    "train/move_prob_mass": metrics["move_prob_mass"],
                    "train/wall_prob_mass": metrics["wall_prob_mass"],
                    "train/move_entropy":   metrics["move_entropy"],
                    "train/wall_entropy":   metrics["wall_entropy"],
                }, step=total_steps)

            # 5. Periodic evaluation — trigger when we cross an EVAL_FREQ boundary.
            # total_steps advances in ROLLOUT_STEPS increments, so modulo-based
            # checks would only fire at LCM(ROLLOUT_STEPS, EVAL_FREQ) intervals.
            if total_steps // EVAL_FREQ > prev_steps // EVAL_FREQ:
                win_rate, mean_length = evaluate(model, EVAL_EPISODES, device, eval_env_obj)
                print(
                    f"Step {total_steps:>7} | "
                    f"win_rate={win_rate:.2%} | "
                    f"entropy={metrics['entropy']:.3f} | "
                    f"kl={metrics['approx_kl']:.4f} | "
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
                    torch.save(model.state_dict(), "checkpoints/best_ppo.pt")
                    print(f"  ✓ New best win rate ({win_rate:.2%}) — checkpoint saved.")

                if win_rate >= WIN_RATE_TARGET:
                    print(f"\nGraduated at step {total_steps}! Win rate: {win_rate:.2%}")
                    graduated = True

        # Normal completion
        final_path = f"checkpoints/ppo_final_{total_steps}.pt"
        torch.save(model.state_dict(), final_path)
        print(f"\nTraining complete. Final checkpoint: {final_path}")
        print(f"Best win rate: {best_win_rate:.2%}")

    finally:
        # Always saves on normal exit, Ctrl+C, or crash.
        emergency_path = f"checkpoints/ppo_emergency_{total_steps}.pt"
        torch.save(model.state_dict(), emergency_path)
        print(f"\nCheckpoint saved: {emergency_path}")
        if use_wandb:
            import wandb
            wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for Quoridor.")
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint to warm-start the CNN backbone from "
             "(e.g. checkpoints/bc_pretrained.pt).",
    )
    parser.add_argument(
        "--reward-shaping",
        action="store_true",
        dest="reward_shaping",
        help=(
            "Enable path-length delta reward shaping on non-terminal steps. "
            "Adds a small reward for lengthening the opponent's shortest path "
            "and shortening the agent's own path. Terminal rewards stay pure ±1. "
            "Eval env is never shaped so win rate remains comparable across runs."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "resnet", "bfs", "bfs_resnet"],
        help=(
            "Model architecture to train. "
            "'baseline': 4-ch CNN + BatchNorm (original). "
            "'resnet': 4-ch ResNet + LayerNorm. "
            "'bfs': 6-ch CNN + LayerNorm (adds BFS distance maps). "
            "'bfs_resnet': 6-ch ResNet + LayerNorm (BFS + residual, recommended). "
            "BFS variants automatically use QuoridorEnv(use_bfs=True)."
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=PPO_MAX_STEPS,
        dest="max_steps",
        help=f"Override total training steps (default: {PPO_MAX_STEPS}). "
             "Use a small value (e.g. 1024) for smoke tests.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=PPO_GAMMA,
        help=f"Discount factor (default: {PPO_GAMMA}). "
             "Try 0.999 for longer effective horizon on 20-40 move games.",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=PPO_GAE_LAMBDA,
        help=f"GAE lambda smoothing (default: {PPO_GAE_LAMBDA}). "
             "Try 0.97 alongside --gamma 0.999 for longer credit assignment horizon.",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=PPO_GRAD_CLIP,
        dest="grad_clip",
        help=f"Gradient norm clip threshold (default: {PPO_GRAD_CLIP}). "
             "Try 0.5 for tighter trust-region enforcement.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        dest="run_name",
        help="W&B run display name. Set by run_batch.sh to identify experiments. "
             "If omitted, W&B auto-generates a name.",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="heuristic",
        choices=["heuristic", "self_play"],
        help=(
            "Training opponent strategy. "
            "'heuristic': always train vs HeuristicBot (default). "
            "'self_play': train vs a frozen copy of self (AlphaZero style), "
            f"updated every OPPONENT_UPDATE_FREQ={OPPONENT_UPDATE_FREQ} steps. "
            "Eval always uses HeuristicBot regardless of this flag."
        ),
    )
    args = parser.parse_args()
    main(use_wandb=not args.no_wandb, args=args)
