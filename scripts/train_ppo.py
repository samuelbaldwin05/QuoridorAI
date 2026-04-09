"""
train_ppo.py — PPO training loop for the Quoridor agent.

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

Usage:
    python -m scripts.train_ppo                               # requires wandb login
    python -m scripts.train_ppo --no-wandb                   # dry run
    python -m scripts.train_ppo --checkpoint checkpoints/best_mixed-v2.pt
    python -m scripts.train_ppo --no-wandb --max-steps 1024  # smoke test (2 updates)
"""

import argparse
import collections
import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from agents.bot import EpsilonHeuristicBot, HeuristicBot
from agents.ppo_model import PPOModel
from agents.ppo_model_resnet import PPOModelResNet
from agents.ppo_model_bfs import PPOModelBFS
from agents.ppo_model_bfs_resnet import PPOModelBFSResNet
from config import (
    AUX_LOSS_COEF,
    EPSILON_CURRICULUM_START,
    EPSILON_CURRICULUM_THRESHOLD,
    EVAL_EPISODES,
    EVAL_FREQ,
    FENCE_GRID,
    NUM_CHANNELS_BFS,
    OPPONENT_HEURISTIC_RATIO,
    OPPONENT_POOL_SIZE,
    OPPONENT_UPDATE_FREQ,
    PPO_CLIP_EPS,
    PPO_ENTROPY_COEF_END,
    PPO_ENTROPY_COEF_START,
    PPO_MOVE_ENTROPY_COEF,
    PPO_WALL_ENTROPY_COEF,
    PPO_EPOCHS,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_GRAD_CLIP,
    PPO_LR,
    PPO_MAX_STEPS,
    PPO_MINI_BATCH_SIZE,
    PPO_VALUE_COEF,
    REWARD_SHAPING_OPP_COEF,
    REWARD_SHAPING_SELF_COEF,
    ROLLOUT_STEPS,
    SEED,
    WIN_RATE_TARGET,
)
from quoridor.action_encoding import (
    index_to_action,
    H_WALL_OFFSET, V_WALL_OFFSET,
)
from quoridor.env import QuoridorEnv
from quoridor.vec_env import VecQuoridorEnv


# ---------------------------------------------------------------------------
# Legal mask perspective flip (used by PPOBot)
# ---------------------------------------------------------------------------

# Maps flipped-space move action index → actual-space move action index.
# np.flipud negates dr but leaves dc unchanged, so:
#   up(0) ↔ down(1), left(2)/right(3) stay, NW(4)↔SW(6), NE(5)↔SE(7).
# Diagonal derivation: flip(NW: dr=-1,dc=-1) → (dr=+1,dc=-1) = SW = action 6.
_MOVE_FLIP = [1, 0, 2, 3, 6, 7, 4, 5]


def _flip_legal_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remap a legal mask from actual board coordinates to flipped (P1-perspective)
    coordinates so it is consistent with the flipped observation fed to the model.

    np.flipud flips rows but not columns, so:
      - Move up/down (0↔1) and diagonals (NW↔SE, NE↔SW) are swapped.
      - Left/right (2, 3) are unchanged.
      - Fence row r becomes FENCE_GRID-1-r in the flipped view.
      - Pass (index 136) is always False; no remapping needed.
    """
    flipped = np.zeros_like(mask)

    # --- move actions (0-7) ---
    for flipped_idx, actual_idx in enumerate(_MOVE_FLIP):
        flipped[flipped_idx] = mask[actual_idx]

    # --- fence actions (8-135): mirror row index ---
    for r in range(FENCE_GRID):
        actual_r = FENCE_GRID - 1 - r
        for c in range(FENCE_GRID):
            flipped[H_WALL_OFFSET + r * FENCE_GRID + c] = (
                mask[H_WALL_OFFSET + actual_r * FENCE_GRID + c]
            )
            flipped[V_WALL_OFFSET + r * FENCE_GRID + c] = (
                mask[V_WALL_OFFSET + actual_r * FENCE_GRID + c]
            )

    # pass action (136) is always False; initialised-to-zero covers it
    return flipped


def _model_forward(
    model: torch.nn.Module,
    spatial: torch.Tensor,
    scalars: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> tuple:
    """
    Uniform forward-pass wrapper that always returns (dist, value, aux_pred).

    BFS models (bfs, bfs_resnet) return 3 values natively; non-BFS models return 2.
    This wrapper pads with None so all call sites can unpack 3 values without
    knowing which model variant is running.

    temperature: scales logits before building the Categorical distribution.
    >1.0 flattens the distribution (more exploration), <1.0 sharpens it.
    Only meaningful during rollout collection — pass 1.0 during PPO updates.
    """
    result = model(spatial, scalars, mask)
    if len(result) == 3:
        dist, value, aux_pred = result
    else:
        dist, value = result
        aux_pred = None

    # Apply temperature scaling to the logits if T != 1.0.
    # This preserves action ranking but widens the probability spread,
    # encouraging the agent to try plausible alternative moves.
    if temperature != 1.0:
        from torch.distributions import Categorical
        scaled_logits = dist.logits / temperature
        dist = Categorical(logits=scaled_logits)

    return dist, value, aux_pred


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

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 temperature: float = 1.0) -> None:
        self.model  = model
        self.device = device
        self.temperature = temperature
        # Detect whether this model expects 6-channel (BFS) or 4-channel input.
        first_weight = next(iter(model.parameters()))
        self.use_bfs = (first_weight.shape[1] == NUM_CHANNELS_BFS)

    def reset(self) -> None:
        pass  # PPO model is stateless across episodes

    def choose_action(self, game) -> tuple:
        # The observation is always from the current player's perspective.
        # For P1 (turn==1), game.get_observation() flips the board vertically
        # so the model always sees its own pawn at the bottom and goal at top —
        # matching P0's training distribution.
        flip = (game.turn == 1)

        spatial, scalars = game.get_observation(use_bfs=self.use_bfs)

        # get_legal_mask() returns indices in ACTUAL board coordinates, but the
        # model expects a mask consistent with the FLIPPED observation. When
        # flip=True, remap the mask so move directions and fence rows match the
        # flipped view. Without this, the model can't select its intended actions.
        actual_mask = game.get_legal_mask()
        legal_mask = _flip_legal_mask(actual_mask) if flip else actual_mask

        s  = torch.tensor(spatial,    dtype=torch.float32).unsqueeze(0).to(self.device)
        sc = torch.tensor(scalars,    dtype=torch.float32).unsqueeze(0).to(self.device)
        m  = torch.tensor(legal_mask, dtype=torch.bool   ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist, _, _aux = _model_forward(self.model, s, sc, m,
                                           temperature=self.temperature)
            action_idx = int(dist.sample().item())

        # action is decoded in flipped (model) space; translate back to actual coords.
        action = index_to_action(action_idx)

        # index_to_action returns ("move", dr, dc) with direction deltas, but
        # _dispatch_bot_action expects ("move", dest_r, dest_c) with absolute
        # board coordinates. Resolve the direction to a valid destination.
        if action[0] == "move":
            _, dr, dc = action
            # np.flipud inverts rows but not columns: negate dr to translate
            # from flipped space to actual space. dc is unchanged.
            if flip:
                dr = -dr
            cur_r = int(game.pos[game.turn, 0])
            cur_c = int(game.pos[game.turn, 1])
            for dest_r, dest_c in game.get_valid_moves():
                if (np.sign(dest_r - cur_r) == np.sign(dr)
                        and np.sign(dest_c - cur_c) == np.sign(dc)):
                    return ("move", dest_r, dest_c)
            # Should never be reached: _flip_legal_mask guarantees the model
            # only selects actions that correspond to valid actual moves.
            raise ValueError(
                f"PPOBot: no valid destination for action {action_idx} "
                f"(flip={flip}, dr={dr}, dc={dc}, pos={game.pos[game.turn]})"
            )

        if action[0] == "fence":
            _, r, c, ori = action
            # Fence rows in flipped space correspond to actual row (FENCE_GRID-1-r).
            if flip:
                r = FENCE_GRID - 1 - r
            return ("fence", r, c, ori)

        return action  # pass action needs no coordinate translation


# ---------------------------------------------------------------------------
# Opponent pool (Fictitious Self-Play)
# ---------------------------------------------------------------------------

class OpponentPool:
    """
    Maintains a rolling pool of frozen past-self checkpoints for self-play training.

    At the start of each episode (when QuoridorEnv calls bot.reset()), the pool
    samples an opponent: HeuristicBot with probability heuristic_ratio, otherwise
    a uniformly random past checkpoint. This implements Neural Fictitious Self-Play
    (Heinrich & Silver, 2016) with a heuristic anchor.

    Why a heuristic anchor?
        Early in training all pool checkpoints are weak. Sampling only from weak
        opponents produces easy wins that teach nothing. The heuristic provides a
        stable non-trivial opponent until the pool fills with stronger checkpoints.
        Research basis: OpenAI Five's 80/20 latest-vs-historical split; here the
        heuristic fills the "historical" role until strong checkpoints accumulate.

    Why uniform sampling over the pool?
        Full PFSP (AlphaStar) weights checkpoints by difficulty — overkill at this
        scale. Uniform sampling still prevents cycling by exposing the agent to a
        diverse range of past strategies rather than only the latest.

    Usage:
        pool = OpponentPool(model_type="bfs_resnet", device=device)
        env.bot = pool                  # QuoridorEnv calls reset()/choose_action()
        ...
        pool.add(model)                 # call at each OPPONENT_UPDATE_FREQ boundary
    """

    def __init__(
        self,
        model_type:       str,
        device:           torch.device,
        heuristic_ratio:  float = OPPONENT_HEURISTIC_RATIO,
        max_size:         int   = OPPONENT_POOL_SIZE,
        opponent_temperature: float = 1.0,
    ) -> None:
        self.model_type      = model_type
        self.device          = device
        self.heuristic_ratio = heuristic_ratio
        self.max_size        = max_size
        self.opponent_temperature = opponent_temperature
        # Each entry is a frozen PPOBot wrapping a deepcopy of the model at that point.
        self._pool: list     = []
        self._heuristic      = HeuristicBot()
        self._current        = self._heuristic  # bot used for this episode

    def add(self, model: torch.nn.Module) -> None:
        """
        Snapshot the current model and add it to the pool.

        Deep-copies the state_dict so the pool entry is fully independent of
        future gradient updates. Drops the oldest entry when the pool is full.
        """
        frozen = copy.deepcopy(model)
        frozen.eval()
        self._pool.append(PPOBot(frozen, self.device,
                                 temperature=self.opponent_temperature))
        if len(self._pool) > self.max_size:
            self._pool.pop(0)  # evict oldest — FIFO

    def fork(self) -> "OpponentPool":
        """Create a lightweight copy that shares the checkpoint pool.

        Each fork has its own _current opponent, so multiple envs can call
        reset()/choose_action() without clobbering each other's state.
        Used by VecQuoridorEnv to give each sub-env an independent pool view.
        """
        child = OpponentPool.__new__(OpponentPool)
        child.model_type      = self.model_type
        child.device          = self.device
        child.heuristic_ratio = self.heuristic_ratio
        child.max_size        = self.max_size
        child.opponent_temperature = self.opponent_temperature
        child._pool           = self._pool      # shared list — add() on parent updates all forks
        child._heuristic      = HeuristicBot()  # each fork gets its own heuristic instance
        child._current        = child._heuristic
        return child

    def reset(self) -> None:
        """
        Sample an opponent for the coming episode.

        Called by QuoridorEnv.reset() at the start of every episode. Sampling
        here (not in choose_action) means the opponent is consistent for the
        whole episode, which is important for coherent game-play and stable
        gradient estimation.
        """
        if not self._pool or random.random() < self.heuristic_ratio:
            self._current = self._heuristic
        else:
            self._current = random.choice(self._pool)
        self._current.reset()

    def choose_action(self, game) -> tuple:
        """Delegate to whichever opponent was sampled at the start of this episode."""
        return self._current.choose_action(game)

    @property
    def pool_size(self) -> int:
        """Current number of checkpoint entries in the pool (excludes heuristic)."""
        return len(self._pool)


class MixedOpponentPool(OpponentPool):
    """Per-move mixed opponent: each action is independently sampled from
    either HeuristicBot (with probability heuristic_ratio) or a random pool
    checkpoint (with probability 1 - heuristic_ratio).

    Unlike OpponentPool which picks ONE opponent per game, this mixes styles
    within a single game. The opponent might play a heuristic move, then a
    self-play move, then another heuristic move. This creates diverse board
    states that neither pure opponent would produce alone, forcing the agent
    to handle unexpected mid-game transitions.
    """

    def reset(self) -> None:
        """Reset all pool bots. No per-episode opponent selection needed."""
        self._heuristic.reset()
        for bot in self._pool:
            bot.reset()

    def choose_action(self, game) -> tuple:
        """Per-move sampling: heuristic with probability heuristic_ratio,
        otherwise a random pool checkpoint."""
        if not self._pool or random.random() < self.heuristic_ratio:
            return self._heuristic.choose_action(game)
        return random.choice(self._pool).choose_action(game)

    def fork(self) -> "MixedOpponentPool":
        """Create a lightweight copy that shares the checkpoint pool."""
        child = MixedOpponentPool.__new__(MixedOpponentPool)
        child.model_type      = self.model_type
        child.device          = self.device
        child.heuristic_ratio = self.heuristic_ratio
        child.max_size        = self.max_size
        child.opponent_temperature = self.opponent_temperature
        child._pool           = self._pool      # shared list
        child._heuristic      = HeuristicBot()
        child._current        = child._heuristic
        return child


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
    use_reward_shaping:  bool  = False,
    n_envs:              int   = 1,
    reward_self_coef:    float = None,
    reward_opp_coef:     float = None,
    randomize_start:     bool  = False,
    repetition_penalty:  float = None,
) -> tuple:
    """
    Instantiate the correct model and vectorized environment pair.

    Keeping this in one place ensures the model channel count, env BFS flag,
    and reward shaping flag are always in sync.

    Args:
        model_type:         One of "baseline", "resnet", "bfs", "bfs_resnet".
        bot:                Opponent bot instance shared across all sub-envs.
        device:             Torch device; model is moved to device before returning.
        use_reward_shaping: If True, training envs use path-length delta shaping.
                            Eval env is never shaped so win rate is comparable.
        n_envs:             Number of parallel environments (default 1).
                            n_envs=1 is identical to the single-env code path.
        reward_self_coef:   Override for REWARD_SHAPING_SELF_COEF (None = use config default).
        reward_opp_coef:    Override for REWARD_SHAPING_OPP_COEF (None = use config default).

    Returns:
        (model, vec_train_env, eval_env)
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Choose from: {list(_MODEL_REGISTRY.keys())}"
        )
    ModelClass, use_bfs = _MODEL_REGISTRY[model_type]
    model    = ModelClass().to(device)
    env      = VecQuoridorEnv(n_envs=n_envs, bot=bot, use_bfs=use_bfs,
                              use_reward_shaping=use_reward_shaping,
                              reward_self_coef=reward_self_coef,
                              reward_opp_coef=reward_opp_coef,
                              randomize_start=randomize_start,
                              repetition_penalty=repetition_penalty)
    # Eval env: no shaping, no randomized start, no repetition penalty — clean measurement.
    eval_env = QuoridorEnv(bot=HeuristicBot(), use_bfs=use_bfs, use_reward_shaping=False,
                           repetition_penalty=0.0)
    return model, env, eval_env


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollout_vec(
    model:         torch.nn.Module,
    vec_env:       VecQuoridorEnv,
    steps_per_env: int,
    device:        torch.device,
    spatial:       np.ndarray,  # (N, C, 9, 9)
    scalars:       np.ndarray,  # (N, 2)
    temperature:   float = 1.0,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect steps_per_env steps from each of N parallel envs.

    Each loop iteration dispatches one action per env and records the
    transition. Done envs auto-reset inside vec_env.step(), so the returned
    next_spatial/scalars is always the first obs of the new episode for done
    envs — ready to feed back as the next loop iteration's input.

    Args:
        model:         PPOModel (set to eval during collection).
        vec_env:       VecQuoridorEnv with N sub-envs.
        steps_per_env: Steps to collect per env (= rollout_steps // n_envs).
        device:        Torch device for inference.
        spatial:       Current stacked spatial obs (N, C, 9, 9).
        scalars:       Current stacked scalar obs  (N, 2).

    Returns:
        rollout:      Dict of (T, N, ...) tensors where T = steps_per_env.
        next_spatial: (N, C, 9, 9) — observation after the final step.
        next_scalars: (N, 2)       — scalars after the final step.
        last_dones:   (N,) bool    — True if env i ended on a terminal step.
    """
    N = vec_env.n_envs

    spatials:    list = []
    scalars_lst: list = []
    legal_masks: list = []
    actions:     list = []
    log_probs:   list = []
    rewards:     list = []
    dones:       list = []
    values:      list = []

    model.eval()

    for _ in range(steps_per_env):
        legal_mask = vec_env.get_legal_mask()  # (N, 137)

        with torch.no_grad():
            spatial_t = torch.tensor(spatial,    dtype=torch.float32).to(device)  # (N, C, 9, 9)
            scalars_t = torch.tensor(scalars,    dtype=torch.float32).to(device)  # (N, 2)
            mask_t    = torch.tensor(legal_mask, dtype=torch.bool    ).to(device)  # (N, 137)
            dist, value, _aux = _model_forward(model, spatial_t, scalars_t, mask_t,
                                                   temperature=temperature)
            action   = dist.sample()                             # (N,)
            # Log-prob at T=1.0 so PPO ratios are unbiased by temperature.
            if temperature != 1.0:
                dist_base, _, _ = _model_forward(model, spatial_t, scalars_t, mask_t)
                log_prob = dist_base.log_prob(action)            # (N,)
            else:
                log_prob = dist.log_prob(action)                 # (N,)

        action_arr = action.cpu().numpy().astype(np.int32)       # (N,)

        next_spatial, next_scalars, reward_arr, done_arr, _infos = vec_env.step(action_arr)
        # next_spatial: (N, C, 9, 9) — first obs of new episode for done envs
        # reward_arr:   (N,) float32
        # done_arr:     (N,) bool

        spatials.append(spatial)             # (N, C, 9, 9)
        scalars_lst.append(scalars)          # (N, 2)
        legal_masks.append(legal_mask)       # (N, 137)
        actions.append(action_arr)           # (N,)
        log_probs.append(log_prob.cpu().numpy())               # (N,)
        rewards.append(reward_arr)           # (N,)
        dones.append(done_arr)               # (N,)
        values.append(value.squeeze(-1).cpu().numpy())         # (N,)

        spatial = next_spatial
        scalars = next_scalars

    # Stack into (T, N, ...) tensors
    rollout = {
        "spatial":    torch.tensor(np.stack(spatials),    dtype=torch.float32).to(device),  # (T, N, C, 9, 9)
        "scalars":    torch.tensor(np.stack(scalars_lst), dtype=torch.float32).to(device),  # (T, N, 2)
        "legal_mask": torch.tensor(np.stack(legal_masks), dtype=torch.bool   ).to(device),  # (T, N, 137)
        "actions":    torch.tensor(np.stack(actions),     dtype=torch.long   ).to(device),  # (T, N)
        "log_probs":  torch.tensor(np.stack(log_probs),   dtype=torch.float32).to(device),  # (T, N)
        "rewards":    torch.tensor(np.stack(rewards),     dtype=torch.float32).to(device),  # (T, N)
        "dones":      torch.tensor(np.stack(dones),       dtype=torch.bool   ).to(device),  # (T, N)
        "values":     torch.tensor(np.stack(values),      dtype=torch.float32).to(device),  # (T, N)
    }

    last_dones = dones[-1]  # (N,) bool ndarray — one per env
    return rollout, spatial, scalars, last_dones


# ---------------------------------------------------------------------------
# Generalized Advantage Estimation
# ---------------------------------------------------------------------------

def compute_gae_vec(
    rewards:     torch.Tensor,  # (T, N)
    values:      torch.Tensor,  # (T, N)
    dones:       torch.Tensor,  # (T, N) bool
    last_values: torch.Tensor,  # (N,)   — V(s_{T+1}); 0 where last step was terminal
    gamma:       float,
    lam:         float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GAE over N parallel environments.

    Each column (env) is processed independently. Episode boundaries zero the
    bootstrap value and carry-over advantage, so credit never bleeds between
    episodes within the same column — identical semantics to compute_gae but
    applied in parallel across N envs.

    Returns:
        advantages : (T, N)
        returns    : (T, N) — advantages + values; value-head targets
    """
    T, N     = rewards.shape
    advantages = torch.zeros(T, N, device=rewards.device)
    gae        = torch.zeros(N,    device=rewards.device)  # (N,) running advantage

    for t in reversed(range(T)):
        not_done = (~dones[t]).float()                          # (N,)
        next_val = values[t + 1] if t < T - 1 else last_values  # (N,)
        delta    = rewards[t] + gamma * next_val * not_done - values[t]  # (N,)
        gae      = delta + gamma * lam * not_done * gae         # (N,)
        advantages[t] = gae

    returns = advantages + values  # (T, N)
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
    aux_loss_coef:   float,
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
        "move_prob_mass", "wall_prob_mass", "move_entropy", "wall_entropy", "aux_loss",
        "entropy_loss",  # actual entropy loss magnitude — compare against policy_loss to detect dominance
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
            dist, value, aux_pred = _model_forward(model, mb_spatial, mb_scalars, mb_mask)
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

            # Per-action-type entropy bonus.
            # 128 wall actions vs 8 move actions means a single coef pushes
            # probability mass toward walls. Weight them separately.
            probs_ent = dist.probs                                    # (B, 137)
            move_p_ent = probs_ent[:, :8]                             # (B, 8)
            wall_p_ent = probs_ent[:, 8:136]                          # (B, 128)
            move_ent = -(move_p_ent * (move_p_ent + 1e-8).log()).sum(dim=1)  # (B,)
            wall_ent = -(wall_p_ent * (wall_p_ent + 1e-8).log()).sum(dim=1)  # (B,)
            entropy_loss = -entropy_coef * (
                PPO_MOVE_ENTROPY_COEF * move_ent.mean() +
                PPO_WALL_ENTROPY_COEF * wall_ent.mean()
            )

            # Auxiliary path-length regression loss (BFS models only).
            # Targets extracted from rollout observations at no extra env cost:
            #   ch0 = current-player pawn (one-hot), ch4 = current-player BFS map
            #   ch1 = opponent pawn (one-hot),       ch5 = opponent BFS map
            # dot(ch4, ch0) gives the normalised BFS distance at the pawn's cell.
            if aux_pred is not None:
                my_dist_target  = (mb_spatial[:, 4] * mb_spatial[:, 0]).sum(dim=(1, 2))  # (B,)
                opp_dist_target = (mb_spatial[:, 5] * mb_spatial[:, 1]).sum(dim=(1, 2))  # (B,)
                aux_targets = torch.stack([my_dist_target, opp_dist_target], dim=1)       # (B, 2)
                aux_loss    = F.mse_loss(aux_pred, aux_targets)
            else:
                aux_loss = torch.zeros(1, device=device)

            loss = policy_loss + value_coef * value_loss + entropy_loss + aux_loss_coef * aux_loss

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
                metrics["aux_loss"]       += aux_loss.item()
                metrics["entropy_loss"]   += entropy_loss.item()
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

                dist, _, _aux = _model_forward(model, spatial_t, scalars_t, mask_t)
                action  = int(dist.probs.argmax(dim=1).item())  # greedy

                spatial, scalars, reward, done, _ = eval_env.step(action)
                ep_length += 1

            if reward > 0:
                wins += 1
            total_length += ep_length

    model.train()
    return wins / n_episodes, total_length / n_episodes


# ---------------------------------------------------------------------------
# Checkpoint helpers (new format: full training state)
# ---------------------------------------------------------------------------

def save_checkpoint(
    path:             str,
    model:            torch.nn.Module,
    optimizer:        torch.optim.Optimizer,
    total_steps:      int,
    max_steps:        int,
    rolling_win_rate: float,
) -> None:
    """
    Save full training state so a run can be resumed exactly.

    New format stores a dict with 'model' and 'optimizer' keys so it can be
    distinguished from old-format checkpoints (bare state_dict tensors).
    """
    torch.save({
        "model":            model.state_dict(),
        "optimizer":        optimizer.state_dict(),
        "total_steps":      total_steps,
        "max_steps":        max_steps,
        "rolling_win_rate": rolling_win_rate,
    }, path)


def load_checkpoint(path: str, device: torch.device) -> dict:
    """
    Load a checkpoint and return a normalised dict regardless of format.

    New format: {"model": state_dict, "optimizer": ..., "total_steps": ..., ...}
    Old format: bare state_dict (dict of str → Tensor, no 'optimizer' key)

    Returns:
        is_new_format : True if optimizer state is available.
        model         : model state_dict
        optimizer     : optimizer state_dict (None for old format)
        total_steps   : step count at save time (0 for old format)
        max_steps     : max_steps at save time (None for old format)
        rolling_win_rate : epsilon curriculum win rate (0.0 for old format)
    """
    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        return {
            "model":            raw["model"],
            "optimizer":        raw.get("optimizer"),
            "total_steps":      raw.get("total_steps", 0),
            "max_steps":        raw.get("max_steps"),
            "rolling_win_rate": raw.get("rolling_win_rate", 0.0),
            "is_new_format":    True,
        }
    else:
        # Old format: bare state_dict — no training metadata available.
        return {
            "model":            raw,
            "optimizer":        None,
            "total_steps":      0,
            "max_steps":        None,
            "rolling_win_rate": 0.0,
            "is_new_format":    False,
        }


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
    #
    # Epsilon curriculum: swap HeuristicBot for EpsilonHeuristicBot when requested.
    # We keep a direct reference (epsilon_bot) so the training loop can update
    # epsilon in-place without rebuilding the env. When the pool opponent is used,
    # we also replace its internal heuristic with the epsilon-wrapped version so
    # the heuristic fraction of pool games is also weakened.
    epsilon_bot: EpsilonHeuristicBot | None = None
    if args.epsilon_curriculum:
        epsilon_bot = EpsilonHeuristicBot(epsilon=EPSILON_CURRICULUM_START)
        bot = epsilon_bot
    else:
        bot = HeuristicBot()
    model, env, eval_env_obj = build_model_and_env(
        args.model, bot, device, args.reward_shaping, args.num_envs,
        reward_self_coef=args.reward_self_coef,
        reward_opp_coef=args.reward_opp_coef,
        randomize_start=args.randomize_start,
        repetition_penalty=args.repetition_penalty,
    )
    steps_per_env = args.rollout_steps // args.num_envs
    print(f"Model: {args.model}  |  BFS env: {_MODEL_REGISTRY[args.model][1]}  |  "
          f"Opponent: {args.opponent}  |  Reward shaping: {args.reward_shaping}  |  "
          f"Envs: {args.num_envs}  |  Steps/env: {steps_per_env}")

    # Opponent setup — three modes:
    #   heuristic : always train vs HeuristicBot (stable, no non-stationarity).
    #   self_play : frozen copy of self, updated every OPPONENT_UPDATE_FREQ steps.
    #   pool      : mix of HeuristicBot + rolling window of past checkpoints (FSP).
    #               Heuristic anchors training while pool fills with real checkpoints.
    frozen_model = None
    pool         = None
    if args.opponent == "self_play":
        frozen_model = copy.deepcopy(model)
        frozen_model.eval()
        env.bot = PPOBot(frozen_model, device)
        print(f"Self-play: frozen opponent initialized. "
              f"Updates every {OPPONENT_UPDATE_FREQ} steps.")
    elif args.opponent == "pool":
        pool = OpponentPool(
            model_type           = args.model,
            device               = device,
            heuristic_ratio      = OPPONENT_HEURISTIC_RATIO,
            max_size             = OPPONENT_POOL_SIZE,
            opponent_temperature = args.temperature,
        )
        # If epsilon curriculum is active, replace the pool's internal heuristic
        # with the epsilon-wrapped version so the heuristic fraction of pool
        # episodes is also weakened during early training.
        if epsilon_bot is not None:
            pool._heuristic = epsilon_bot
        env.bot = pool
        print(f"Opponent pool: heuristic_ratio={OPPONENT_HEURISTIC_RATIO}, "
              f"max_size={OPPONENT_POOL_SIZE}, "
              f"sync_every={OPPONENT_UPDATE_FREQ} steps.")
    elif args.opponent == "mixed":
        pool = MixedOpponentPool(
            model_type           = args.model,
            device               = device,
            heuristic_ratio      = OPPONENT_HEURISTIC_RATIO,
            max_size             = OPPONENT_POOL_SIZE,
            opponent_temperature = args.temperature,
        )
        if epsilon_bot is not None:
            pool._heuristic = epsilon_bot
        env.bot = pool
        print(f"Mixed opponent: per-move sampling, heuristic_ratio={OPPONENT_HEURISTIC_RATIO}, "
              f"max_size={OPPONENT_POOL_SIZE}, "
              f"sync_every={OPPONENT_UPDATE_FREQ} steps.")

    # Optional BC warm-start: load weights from a pretrained checkpoint.
    #
    # Two checkpoint types are supported; the exclusion filter handles both:
    #
    #   PPO BC checkpoint (proj.*, res_blocks.*, actor_head.*, value_head.*, aux_head.*):
    #     → Load backbone + actor_head (skip value_head, aux_head).
    #     → actor_head was trained by BC to predict expert actions — preserving it
    #       is the whole point of BC pretraining. Discarding it (previous bug) meant
    #       the policy started randomly initialized despite BC training.
    #     → value_head and aux_head are skipped: they were frozen during BC (no
    #       supervised signal) so their weights are random anyway, and value must
    #       be trained from scratch on RL returns.
    #
    #   DQN checkpoint (conv.*, fc.*):
    #     → fc.* excluded; conv.* backbone loads into baseline/resnet models.
    #     → actor_head not present in DQN checkpoint — strict=False handles this
    #       as missing (randomly initialized).
    #
    # strict=False handles any remaining shape or name mismatches gracefully.
    _HEAD_PREFIXES = ("value_head.", "aux_head.", "fc.")
    if args.checkpoint is not None:
        bc_raw     = torch.load(args.checkpoint, map_location=device)
        # Support both full training checkpoints (dict with "model" key) and
        # raw state_dicts (from torch.save(model.state_dict(), path)).
        if isinstance(bc_raw, dict) and "model" in bc_raw:
            bc_state = bc_raw["model"]
        elif isinstance(bc_raw, dict) and "model_state_dict" in bc_raw:
            bc_state = bc_raw["model_state_dict"]
        else:
            bc_state = bc_raw
        to_load    = {k: v for k, v in bc_state.items()
                      if not any(k.startswith(p) for p in _HEAD_PREFIXES)}
        missing, unexpected = model.load_state_dict(to_load, strict=False)
        loaded = len(to_load) - len(unexpected)
        has_actor = any(k.startswith("actor_head.") for k in to_load)
        print(f"Loaded checkpoint from {args.checkpoint}.")
        print(f"  Tensors in checkpoint (excl. value/aux heads) : {len(to_load)}")
        print(f"  Successfully loaded                           : {loaded}")
        print(f"  Includes trained actor_head                   : {has_actor}")
        print(f"  Shape/name mismatch (skipped)                 : {len(unexpected)}")
        print(f"  Randomly initialized                          : {len(missing)}")
        if loaded == 0:
            print("  WARNING: No weights loaded. Checkpoint architecture "
                  "may not match this model — check key names and channel counts. "
                  "Run pretrain_bc.py --model bfs_resnet for a compatible checkpoint.")
        if not has_actor:
            print("  NOTE: No actor_head in checkpoint (DQN or backbone-only checkpoint). "
                  "Policy head starts randomly initialized.")

    # --- Value head burn-in (optional) ---
    # When loading a BC checkpoint, the actor has learned weights (~20% win rate) but
    # the value head is randomly initialized. Random value predictions → garbage GAE
    # advantages → policy gradient can destroy BC knowledge in the first few updates.
    #
    # Fix: freeze actor_head.* and train only the backbone + value head for
    # value_burnin_steps steps. The frozen BC actor generates ~20% win-rate
    # trajectories — giving the value head real signal to fit. After burn-in,
    # we unfreeze the actor and run PPO normally.
    #
    # Why freeze only actor_head, not the backbone?
    #   The backbone encodes spatial features needed for BOTH policy and value.
    #   Keeping it unfrozen lets the backbone adapt its representations to the
    #   value task, which gives the value head better inputs from the start.
    actor_frozen = False
    if args.value_burnin_steps > 0:
        if args.checkpoint is None:
            print("[Burn-in] WARNING: --value-burnin-steps set but no --checkpoint provided. "
                  "Burn-in skipped (actor head not frozen).")
        else:
            for name, param in model.named_parameters():
                if name.startswith("actor_head."):
                    param.requires_grad = False
            actor_frozen = True
            frozen_count = sum(
                1 for n, _ in model.named_parameters() if n.startswith("actor_head.")
            )
            print(f"[Burn-in] Actor head frozen ({frozen_count} param tensors). "
                  f"Training backbone + value head only for {args.value_burnin_steps} steps.")

    # --- Optimizer ---
    # Single Adam over all parameters (backbone + both heads) — standard PPO practice.
    # Adam allocates optimizer state lazily on first backward, so frozen actor_head
    # params (requires_grad=False) get no state until they're unfrozen after burn-in.
    optimizer = torch.optim.Adam(model.parameters(), lr=PPO_LR)

    os.makedirs("checkpoints", exist_ok=True)

    # --- Training state ---
    total_steps      = 0
    start_step       = 0   # step at which this run began (>0 when resuming)
    best_win_rate    = 0.0
    graduated        = False
    rolling_win_rate = 0.0  # epsilon curriculum rolling win rate; updated at each eval

    # Rolling window of training episode outcomes (1=win, 0=loss/ongoing) used
    # to drive epsilon curriculum annealing. Deliberately separate from eval win
    # rate so the annealing signal reflects actual training experience, not the
    # eval snapshot against full-strength HeuristicBot.
    # 200 episodes ≈ 6k steps at ~30 plies/game — large enough to smooth noise,
    # small enough to respond to genuine improvement within a few thousand steps.
    training_outcomes: collections.deque = collections.deque(maxlen=200)

    # --- Resume from checkpoint (full training state restore) ---
    # --resume is for continuing a training run; --checkpoint is for BC warm-starting
    # a fresh run. They are mutually exclusive.
    if args.resume is not None:
        if args.checkpoint is not None:
            raise ValueError(
                "--resume and --checkpoint cannot be combined. "
                "--resume continues a prior training run (restores optimizer state, step "
                "count, and epsilon curriculum). --checkpoint BC-warm-starts a fresh run."
            )
        if args.value_burnin_steps > 0:
            raise ValueError(
                "--resume and --value-burnin-steps cannot be combined. "
                "Burn-in is only meaningful on a fresh BC-warm-started run."
            )

        ckpt = load_checkpoint(args.resume, device)
        model.load_state_dict(ckpt["model"])

        if ckpt["optimizer"] is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
            print(f"[Resume] Optimizer state restored — Adam m/v vectors preserved.")
        else:
            print(
                f"[Resume] WARNING: Old-format checkpoint — optimizer state unavailable. "
                f"Adam starts fresh; effective LR will be elevated for ~1k-5k gradient "
                f"steps until the moment estimates warm up. Unavoidable for this checkpoint."
            )

        # Restore training position.
        restored_steps   = ckpt["total_steps"]
        total_steps      = restored_steps
        start_step       = restored_steps
        rolling_win_rate = ckpt["rolling_win_rate"]

        # Pre-fill training_outcomes so the epsilon curriculum doesn't reset to
        # maximum weakening (ε=0.3) after a resume.  200 synthetic entries
        # (full deque) give a stable rolling estimate immediately.
        if args.epsilon_curriculum and rolling_win_rate > 0.0:
            n_wins = int(round(rolling_win_rate * training_outcomes.maxlen))
            n_loss = training_outcomes.maxlen - n_wins
            training_outcomes.extend([1] * n_wins + [0] * n_loss)

        # Entropy schedule uses total_steps / max_steps, so restoring total_steps
        # while setting max_steps = restored + additional keeps the schedule
        # fraction correct (it continues from where it left off, not from 0).
        args.max_steps = restored_steps + args.additional_steps

        print(
            f"[Resume] Restored from step {restored_steps:,}. "
            f"Training until step {args.max_steps:,} (+{args.additional_steps:,})."
        )
        if rolling_win_rate > 0.0:
            print(f"[Resume] Rolling win rate restored: {rolling_win_rate:.2%}")

    # --- W&B ---
    if use_wandb:
        import wandb
        wandb.init(
            project="quoridor-ppo",
            name=args.run_name,   # None → wandb auto-names; set via --run-name for batch runs
            config={
                "model":                    args.model,
                "opponent":                 args.opponent,
                "reward_shaping":           args.reward_shaping,
                "epsilon_curriculum":       args.epsilon_curriculum,
                "epsilon_curriculum_start": EPSILON_CURRICULUM_START     if args.epsilon_curriculum else None,
                "epsilon_curriculum_thr":   EPSILON_CURRICULUM_THRESHOLD if args.epsilon_curriculum else None,
                "opponent_pool_size":       OPPONENT_POOL_SIZE   if args.opponent in ("pool", "mixed") else None,
                "heuristic_ratio":          OPPONENT_HEURISTIC_RATIO if args.opponent in ("pool", "mixed") else None,
                "opponent_update_freq":     OPPONENT_UPDATE_FREQ,
                "device":             str(device),
                "checkpoint":           args.checkpoint,
                "value_burnin_steps":  args.value_burnin_steps,
                "num_envs":            args.num_envs,
                "rollout_steps":       args.rollout_steps,
                "steps_per_env":       steps_per_env,
                "ppo_epochs":         PPO_EPOCHS,
                "mini_batch_size":    PPO_MINI_BATCH_SIZE,
                "clip_eps":              PPO_CLIP_EPS,
                "gae_lambda":            args.lam,
                "gamma":                 args.gamma,
                "entropy_coef_start":    args.entropy_start if args.entropy_start is not None else PPO_ENTROPY_COEF_START,
                "entropy_coef_end":      args.entropy_end if args.entropy_end is not None else PPO_ENTROPY_COEF_END,
                "value_coef":            PPO_VALUE_COEF,
                "aux_loss_coef":         AUX_LOSS_COEF,
                "lr":                    PPO_LR,
                "grad_clip":             args.grad_clip,
                "run_name":              args.run_name,
                "max_steps":          args.max_steps,
                "eval_freq":          EVAL_FREQ,
                "eval_episodes":      EVAL_EPISODES,
                "seed":               SEED,
                "resume_from":        args.resume,
                "additional_steps":   args.additional_steps if args.resume else None,
                "resumed_from_step":  total_steps if args.resume else None,
            },
        )

    # Carry the current observation across rollout calls
    spatial, scalars = env.reset()

    print("Starting PPO training...")
    print(f"Each rollout: {args.rollout_steps} total steps ({args.num_envs} envs × "
          f"{steps_per_env} steps/env) → {PPO_EPOCHS} epochs × "
          f"{args.rollout_steps // PPO_MINI_BATCH_SIZE} mini-batches")

    try:
        while total_steps < args.max_steps and not graduated:

            # Unfreeze actor once burn-in is complete.
            # Checked at the TOP of the loop so the first post-burnin rollout
            # already uses the unfrozen actor (exploration recovers immediately).
            if actor_frozen and total_steps >= args.value_burnin_steps:
                for name, param in model.named_parameters():
                    if name.startswith("actor_head."):
                        param.requires_grad = True
                actor_frozen = False
                print(f"[Burn-in] Completed {args.value_burnin_steps} steps. "
                      "Actor head unfrozen — full PPO begins.")

            # 1. Collect rollout — (T, N, ...) tensors, T = steps_per_env
            rollout, spatial, scalars, last_dones = collect_rollout_vec(
                model, env, steps_per_env, device, spatial, scalars,
                temperature=args.temperature,
            )
            # rollout["rewards"]: (T, N), rollout["dones"]: (T, N), etc.

            # Extract training episode outcomes for epsilon curriculum annealing.
            # Terminal win: done=True and reward=+1.0 (pure terminal, not shaped).
            # Terminal loss: done=True and reward=-1.0.
            # Scan all (T, N) done flags — each done cell is one episode boundary.
            if args.epsilon_curriculum:
                T_steps = rollout["dones"].shape[0]
                for t in range(T_steps):
                    for i in range(args.num_envs):
                        if rollout["dones"][t, i]:
                            training_outcomes.append(
                                1 if rollout["rewards"][t, i].item() > 0 else 0
                            )

            # 2. Bootstrap values for the step after the rollout ends, per env.
            #    Zero out entries where the last step was terminal (true bootstrap = 0).
            last_values = torch.zeros(args.num_envs, device=device)  # (N,)
            non_terminal_envs = [i for i, d in enumerate(last_dones) if not d]
            if non_terminal_envs:
                with torch.no_grad():
                    model.eval()
                    # Batch bootstrap query for all non-terminal envs at once.
                    s_t  = torch.tensor(spatial[non_terminal_envs],  dtype=torch.float32).to(device)  # (K, C, 9, 9)
                    sc_t = torch.tensor(scalars[non_terminal_envs],  dtype=torch.float32).to(device)  # (K, 2)
                    bootstrap_vals = model.get_value(s_t, sc_t).squeeze(-1)   # (K,)
                    for k, i in enumerate(non_terminal_envs):
                        last_values[i] = bootstrap_vals[k]

            # 3. Vectorized GAE — (T, N) → (T, N) advantages and returns.
            #    Uses args.gamma/lam so CLI overrides work for experiments.
            advantages, returns = compute_gae_vec(
                rollout["rewards"], rollout["values"], rollout["dones"],
                last_values, args.gamma, args.lam,
            )

            # 4. Flatten (T, N, ...) → (T*N, ...) so ppo_update sees a flat batch.
            #    ppo_update is unchanged — it always operates on flat (B, ...) tensors.
            T_steps, N_envs = rollout["rewards"].shape
            B = T_steps * N_envs
            flat_rollout = {
                "spatial":    rollout["spatial"].view(B, *rollout["spatial"].shape[2:]),    # (B, C, 9, 9)
                "scalars":    rollout["scalars"].view(B, *rollout["scalars"].shape[2:]),    # (B, 2)
                "legal_mask": rollout["legal_mask"].view(B, *rollout["legal_mask"].shape[2:]),  # (B, 137)
                "actions":    rollout["actions"].view(B),    # (B,)
                "log_probs":  rollout["log_probs"].view(B),  # (B,)
                "rewards":    rollout["rewards"].view(B),    # (B,)
                "dones":      rollout["dones"].view(B),      # (B,)
                "values":     rollout["values"].view(B),     # (B,)
            }
            flat_advantages = advantages.view(B)  # (B,)
            flat_returns    = returns.view(B)      # (B,)

            # Normalise advantages per-rollout (not per-minibatch).
            # With sparse ±1 rewards, many minibatches have near-constant advantages.
            # Per-minibatch normalisation would amplify that near-zero signal to
            # unit variance, creating artificial gradient signal from noise.
            # Per-rollout normalisation only amplifies real variance across the
            # full rollout_steps-step collection window.
            flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

            # 5. Entropy coefficient — linearly decayed over training.
            # High early to prevent wall-action collapse; low late to allow
            # the policy to specialise once it has learned which actions matter.
            # Uses progress since start_step so resumed runs get the full
            # entropy schedule over their additional_steps budget.
            steps_since_start = total_steps - start_step
            total_budget = args.max_steps - start_step
            ent_start = args.entropy_start if args.entropy_start is not None else PPO_ENTROPY_COEF_START
            ent_end   = args.entropy_end   if args.entropy_end   is not None else PPO_ENTROPY_COEF_END
            entropy_coef = ent_start + (
                ent_end - ent_start
            ) * min(1.0, steps_since_start / total_budget)

            # 6. PPO update — uses args.grad_clip so CLI override works.
            metrics = ppo_update(
                model, optimizer, flat_rollout, flat_advantages, flat_returns,
                PPO_EPOCHS, PPO_MINI_BATCH_SIZE,
                PPO_CLIP_EPS, PPO_VALUE_COEF, entropy_coef,
                AUX_LOSS_COEF, args.grad_clip, device,
            )

            prev_steps   = total_steps
            total_steps += args.rollout_steps  # total env steps = T * N

            # Opponent sync: fires once per OPPONENT_UPDATE_FREQ boundary crossing.
            #   self_play — update the single frozen model in-place.
            #   pool      — snapshot the current model and add it to the pool.
            #               The pool's heuristic anchor remains unchanged.
            if (args.opponent in ("self_play", "pool", "mixed")
                    and total_steps // OPPONENT_UPDATE_FREQ > prev_steps // OPPONENT_UPDATE_FREQ):
                if args.opponent == "self_play":
                    frozen_model.load_state_dict(copy.deepcopy(model.state_dict()))
                    frozen_model.eval()
                    env.bot = PPOBot(frozen_model, device)
                    print(f"[self_play] Opponent synced at step {total_steps}")
                else:
                    pool.add(model)
                    label = "mixed" if args.opponent == "mixed" else "pool"
                    print(f"[{label}] Checkpoint added at step {total_steps} "
                          f"(pool size: {pool.pool_size}/{OPPONENT_POOL_SIZE})")
                if use_wandb:
                    import wandb
                    log = {"train/self_play_sync": total_steps}
                    if args.opponent in ("pool", "mixed"):
                        log["train/pool_size"] = pool.pool_size
                    wandb.log(log, step=total_steps)

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
                    # Auxiliary path-length regression loss (0.0 for non-BFS models).
                    # Should decrease quickly — if it stays high the backbone isn't
                    # learning path distances, which means wall signals won't emerge.
                    "train/aux_loss":       metrics["aux_loss"],
                    # Entropy loss magnitude — compare directly against policy_loss.
                    # If |entropy_loss| >> |policy_loss|, entropy dominates and the
                    # agent is too random. Healthy: within 2–3x of policy_loss magnitude.
                    "train/entropy_loss":   metrics["entropy_loss"],
                }, step=total_steps)

            # During burn-in, also log under a separate key so value head convergence
            # is visible as a distinct phase in W&B charts.
            if actor_frozen and use_wandb:
                import wandb
                wandb.log({"burnin/value_loss": metrics["value_loss"]}, step=total_steps)

            # 5. Periodic evaluation — trigger when we cross an EVAL_FREQ boundary.
            # total_steps advances in ROLLOUT_STEPS increments, so modulo-based
            # checks would only fire at LCM(ROLLOUT_STEPS, EVAL_FREQ) intervals.
            if total_steps // EVAL_FREQ > prev_steps // EVAL_FREQ:
                win_rate, mean_length = evaluate(model, EVAL_EPISODES, device, eval_env_obj)

                # Epsilon curriculum: anneal opponent difficulty based on training
                # rolling win rate (not eval win rate, which uses full-strength
                # HeuristicBot and is kept as a clean comparison metric).
                # Formula: epsilon = max(0, START * (1 - rolling / THRESHOLD))
                # At rolling=0.0 → epsilon=START; at rolling=THRESHOLD → epsilon=0.
                # rolling_win_rate is declared in outer scope so it's available
                # to save_checkpoint() calls at the end of training.
                opponent_epsilon = 0.0
                if args.epsilon_curriculum and epsilon_bot is not None:
                    rolling_win_rate = (
                        sum(training_outcomes) / len(training_outcomes)
                        if training_outcomes else 0.0
                    )
                    opponent_epsilon = max(
                        0.0,
                        EPSILON_CURRICULUM_START
                        * (1.0 - rolling_win_rate / EPSILON_CURRICULUM_THRESHOLD),
                    )
                    epsilon_bot.epsilon = opponent_epsilon  # in-place update

                print(
                    f"Step {total_steps:>7} | "
                    f"win_rate={win_rate:.2%} | "
                    f"entropy={metrics['entropy']:.3f} | "
                    f"kl={metrics['approx_kl']:.4f} | "
                    f"mean_ep_len={mean_length:.1f}"
                    + (f" | opp_ε={opponent_epsilon:.3f} (rolling={rolling_win_rate:.2%})"
                       if args.epsilon_curriculum else "")
                )

                if use_wandb:
                    import wandb
                    eval_log = {
                        "eval/win_rate":            win_rate,
                        "eval/mean_episode_length": mean_length,
                    }
                    if args.epsilon_curriculum:
                        eval_log["train/opponent_epsilon"]  = opponent_epsilon
                        eval_log["train/rolling_win_rate"]  = rolling_win_rate
                    wandb.log(eval_log, step=total_steps)

                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_name = f"checkpoints/best_{args.run_name}.pt" if args.run_name else "checkpoints/best_ppo.pt"
                    save_checkpoint(
                        best_name,
                        model, optimizer, total_steps, args.max_steps, rolling_win_rate,
                    )
                    print(f"  ✓ New best win rate ({win_rate:.2%}) — checkpoint saved.")

                if win_rate >= WIN_RATE_TARGET and not args.no_graduate:
                    print(f"\nGraduated at step {total_steps}! Win rate: {win_rate:.2%}")
                    graduated = True

        # Normal completion
        run_tag = f"_{args.run_name}" if args.run_name else ""
        final_path = f"checkpoints/ppo_final{run_tag}_{total_steps}.pt"
        save_checkpoint(
            final_path,
            model, optimizer, total_steps, args.max_steps, rolling_win_rate,
        )
        print(f"\nTraining complete. Final checkpoint: {final_path}")
        print(f"Best win rate: {best_win_rate:.2%}")

    finally:
        # Always saves on normal exit, Ctrl+C, or crash.
        run_tag = f"_{args.run_name}" if args.run_name else ""
        emergency_path = f"checkpoints/ppo_emergency{run_tag}_{total_steps}.pt"
        save_checkpoint(
            emergency_path,
            model, optimizer, total_steps, args.max_steps, rolling_win_rate,
        )
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
        "--epsilon-curriculum",
        action="store_true",
        dest="epsilon_curriculum",
        help=(
            "Wrap the training opponent in EpsilonHeuristicBot: with probability ε "
            "it takes a random legal action instead of the heuristic move. "
            f"ε starts at {EPSILON_CURRICULUM_START} and is annealed toward 0 as "
            "the agent's rolling training win rate approaches "
            f"{EPSILON_CURRICULUM_THRESHOLD}. "
            "Eval always uses full-strength HeuristicBot so win rate is comparable. "
            "Default: off (existing experiments unaffected)."
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
        "--value-burnin-steps",
        type=int,
        default=0,
        dest="value_burnin_steps",
        help=(
            "Freeze the actor head and train only the backbone + value head for "
            "this many env steps before starting full PPO updates. "
            "Only meaningful with --checkpoint: the BC-pretrained actor generates "
            "~20%% win-rate trajectories, giving the randomly-initialized value head "
            "real signal before policy gradients can corrupt the actor. "
            "Suggested value: 10000 (≈333 games at ~30 steps/game). Default: 0 (off)."
        ),
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        dest="num_envs",
        help=(
            "Number of parallel environments to collect rollouts from. "
            "ROLLOUT_STEPS is split as steps_per_env = rollout_steps // num_envs, "
            "so the total steps per update stays constant. "
            "num_envs=1 (default) is identical to the old single-env code path. "
            "Higher values reduce Python loop overhead at the cost of more RAM. "
            "NOTE: --opponent=pool is not compatible with num_envs > 1 "
            "(OpponentPool._current state is shared across envs)."
        ),
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=ROLLOUT_STEPS,
        dest="rollout_steps",
        help=(
            f"Total env steps to collect before each PPO update (default: {ROLLOUT_STEPS}). "
            "Divided evenly across --num-envs; each env runs rollout_steps // num_envs steps. "
            "Must be divisible by --num-envs and by PPO_MINI_BATCH_SIZE."
        ),
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint saved by this script (new format) to resume "
            "training from. Restores model weights, optimizer state (Adam m/v "
            "vectors), step count, and epsilon curriculum progress. "
            "Cannot be combined with --checkpoint (BC warm-start) or "
            "--value-burnin-steps."
        ),
    )
    parser.add_argument(
        "--additional-steps",
        type=int,
        default=PPO_MAX_STEPS,
        dest="additional_steps",
        help=(
            f"When using --resume, the number of additional steps to train beyond "
            f"the restored step count. "
            f"args.max_steps is set to restored_steps + additional_steps so the "
            f"entropy schedule continues smoothly. "
            f"Default: {PPO_MAX_STEPS} (run a full extra budget)."
        ),
    )
    parser.add_argument(
        "--reward-self-coef",
        type=float,
        default=None,
        dest="reward_self_coef",
        help=f"Reward for agent moving closer to goal (default: {REWARD_SHAPING_SELF_COEF} from config). "
             "Higher values teach 'advance toward goal' more strongly.",
    )
    parser.add_argument(
        "--reward-opp-coef",
        type=float,
        default=None,
        dest="reward_opp_coef",
        help=f"Reward for lengthening opponent's path via walls (default: {REWARD_SHAPING_OPP_COEF} from config). "
             "Higher values teach 'block opponent with walls' more strongly.",
    )
    parser.add_argument(
        "--entropy-start",
        type=float,
        default=None,
        dest="entropy_start",
        help=f"Override PPO_ENTROPY_COEF_START (default: {PPO_ENTROPY_COEF_START} from config). "
             "Higher = more forced exploration early in training.",
    )
    parser.add_argument(
        "--entropy-end",
        type=float,
        default=None,
        dest="entropy_end",
        help=f"Override PPO_ENTROPY_COEF_END (default: {PPO_ENTROPY_COEF_END} from config). "
             "Higher = policy stays stochastic even late in training. "
             "Set equal to --entropy-start for constant (no decay).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help=(
            "Logit temperature for action sampling during rollout collection. "
            ">1.0 flattens the policy distribution (more exploration of plausible "
            "alternative moves), <1.0 sharpens it. Only affects rollout sampling — "
            "the PPO update always uses T=1.0 so log-probs remain unbiased. "
            "Try 1.5 for moderate exploration, 2.0 for aggressive. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--no-graduate",
        action="store_true",
        dest="no_graduate",
        help="Disable early stopping when win rate reaches the graduation threshold. "
             "Training runs for the full step budget regardless of performance.",
    )
    parser.add_argument(
        "--randomize-start",
        action="store_true",
        dest="randomize_start",
        help="50%% of episodes the bot moves first, so the agent practises "
             "responding to openings (effectively playing as P1 half the time).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        dest="repetition_penalty",
        help="Penalty per revisit of a pawn position within an episode. "
             "Scales with visit count: penalty * (visits-1). Default uses "
             "config.REPETITION_PENALTY. Pass 0 to disable.",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="heuristic",
        choices=["heuristic", "self_play", "pool", "mixed"],
        help=(
            "Training opponent strategy. "
            "'heuristic': always train vs HeuristicBot (default, most stable). "
            "'self_play': train vs a frozen copy of self (AlphaZero style), "
            f"updated every OPPONENT_UPDATE_FREQ={OPPONENT_UPDATE_FREQ} steps. "
            "'pool': Fictitious Self-Play — picks one opponent per game: HeuristicBot "
            "(heuristic_ratio%% of games) or a random past checkpoint. "
            "'mixed': Per-move mixed — each move independently sampled from "
            "HeuristicBot (heuristic_ratio%%) or a random past checkpoint. Creates "
            "diverse mid-game transitions that neither pure opponent would produce. "
            "Eval always uses HeuristicBot regardless of this flag."
        ),
    )
    args = parser.parse_args()
    main(use_wandb=not args.no_wandb, args=args)
