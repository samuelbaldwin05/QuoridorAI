"""
config.py — Centralized hyperparameters for the DQN training pipeline.

All constants are defined here so that the model, replay buffer, and training
loop stay in sync without hunting through multiple files. Change a value once
here and it propagates everywhere.

Import pattern:
    from config import BATCH_SIZE, GAMMA, ...
"""

# ---------------------------------------------------------------------------
# Game / action space
# ---------------------------------------------------------------------------

BOARD_SIZE = 9          # Quoridor is played on a 9x9 grid
FENCE_GRID = BOARD_SIZE - 1  # fence placement grid is (BOARD_SIZE-1)×(BOARD_SIZE-1)
INITIAL_WALLS_PER_PLAYER = 10  # each player starts with 10 walls
NUM_ACTIONS = 137       # 8 pawn moves + 64 h-walls + 64 v-walls + 1 pass
                        # (see quoridor/action_encoding.py for full layout)
NUM_CHANNELS = 4        # spatial input channels: my pawn, opp pawn, h-walls, v-walls
NUM_CHANNELS_BFS = 6    # extended channels: adds ch4 (my BFS dist map) + ch5 (opp BFS dist map)
NUM_SCALARS = 2         # non-spatial features: [my_walls_left/10, opp_walls_left/10]

# BFS distance map normalisation divisor. On a 9×9 board the worst-case shortest
# path is well under 18 steps in any reachable game position. Dividing by this
# keeps all distance map values in [0, 1] without saturating at typical distances.
BFS_NORM_FACTOR = BOARD_SIZE * 2  # = 18

# Number of residual blocks in ResNet backbone variants. Increasing this deepens
# the network without gradient vanishing (skip connections carry gradient directly).
# 2 is a good starting point; try 3-4 if the agent still can't learn wall strategy.
NUM_RESIDUAL_BLOCKS = 2

# Reward shaping coefficients. Applied only on non-terminal steps; terminal
# rewards (+1/-1) remain pure so GAE advantage estimation is not distorted.
#
# shaped_reward = OPP_COEF  * (opp_path_before - opp_path_after)   [good wall]
#               + SELF_COEF * (my_path_before  - my_path_after)    [good pawn move]
#
# With ~12 pawn moves and ~4 wall placements per episode, cumulative shaped
# reward is ~0.18 for a well-played game — dominated by the terminal ±1.0.
# Tune the ratio between the two coefficients if the agent over-prioritizes
# pawn movement at the expense of wall strategy (or vice versa).
REWARD_SHAPING_OPP_COEF  = 0.05  # reward per step gained on opponent's shortest path
REWARD_SHAPING_SELF_COEF = 0.01  # reward per step saved on agent's own shortest path

# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------

CONV1_FILTERS = 32
CONV2_FILTERS = 64
CONV_KERNEL_SIZE = 3    # 3x3 kernels; padding=1 preserves spatial dims
FC_HIDDEN_SIZE = 256    # size of the fully-connected layer after late fusion

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

LEARNING_RATE = 1e-4
# Lower LR for heuristic phase: warm-started weights should be updated
# conservatively so early noisy gradients don't overwrite what was learned
# against the random bot.
LEARNING_RATE_HEURISTIC = 5e-5
GAMMA = 0.99            # discount factor; Quoridor episodes are long so keep high
BATCH_SIZE = 64         # CPU-scaled; larger batches need more RAM and slow each step

# Hard target-network update: copy online → target every N env steps.
# Periodic hard updates are more stable than soft updates for vanilla DQN.
TARGET_UPDATE_FREQ = 500

# Gradient clipping prevents exploding gradients, which are common early in
# training when Q-value estimates are wildly inaccurate.
GRADIENT_CLIP_NORM = 10.0

# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

# CPU constraint: 50k transitions × (4,9,9) float32 spatial ≈ ~650 MB.
# Increase to 100k+ if RAM allows; more diversity helps training stability.
REPLAY_BUFFER_SIZE = 50_000

# ---------------------------------------------------------------------------
# Exploration (epsilon-greedy)
# ---------------------------------------------------------------------------

# Shared floor — always keep 5% exploration so the agent doesn't over-exploit
EPSILON_END = 0.05

# RandomBot phase (curriculum phase 1) — weak opponent, less exploration needed.
# Start half-exploiting; reaches EPSILON_END at ~130k steps.
EPSILON_START_RANDOM = 0.5
EPSILON_DECAY_RANDOM = 0.999985

# HeuristicBot phase (curriculum phase 2) — stronger opponent, warm-started weights.
# Start at 0.5 (not 0.99) so the warm-started policy is exploited early.
# Starting at 0.99 would ignore the checkpoint for ~30k steps, defeating the
# purpose of curriculum warm-starting.
EPSILON_START_HEURISTIC = 0.5

# Self-play phase — frozen DQNBot opponent, periodically updated.
# Start at 0.3: warm-started weights are good, exploit early but keep some
# exploration to adapt as the opponent gets stronger after each sync.
# Reaches EPSILON_END at ~200k steps, allowing re-exploration after syncs.
EPSILON_START_SELF_PLAY = 0.3
EPSILON_DECAY_SELF_PLAY = 0.99997
EPSILON_DECAY_HEURISTIC = 0.99999

# ---------------------------------------------------------------------------
# Training budget
# ---------------------------------------------------------------------------

# Total env steps before training stops. At ~30 steps/episode this is ~17k
# episodes. Tune up to 1_000_000 if the agent hasn't converged by then.
MAX_STEPS = 500_000

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Evaluate win rate every N *env steps* (not gradient steps) so the frequency
# is consistent regardless of when the replay buffer starts filling.
EVAL_FREQ = 2_000       # was 1_000; halved to reduce eval overhead per training run

EVAL_EPISODES = 20      # was 50; fewer games per window, faster eval
WIN_RATE_TARGET = 0.80  # graduation threshold to advance to PPO phase

# ---------------------------------------------------------------------------
# Prioritized Experience Replay (PER)
# ---------------------------------------------------------------------------

# How strongly to prioritize high-TD-error transitions.
# 0 = uniform sampling (same as ReplayBuffer), 1 = fully proportional.
PER_ALPHA = 0.6

# Importance-sampling (IS) correction exponent, annealed from START → END
# over MAX_STEPS. At beta=1.0 the bias introduced by non-uniform sampling
# is fully corrected; at beta=0 no correction is applied.
PER_BETA_START = 0.4
PER_BETA_END   = 1.0

# Small floor so every transition has non-zero sampling probability.
PER_EPSILON = 1e-6

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Self-play opponent update frequency
# ---------------------------------------------------------------------------

# Hard-copy online_net → frozen opponent every N env steps during self-play.
# Too frequent = non-stationarity (DQN assumes stable target distribution).
# Too infrequent = agent quickly surpasses opponent, games become one-sided.
OPPONENT_UPDATE_FREQ = 10_000

# ---------------------------------------------------------------------------
# N-step returns
# ---------------------------------------------------------------------------

# Number of steps to accumulate before computing a bootstrapped TD target.
# With n=5, the terminal reward propagates to states 5 steps earlier in a
# single replay update instead of requiring 5 separate gradient passes.
# This is especially effective when episodes are short and rewards are sparse.
N_STEP_RETURNS = 5

# ---------------------------------------------------------------------------
# PPO hyperparameters
# ---------------------------------------------------------------------------

# Steps to collect before each PPO update. At ~30 steps/episode this is ~17
# episodes per rollout — enough to span multiple complete games so GAE sees
# terminal rewards, but short enough that the policy doesn't drift too far.
ROLLOUT_STEPS = 512

# Gradient passes over each rollout. More epochs extract more signal but risk
# violating the trust-region assumption (policy drifts from collection policy).
# Reduced from 4 → 2: KL was 0.27-0.37 (want < 0.05) with 4 epochs, destroying
# the BC warmstart within the first few thousand steps.
PPO_EPOCHS = 2

# Mini-batch size within each epoch. 512 steps / 64 = 8 gradient updates/epoch.
# Must be >= 32 for stable BatchNorm statistics.
PPO_MINI_BATCH_SIZE = 64

# Probability ratio clipping threshold. Prevents the policy from moving more
# than ±20% away from the collection policy in a single update.
PPO_CLIP_EPS = 0.2

# GAE smoothing parameter (lambda). 0.95 is close to Monte Carlo (λ=1) which
# gives low bias at the cost of some variance — appropriate for Quoridor's
# long episodes where accurate credit assignment requires long return horizons.
PPO_GAE_LAMBDA = 0.95

# Discount factor — matches DQN GAMMA. High value needed for long-horizon
# planning across 20-40 move games.
PPO_GAMMA = 0.99

# Entropy regularization coefficient — scheduled, not fixed.
# THE key hyperparameter preventing "always-lose collapse": penalizes the policy
# for becoming deterministic, keeping it exploring even when all outcomes are -1.
#
# Start high (0.05) to prevent early wall-action collapse: without this, the
# policy quickly assigns near-zero probability to all 128 wall placements and
# only moves pawns. Decay to 0.005 so the policy can specialize once it has
# learned which actions matter. train_ppo.py linearly interpolates between these
# values over the full training run.
PPO_ENTROPY_COEF_START = 0.05
PPO_ENTROPY_COEF_END   = 0.005

# Weight on the value loss term. Keeps policy and value loss scales balanced.
PPO_VALUE_COEF = 0.5

# Single Adam optimizer LR for backbone + both heads.
# Reduced from 3e-4 → 3e-5: combined with PPO_EPOCHS=4, the original LR caused
# KL of 0.27-0.37 and destroyed the BC warmstart (20% win rate → 0%) within 2k steps.
PPO_LR = 3e-5

# Gradient norm clip — matches existing GRADIENT_CLIP_NORM.
PPO_GRAD_CLIP = 10.0

# Total env steps before stopping. PPO is more sample-efficient than DQN
# (no replay buffer lag, on-policy data), so 500k should be a good first run.
PPO_MAX_STEPS = 500_000

# ---------------------------------------------------------------------------
# Behavioral Cloning (BC) pretraining
# ---------------------------------------------------------------------------

# Number of HeuristicBot vs HeuristicBot games to generate as expert data.
# 1000 games × ~30 moves/game ≈ 60k (state, action) pairs — enough for
# stable supervised training without excessive generation time (~10 min).
BC_GAMES = 1_000

# Supervised training passes over the full BC dataset.
# 20 epochs is typically enough for CE loss to plateau on 60k samples.
BC_EPOCHS = 20

# Higher LR than RL phase: supervised labels are clean (one-hot expert action),
# so we can take larger steps without the instability seen in noisy TD targets.
BC_LR = 1e-3

# Larger batch than RL: no replay buffer capacity pressure, and larger batches
# reduce gradient noise for the supervised task.
BC_BATCH_SIZE = 128

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42               # applied to torch, numpy, and the environment at startup
