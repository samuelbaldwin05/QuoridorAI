"""
Hyperparameters for DQN training. Change values here and they
propagate to the model, buffer, and training loop.
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
# shaped_reward = OPP_COEF  * (opp_path_after  - opp_path_before)  [good wall]
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

# network architecture
CONV1_FILTERS = 32
CONV2_FILTERS = 64
CONV_KERNEL_SIZE = 3
FC_HIDDEN_SIZE = 256

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
STEP_PENALTY = -0.01   # small penalty per step to discourage stalling

# Penalty applied each time the agent revisits a pawn position it has already
# occupied this episode. Scales with visit count: penalty = REPETITION_PENALTY * (visits - 1).
# Discourages oscillation (moving back and forth) which is the #1 failure mode
# when two memoryless policies face each other. Set to 0.0 to disable.
REPETITION_PENALTY = -0.03

# replay buffer
REPLAY_BUFFER_SIZE = 50_000

# epsilon-greedy exploration
EPSILON_END = 0.05

# curriculum phase 1: vs RandomBot (weak, less exploration needed)
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

# training budget
MAX_STEPS = 500_000

# Evaluate win rate every N *env steps* (not gradient steps) so the frequency
# is consistent regardless of when the replay buffer starts filling.
EVAL_FREQ = 2_000       # was 5_000 → 1_000 → 2_000; halved to reduce eval overhead
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
# Too frequent = non-stationarity (policy oscillates before converging vs current opp).
# Too infrequent = agent quickly surpasses opponent, games become one-sided.
# At ~30 plies/game and ROLLOUT_STEPS=512, 25k steps ≈ 833 games between syncs —
# enough for the policy to converge against one opponent before seeing a new one.
# Research: OpenAI Five saves every 20k–50k eps; AlphaGo Zero uses a 55% win-rate gate.
OPPONENT_UPDATE_FREQ = 25_000

# ---------------------------------------------------------------------------
# Opponent pool (Fictitious Self-Play)
# ---------------------------------------------------------------------------

# Maximum number of past checkpoints kept in memory. Oldest entry dropped when full.
# 5 checkpoints × 25k steps = strategies spanning ~125k steps of history, enough
# diversity to prevent the cycling seen with a single frozen opponent.
# Research: Neural FSP (Heinrich & Silver, 2016) — uniform pool sampling converges
# to Nash. Full PFSP (AlphaStar) adds difficulty-weighting, overkill at this scale.
OPPONENT_POOL_SIZE = 5

# Fraction of episodes that sample HeuristicBot instead of a pool checkpoint.
# Provides a stable non-moving anchor that prevents full collapse when all pool
# checkpoints are still weak (inevitable early in training).
# Research: OpenAI Five 80/20 latest-vs-historical split. Here the heuristic acts
# as the "historical" anchor until enough strong checkpoints accumulate.
OPPONENT_HEURISTIC_RATIO = 0.4

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

# Probability ratio clipping threshold.
# Tightened from 0.2 → 0.1 for BC warm-start runs: 0.2 allowed the policy to
# drift far enough from the BC distribution in the first few updates to erase
# trained actor-head weights. 0.1 enforces a tighter trust region, preserving
# BC knowledge while still allowing RL refinement.
PPO_CLIP_EPS = 0.1

# GAE smoothing parameter (lambda). Combined with PPO_GAMMA=0.999, gives an
# effective horizon of 1/(1-0.999*0.97) ≈ 34 steps — covering Quoridor's
# 20-40 move games. Previous 0.95 gave ~17 steps, meaning wall placements in
# the first half of the game received almost no gradient signal from terminal rewards.
PPO_GAE_LAMBDA = 0.97

# Discount factor. Raised from 0.99 → 0.999 so all moves across a 30-40 step
# episode are weighted nearly equally. With 0.99, move 1 value is discounted
# by 0.99^39 ≈ 0.67 relative to the terminal — too much for long-horizon planning.
PPO_GAMMA = 0.999

# Entropy regularization coefficient — scheduled, not fixed.
# THE key hyperparameter preventing "always-lose collapse": penalizes the policy
# for becoming deterministic, keeping it exploring even when all outcomes are -1.
#
# Raised from 0.05 → 0.2: the previous value was insufficient — training
# collapsed to 0.1-0.3 nats (near-deterministic) while healthy range is 1-3 nats.
# At 0.2, the entropy bonus dominates early training and forces genuine wall
# exploration before the policy is allowed to specialise.
# Decay to 0.01 so the policy can commit to a strategy once it has a win signal.
PPO_ENTROPY_COEF_START = 0.2
PPO_ENTROPY_COEF_END   = 0.01
# Separate entropy coefficients for move vs wall actions.
# Wall coef is lower because 128 wall actions naturally dominate the entropy
# term — a single global coef pushes ~50% probability mass onto walls.
PPO_MOVE_ENTROPY_COEF = 1.0   # multiplier on move-action entropy (relative to base coef)
PPO_WALL_ENTROPY_COEF = 0.25  # multiplier on wall-action entropy (relative to base coef)

# Weight on the value loss term. Keeps policy and value loss scales balanced.
PPO_VALUE_COEF = 0.5

# Single Adam optimizer LR for backbone + both heads.
# Reduced from 3e-4 → 3e-5: combined with PPO_EPOCHS=4, the original LR caused
# KL of 0.27-0.37 and destroyed the BC warmstart (20% win rate → 0%) within 2k steps.
PPO_LR = 3e-5

# Gradient norm clip. Tightened from 10.0 → 0.5: the previous value never
# fired (observed grad_norm was 1-3), providing no protection. 0.5 is the
# standard PPO recommendation and prevents large destructive updates
# particularly in early training when loss landscape is steep.
PPO_GRAD_CLIP = 0.5

# Total env steps before stopping. PPO is more sample-efficient than DQN
# (no replay buffer lag, on-policy data), so 500k should be a good first run.
PPO_MAX_STEPS = 1_000_000

# ---------------------------------------------------------------------------
# ε-greedy HeuristicBot curriculum
# ---------------------------------------------------------------------------

# EpsilonHeuristicBot replaces HeuristicBot during training when
# --epsilon-curriculum is passed. With probability epsilon it takes a uniformly
# random legal action; otherwise it delegates to HeuristicBot as normal.
# This gives the agent winnable games early when it would otherwise never beat
# a full-strength HeuristicBot, bootstrapping the positive reward signal needed
# for PPO to learn.
#
# Annealing formula (applied at every eval cycle):
#   epsilon = max(0.0, EPSILON_CURRICULUM_START * (1 - rolling_win_rate / THRESHOLD))
#
# At win_rate=0:     epsilon = 0.3  (bot is random 30% of the time)
# At win_rate=0.6:   epsilon = 0.0  (full-strength HeuristicBot, curriculum complete)
# Linear interpolation between — epsilon tracks the agent's progress.
#
# Rolling win rate is computed from the last 200 *training* episodes (not eval),
# so this adapts to actual training performance, not just the eval snapshot.
EPSILON_CURRICULUM_START     = 0.3   # initial epsilon (bot weakening)
EPSILON_CURRICULUM_THRESHOLD = 0.6   # win rate at which epsilon reaches 0

# ---------------------------------------------------------------------------
# Auxiliary prediction heads
# ---------------------------------------------------------------------------

# Weight on the auxiliary path-length regression loss.
# The aux head predicts [my_dist_to_goal, opp_dist_to_goal] from the shared
# backbone encoding. Targets are derived for free from BFS channels already
# stored in the rollout (ch4 * ch0 dot-product gives distance at pawn position).
# Forces the backbone to build explicit path-distance representations, which
# directly supports wall-placement reasoning.
# Only active for BFS model variants (bfs, bfs_resnet). Non-BFS models ignore this.
AUX_LOSS_COEF = 0.1

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
