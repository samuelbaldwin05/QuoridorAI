"""
Hyperparameters and constants for Quoridor RL training.

This is the single source of truth for all tunable values. Training scripts,
models, and the environment all import from here — never redefine a constant
locally. Change values here and they propagate everywhere.
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

# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------

CONV1_FILTERS = 32
CONV2_FILTERS = 64
CONV_KERNEL_SIZE = 3
FC_HIDDEN_SIZE = 256

# Number of residual blocks in ResNet backbone variants. Increasing this deepens
# the network without gradient vanishing (skip connections carry gradient directly).
# 2 is a good starting point; try 3-4 if the agent still can't learn wall strategy.
NUM_RESIDUAL_BLOCKS = 2

# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

STEP_PENALTY = -0.01   # small penalty per step to discourage stalling

# Penalty applied each time the agent revisits a pawn position it has already
# occupied this episode. Scales with visit count: penalty = REPETITION_PENALTY * (visits - 1).
# Discourages oscillation (moving back and forth) which is the #1 failure mode
# when two memoryless policies face each other. Set to 0.0 to disable.
REPETITION_PENALTY = -0.03

# Applied only on non-terminal steps; terminal rewards (+1/-1) remain pure
# so GAE advantage estimation is not distorted.
#
# shaped_reward = OPP_COEF  * (opp_path_after  - opp_path_before)  [good wall]
#               + SELF_COEF * (my_path_before  - my_path_after)    [good pawn move]
REWARD_SHAPING_OPP_COEF  = 0.05  # reward per step gained on opponent's shortest path
REWARD_SHAPING_SELF_COEF = 0.01  # reward per step saved on agent's own shortest path

# ---------------------------------------------------------------------------
# PPO hyperparameters
# ---------------------------------------------------------------------------

# Steps to collect before each PPO update. At ~30 steps/episode this is ~17
# episodes per rollout — enough to span multiple complete games so GAE sees
# terminal rewards, but short enough that the policy doesn't drift too far.
ROLLOUT_STEPS = 512

# Gradient passes over each rollout. More epochs extract more signal but risk
# violating the trust-region assumption (policy drifts from collection policy).
PPO_EPOCHS = 2

# Mini-batch size within each epoch. 512 steps / 64 = 8 gradient updates/epoch.
PPO_MINI_BATCH_SIZE = 64

# Probability ratio clipping threshold.
PPO_CLIP_EPS = 0.1

# GAE smoothing parameter (lambda). Combined with PPO_GAMMA=0.999, gives an
# effective horizon of 1/(1-0.999*0.97) ≈ 34 steps — covering Quoridor's
# 20-40 move games.
PPO_GAE_LAMBDA = 0.97

# Discount factor. 0.999 means all moves across a 30-40 step episode are
# weighted nearly equally (move 1 discounted by 0.999^39 ≈ 0.96).
PPO_GAMMA = 0.999

# Entropy regularization — scheduled from START → END linearly over training.
# Prevents the policy from becoming deterministic too early.
PPO_ENTROPY_COEF_START = 0.2
PPO_ENTROPY_COEF_END   = 0.01
PPO_MOVE_ENTROPY_COEF = 1.0   # multiplier on move-action entropy (relative to base coef)
PPO_WALL_ENTROPY_COEF = 0.25  # multiplier on wall-action entropy (relative to base coef)

# Weight on the value loss term. Keeps policy and value loss scales balanced.
PPO_VALUE_COEF = 0.5

# Learning rate for Adam optimizer (backbone + both heads).
PPO_LR = 3e-5

# Gradient norm clip. 0.5 is the standard PPO recommendation.
PPO_GRAD_CLIP = 0.5

# Total env steps before stopping.
PPO_MAX_STEPS = 1_000_000

# Eval frequency and episodes.
EVAL_FREQ = 2_000
EVAL_EPISODES = 20
WIN_RATE_TARGET = 0.80  # graduation threshold (disabled with --no-graduate)

# ---------------------------------------------------------------------------
# Self-play / opponent pool
# ---------------------------------------------------------------------------

# Sync current model into the opponent pool every N env steps.
OPPONENT_UPDATE_FREQ = 25_000

# Maximum past checkpoints kept in memory. Oldest dropped when full.
OPPONENT_POOL_SIZE = 5

# Fraction of games/moves that use HeuristicBot instead of a pool checkpoint.
OPPONENT_HEURISTIC_RATIO = 0.4

# ---------------------------------------------------------------------------
# ε-greedy HeuristicBot curriculum
# ---------------------------------------------------------------------------

# With probability epsilon the HeuristicBot takes a random legal action instead.
# Annealing: epsilon = max(0, START * (1 - rolling_win_rate / THRESHOLD))
EPSILON_CURRICULUM_START     = 0.3
EPSILON_CURRICULUM_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Auxiliary prediction head
# ---------------------------------------------------------------------------

# Weight on the aux path-length regression loss (bfs_resnet model only).
AUX_LOSS_COEF = 0.1

# ---------------------------------------------------------------------------
# Behavioral Cloning (BC) pretraining
# ---------------------------------------------------------------------------

BC_GAMES = 1_000        # expert games to generate
BC_EPOCHS = 20          # supervised training passes
BC_LR = 1e-3            # higher LR than RL (clean supervised labels)
BC_BATCH_SIZE = 128

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42
