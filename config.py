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
NUM_ACTIONS = 137       # 8 pawn moves + 64 h-walls + 64 v-walls + 1 pass
                        # (see quoridor/action_encoding.py for full layout)
NUM_CHANNELS = 4        # spatial input channels: my pawn, opp pawn, h-walls, v-walls
NUM_SCALARS = 2         # non-spatial features: [my_walls_left/10, opp_walls_left/10]

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

EPSILON_START = 1.0     # start fully random to collect diverse early experiences
EPSILON_END = 0.05      # always keep 5% exploration so the agent doesn't over-exploit
EPSILON_DECAY = 0.999   # multiplicative decay applied once per episode
                        # reaches EPSILON_END after ~2,900 episodes

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
EVAL_FREQ = 1_000

EVAL_EPISODES = 50      # number of games per evaluation window
WIN_RATE_TARGET = 0.80  # graduation threshold to advance to PPO phase

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42               # applied to torch, numpy, and the environment at startup
