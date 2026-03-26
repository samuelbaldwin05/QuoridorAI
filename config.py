"""
Hyperparameters for DQN training. Change values here and they
propagate to the model, buffer, and training loop.
"""

# game / action space
BOARD_SIZE = 9
NUM_ACTIONS = 137       # 8 pawn + 64 h-walls + 64 v-walls + 1 pass
NUM_CHANNELS = 4        # spatial channels: my pawn, opp pawn, h-walls, v-walls
NUM_SCALARS = 2         # my_walls_left/10, opp_walls_left/10

# network architecture
CONV1_FILTERS = 32
CONV2_FILTERS = 64
CONV_KERNEL_SIZE = 3
FC_HIDDEN_SIZE = 256

# training
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 500
GRADIENT_CLIP_NORM = 10.0

# replay buffer
REPLAY_BUFFER_SIZE = 50_000

# epsilon-greedy exploration
EPSILON_END = 0.05

# curriculum phase 1: vs RandomBot (weak, less exploration needed)
EPSILON_START_RANDOM = 0.5
EPSILON_DECAY_RANDOM = 0.999985

# curriculum phase 2: vs HeuristicBot (stronger, explore more)
EPSILON_START_HEURISTIC = 0.99
EPSILON_DECAY_HEURISTIC = 0.99999

# training budget
MAX_STEPS = 500_000

# evaluation
EVAL_FREQ = 1_000
EVAL_EPISODES = 50
WIN_RATE_TARGET = 0.80

# reproducibility
SEED = 42
