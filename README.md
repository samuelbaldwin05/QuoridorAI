# QuoridorAI

Deep reinforcement learning agent for the board game [Quoridor](https://en.wikipedia.org/wiki/Quoridor), built from scratch for CS4100.

**Team:** Michael Maaseide, Sam Baldwin, Jeff Krapf, Dreshta Boghra

## Overview

We train a neural network to play Quoridor using PPO (Proximal Policy Optimization) with self-play.
The agent learns entirely from playing games against itself and a heuristic opponent — no human game
data, no hand-coded strategy. The project progresses through three algorithmic phases:

1. **DQN** — foundational RL concepts (completed, abandoned due to 0% win rate ceiling)
2. **PPO** — stable adversarial training with self-play (current phase)
3. **AlphaZero** — MCTS + neural network (future goal)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train,dev]"
```

## Quick Start

### Play against the bot

```bash
python -m scripts.play --opponent ppo --checkpoint checkpoints/best_mixed-v2.pt
```

Move by typing a cell (e.g. `e5`). Place walls with `h a3` (horizontal) or `v c2` (vertical).

### Watch two bots play

```bash
# PPO vs heuristic
python -m scripts.watch --checkpoint checkpoints/best_mixed-v2.pt --model bfs_resnet --games 5

# PPO vs PPO
python -m scripts.watch --checkpoint checkpoints/best_mixed-v2.pt --opponent ppo \
    --opponent-checkpoint checkpoints/best_strong_shaping.pt
```

### Evaluate win rate (batch, no visualization)

```bash
python -m scripts.eval_bots --p0 ppo --p0-checkpoint checkpoints/best_mixed-v2.pt --p1 heuristic --games 100
```

### Analyze opening diversity

```bash
python -m scripts.opening_diversity --checkpoint checkpoints/best_mixed-v2.pt --games 100 --moves 8
```

### Train a new agent

```bash
python -m scripts.train_ppo \
    --model bfs_resnet \
    --opponent mixed \
    --reward-shaping \
    --randomize-start \
    --checkpoint checkpoints/best_mixed-v2.pt \
    --no-graduate \
    --run-name my-run \
    --max-steps 500000
```

See `python -m scripts.train_ppo --help` for all options.

## Project Structure

```
QuoridorAI/
├── config.py                       # All hyperparameters (single source of truth)
│
├── quoridor/                       # Game engine
│   ├── game.py                     #   Core rules, wall validation, BFS pathfinding
│   ├── env.py                      #   Gym-style RL wrapper (agent=P0, bot=P1)
│   ├── vec_env.py                  #   Vectorized env for parallel rollout collection
│   ├── action_encoding.py          #   Maps 137 action indices ↔ game moves
│   └── display.py                  #   Terminal board rendering
│
├── agents/                         # Bot implementations
│   ├── bot.py                      #   HeuristicBot + EpsilonHeuristicBot
│   ├── random_bot.py               #   Uniform random baseline
│   ├── ppo_bot.py                  #   PPO checkpoint wrapper (for eval/play)
│   ├── ppo_model.py                #   PPO baseline: 4-ch CNN
│   ├── ppo_model_resnet.py         #   PPO resnet: 4-ch ResNet + LayerNorm
│   ├── ppo_model_bfs.py            #   PPO bfs: 6-ch CNN (adds BFS distance maps)
│   └── ppo_model_bfs_resnet.py     #   PPO bfs_resnet: 6-ch ResNet + aux head (primary)
│
├── scripts/                        # Runnable entry points
│   ├── train_ppo.py                #   PPO training loop (1800 lines, the core script)
│   ├── pretrain_bc.py              #   Behavioral cloning pretraining
│   ├── generate_bc_data.py         #   Generate expert data from HeuristicBot games
│   ├── play.py                     #   Play interactively vs a bot
│   ├── watch.py                    #   Watch bot-vs-bot games in terminal
│   ├── eval_bots.py                #   Batch evaluation (win rates, no visualization)
│   └── opening_diversity.py        #   Analyze how diverse a bot's openings are
│
├── tests/                          # pytest suite
│   ├── test_game.py                #   Game rules and state management
│   ├── test_env.py                 #   Environment wrapper
│   ├── test_bfs_env.py             #   BFS distance map observations
│   ├── test_action_encoding.py     #   Action index ↔ game move bijection
│   ├── test_legal_mask.py          #   Legal action masking
│   ├── test_observation.py         #   State tensor construction
│   ├── test_bot_legality.py        #   Bot actions are always legal
│   └── test_heuristic_bot_speed.py #   Performance regression test
│
├── docs/                           # Project documentation
│   ├── research.md                 #   RL theory primer (beginner-friendly)
│   ├── decisions.md                #   Architectural decision log
│   ├── ppo_techniques.md           #   PPO implementation reference
│   └── member_tracker.md           #   Team contributions
│
├── checkpoints/                    # Saved model weights (not tracked in git)
├── data/                           # BC training data (not tracked in git)
├── logs/                           # Training logs (not tracked in git)
└── pyproject.toml                  # Dependencies
```

## How It Works

### State Representation

The board is encoded as a `(6, 9, 9)` tensor (with BFS channels) or `(4, 9, 9)` (without):

| Channel | Content |
|---|---|
| 0 | Current player pawn position (one-hot) |
| 1 | Opponent pawn position (one-hot) |
| 2 | Horizontal walls |
| 3 | Vertical walls |
| 4 | Current player BFS distance map (normalized) |
| 5 | Opponent BFS distance map (normalized) |

The board is always normalized to the current player's perspective (flipped for P1).

### Action Space

137 discrete actions: 8 pawn moves (cardinal + diagonal jumps) + 64 horizontal walls + 64 vertical walls + 1 pass (always illegal). Illegal actions are masked before sampling.

### Training Pipeline

```
Behavioral Cloning (pretrain_bc.py)
    → warm-starts the policy head from HeuristicBot games
    ↓
PPO Training (train_ppo.py)
    → self-play with opponent pool (mixed per-move sampling)
    → reward shaping (path-length deltas)
    → entropy scheduling (prevents premature convergence)
    → randomized starting side (agent practices as both P0 and P1)
    ↓
Evaluation (eval_bots.py, opening_diversity.py)
    → win rate vs HeuristicBot from both sides
    → opening diversity analysis
```

## Key Design Decisions

- **Action masking** over dynamic output sizes — simpler, standard practice
- **Current-player perspective normalization** — network learns one set of patterns, not two
- **LayerNorm** over BatchNorm in PPO models — consistent train/eval behavior
- **Per-rollout advantage normalization** — avoids amplifying noise from sparse rewards
- **Opponent pool (Fictitious Self-Play)** — prevents cycling against a single frozen opponent

See `docs/decisions.md` for the full decision log with rationale.
