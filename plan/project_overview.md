# QuoridorAI — Project Overview

CS4100 final project. Building a deep RL agent to play Quoridor.

## Team
Michael Maaseide, Sam Baldwin, Jeff Krapf, Dreshta Boghra

## What we're doing
Training a DQN agent from scratch to beat our heuristic bot, then hopefully
moving on to PPO and eventually AlphaZero-style MCTS if time allows.

## Stack
- Python 3.10+, PyTorch
- Custom game engine (no Gym dependency, but Gym-compatible interface)
- W&B for experiment tracking
- Git/GitHub

## State representation
4-channel (9x9) spatial tensor + 2 scalar features:
- ch0: current player pawn
- ch1: opponent pawn
- ch2: horizontal walls
- ch3: vertical walls
- scalars: [my_walls_left/10, opp_walls_left/10]

Board is always shown from the current player's perspective (flipped for P1).

## Action space
137 total: 8 pawn directions + 64 h-walls + 64 v-walls + 1 pass (always illegal).
See `quoridor/action_encoding.py` for the full layout.

## Key files
- `quoridor/game.py` — core game logic (Sam)
- `quoridor/env.py` — RL environment wrapper (Jeff)
- `agents/dqn_model.py` — CNN Q-network (Jeff)
- `agents/replay_buffer.py` — experience replay buffer (Jeff)
- `agents/bot.py` — heuristic bot opponent (Sam)
- `agents/random_bot.py` — simple random opponent for curriculum
- `scripts/train_dqn.py` — training loop
- `config.py` — all hyperparameters in one place

## Training plan
1. Train DQN vs RandomBot until ~80% win rate
2. Warm-start and train vs HeuristicBot
3. If that works, move to PPO/self-play
