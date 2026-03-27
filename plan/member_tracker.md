# Member Tracker

## Sam Baldwin
- Built core game engine (`quoridor/game.py`) — movement, fencing, pathfinding, win detection
- Built heuristic bot (`agents/bot.py`) — priority chain: opening book, win-in-one, fence heuristics, UCS movement
- Built display module (`quoridor/display.py`) and play scripts (`scripts/play.py`, `scripts/play_bot.py`)

## Jeff Krapf — completed 2026-03-23
- Built action encoding (`quoridor/action_encoding.py`) — 137-action bijection
- Added `get_observation()` and `get_legal_mask()` to `quoridor/game.py`
- Built env wrapper (`quoridor/env.py`) — Gym-style single-agent interface
- Built DQN model (`agents/dqn_model.py`) — CNN with late fusion + action masking
- Built replay buffer (`agents/replay_buffer.py`) — 50k circular buffer
- Built random bot (`agents/random_bot.py`) — curriculum phase 1 opponent
- Built training loop (`scripts/train_dqn.py`) — Double DQN, Huber loss, curriculum support
- Built config (`config.py`) — all hyperparameters centralized
- Tests: 61 total across 6 test files, all passing

## Michael Maaseide — completed 2026-03-26
- Code cleanup across all source and test files
- Rewrote planning docs (`plan/project_overview.md`, `plan/pytorch_notes.md`, `plan/design_notes.md`)
- Ran initial training vs RandomBot — early results: ~60% win rate by step 4k, training ongoing

## Dreshta Boghra
- Caught up on ML and RL fundamentals (Q-values, Bellman equation, CNNs, state encoding, action space)
- Researched self-play training strategies: frozen opponent methodology, win-rate-driven epsilon annealing, and league play
- Wrote `Research/self_play_research.md` covering self-play training plan

## Current status
- DQN training vs RandomBot in progress (target: 80% win rate to graduate)
- Next: warm-start trained weights and train vs HeuristicBot
- After that: evaluate final model, consider PPO/self-play if time allows

## What's left
- Finish RandomBot training run, save best checkpoint
- Train vs HeuristicBot: `python scripts/train_dqn.py --opponent heuristic --checkpoint checkpoints/best.pt`
- Set up W&B logging for experiment tracking (`wandb login`, then run without `--no-wandb`)
- Prepare slides for presentation (due Apr 8, presentations Apr 10/14/17)
- Final deliverables due Apr 17
