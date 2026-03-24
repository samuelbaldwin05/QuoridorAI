Dreshta - Catching up on Machine learning
Michael - Research.md on Reinforcement learning
Jeff - Research.md on Reinforcement learning

Jeff — Stages 2 & 3 (DQN Core + Training Loop) — completed 2026-03-23

Built:
- config.py (project root) — all hyperparameters as named constants
- agents/dqn_model.py — CNN Q-network with late fusion, masked Q-values
- agents/replay_buffer.py — preallocated circular buffer, 50k capacity
- scripts/train_dqn.py — full training loop with epsilon-greedy, TD loss,
  target network hard updates, W&B logging, checkpointing
- tests/test_dqn_model.py — 11 tests (shapes, masking, gradients, BatchNorm modes)
- tests/test_replay_buffer.py — 11 tests (push/sample correctness, circular overwrite)

What's next (Stage 4):
- Set up W&B (run: wandb login)
- Run full training: python scripts/train_dqn.py
- Run 200-episode evaluation once training completes
- Save final checkpoint + config as W&B artifact
- Update this file with final win rate and any remaining open questions
Sam - Implementing heuristic agent in python
