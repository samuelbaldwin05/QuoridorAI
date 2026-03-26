# Design Notes

Decisions we made for the DQN phase and why.

## Action encoding (137 actions)
- 0-7: pawn directions (cardinal + diagonal). Diagonal needed for jump situations.
- 8-71: h-walls (row-major, `8 + row*8 + col`)
- 72-135: v-walls (row-major, `72 + row*8 + col`)
- 136: pass (reserved, always illegal)

We encode direction deltas, not absolute positions. "move up" covers both
1-step and 2-step jumps since they share the same sign. The env resolves
which destination to use by scanning get_valid_moves().

## CNN architecture
Kept it small (2 conv layers, 32→64 filters) since we're training on CPU/MPS.
Bigger nets would take too long per step. If learning stalls, try adding a
third conv layer before blaming the algorithm.

## Late fusion for scalars
Wall counts are global info, not spatial. Broadcasting them into 9x9 channels
would be misleading. We concat them after flattening the CNN output.

## Double DQN (upgraded from vanilla)
First training run showed Q-values above 1.0 (should max at 1.0 with our
reward scheme). Classic maximization bias. Double DQN fixes it — online net
picks the action, target net evaluates it. 3-line change.

## Training opponent
Start with RandomBot (easier, agent gets some wins early), then warm-start
against HeuristicBot. Self-play is too unstable for DQN.

## Curriculum epsilon schedules
- RandomBot phase: start at 0.5, decay to 0.05 over ~130k steps
- HeuristicBot phase: start at 0.99, slower decay to 0.05 over ~300k steps

Higher exploration for the harder opponent.

## Replay buffer
50k transitions, preallocated numpy arrays. Stores next_legal_mask because
Double DQN needs it to pick the best next action. Sampling without replacement.

## Hardware constraints
Everything sized for <1hr training on CPU. Buffer=50k, batch=64, max=500k steps.
If it doesn't converge, try running longer before changing the architecture.

## Evaluation
Win rate vs the training opponent every 1k steps, 50 episodes each eval.
Graduation threshold: 80% win rate.

## PyTorch version
Had to upgrade from 2.1.2 to 2.11.0 because old torch couldn't handle numpy 2.x.
torchvision now shows a version warning but we don't use it.
