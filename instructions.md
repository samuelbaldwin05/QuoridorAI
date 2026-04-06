# Running the PPO Quoridor Agent

## Prerequisites

- Python 3.11+
- Install dependencies:
  ```bash
  pip install numpy torch
  ```

## Available Checkpoints

All checkpoints are in `checkpoints/`. The PPO final checkpoints follow the naming pattern `ppo_final_<steps>.pt`:

| Checkpoint | Steps | Date | Notes |
|---|---|---|---|
| `ppo_final_922112.pt` | 922k | Apr 3 | Most trained — latest run |
| `best_ppo.pt` | varies | Apr 3 | Highest eval win rate during training |
| `ppo_final_500224.pt` | 500k | Apr 2 | Earlier run |
| `bc_pretrained_bfs_resnet.pt` | 0 (BC only) | Apr 2 | Behavioral cloning warm-start (no PPO) |

All PPO checkpoints use the **`bfs_resnet`** model architecture (6-channel ResNet with BFS distance maps).

## Key Files

| File | Purpose |
|---|---|
| `agents/ppo_bot.py` | Bot wrapper — loads checkpoint and provides `choose_action(game)` |
| `agents/ppo_model_bfs_resnet.py` | Neural network architecture (6-ch ResNet + LayerNorm + aux head) |
| `quoridor/game.py` | Game engine — rules, move validation, BFS pathfinding |
| `quoridor/env.py` | Gym-compatible environment wrapper |
| `quoridor/action_encoding.py` | Maps 137 action indices to game moves |
| `quoridor/display.py` | Terminal board rendering |
| `config.py` | All hyperparameters |
| `scripts/train_ppo.py` | Training loop (produces the checkpoints) |

## How to Run

### 1. Watch the agent play (terminal visualization)

```bash
# Watch PPO agent vs heuristic bot (5 games, greedy play)
python -m scripts.watch_ppo \
    --checkpoint checkpoints/ppo_final_922112.pt \
    --model bfs_resnet \
    --opponent heuristic \
    --games 5 \
    --delay 0.3

# Watch vs random bot
python -m scripts.watch_ppo \
    --checkpoint checkpoints/ppo_final_922112.pt \
    --opponent random

# Stochastic (sampled) play instead of greedy
python -m scripts.watch_ppo \
    --checkpoint checkpoints/ppo_final_922112.pt \
    --no-greedy
```

### 2. Play against the bot interactively

`scripts/play_bot.py` currently only supports playing against the HeuristicBot. To play against the PPO agent, use the `PPOBot` class directly in Python:

```python
from agents.ppo_bot import PPOBot
from quoridor.game import QuoridorState
from quoridor.display import render

bot = PPOBot("checkpoints/ppo_final_922112.pt", model_type="bfs_resnet")
game = QuoridorState()

# Bot chooses an action
action = bot.choose_action(game)
# Returns ("move", row, col) or ("fence", row, col, "h"/"v")
```

### 3. Use in code (evaluation / tournaments)

```python
from agents.ppo_bot import PPOBot
from agents.bot import HeuristicBot
from quoridor.game import QuoridorState

ppo = PPOBot("checkpoints/ppo_final_922112.pt", model_type="bfs_resnet")
opp = HeuristicBot()

wins = 0
for _ in range(100):
    game = QuoridorState()
    ppo.reset()
    opp.reset()
    while not game.done:
        if game.turn == 0:
            action = ppo.choose_action(game)
        else:
            action = opp.choose_action(game)
        if action[0] == "move":
            game.move_to(action[1], action[2])
        else:
            game.place_fence(action[1], action[2], action[3])
    if game.winner == 0:
        wins += 1

print(f"Win rate: {wins}%")
```

### 4. Resume or start PPO training

```bash
python -m scripts.train_ppo \
    --model bfs_resnet \
    --opponent pool \
    --reward-shaping \
    --checkpoint checkpoints/bc_pretrained_bfs_resnet.pt \
    --value-burnin-steps 10000
```

## Architecture Summary

The `bfs_resnet` model takes:
- **Spatial input:** `(6, 9, 9)` tensor — pawn positions (2ch), walls (2ch), BFS distance maps (2ch)
- **Scalar input:** `(2,)` — normalized wall counts for each player

It outputs:
- **Action distribution** over 137 actions (8 pawn moves + 128 wall placements + 1 pass), with illegal actions masked
- **Value estimate** (scalar)
- **Auxiliary prediction** of path distances (training signal only)
