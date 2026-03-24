
# Quoridor Deep Reinforcement Learning — Project Memory

## Project Overview

We are building a deep RL agent to play Quoridor from scratch, progressing through three algorithmic phases:
1. **DQN** — foundational concept implementation (current phase)
2. **PPO** — stable adversarial/self-play training
3. **AlphaZero** — MCTS + neural network, the end goal

This is both a learning project (beginners to deep RL) and a serious performance goal. Code must be
readable, well-commented, and correct. Clarity is never sacrificed for cleverness.

---

## Stack

- **Language:** Python 3.10+
- **Framework:** PyTorch (not TensorFlow, not JAX)
- **Environment:** Custom Quoridor env built from scratch (Gym-compatible interface)
- **Experiment tracking:** (TBD — likely Weights & Biases or TensorBoard)
- **Version control:** Git / GitHub

---

## Architecture Decisions (Finalized — Do Not Revisit Without Discussion)

| Decision | Choice | Rationale |
|---|---|---|
| State representation | `(4, 9, 9)` tensor — 4 stacked 2D channels | CNN-friendly; encodes spatial structure |
| Action space | 137 discrete actions (8 pawn + 128 wall + 1 pass) | Covers all legal Quoridor moves including diagonal jumps |
| Invalid action handling | **Action masking** — zero out logits/Q-values before softmax/argmax | Never sample illegal moves |
| Perspective | **Current-player normalization** — board always from active player's POV | Simplifies learning; avoids perspective confusion |
| Backbone | **CNN with late fusion** for scalar features (wall counts, etc.) | Translational equivariance for spatial patterns |
| Initial opponent | **Heuristic agent** — train against existing bot before self-play | Avoids non-stationarity of early self-play |

The 4 state channels are:
- `ch0`: Current player pawn position
- `ch1`: Opponent pawn position
- `ch2`: Horizontal walls
- `ch3`: Vertical walls

---

## Coding Standards

- **Comments:** Explain *why*, not *what*. Non-obvious RL design choices must be commented inline.
- **No magic numbers:** Name constants (`BOARD_SIZE = 9`, `NUM_ACTIONS = 137`, etc.)
- **Type hints:** All function signatures must have type hints.
- **Modularity:** Keep environment, model, training loop, and replay buffer in separate files.
- **Reproducibility:** Always seed PyTorch, NumPy, and the environment. Log the seed.
- **Tensor shapes:** Comment tensor shapes at non-obvious points, e.g. `# (B, 4, 9, 9)`

---

## RL-Specific Rules (Enforce These Proactively)

- **Never train on illegal actions.** Action masking must be applied before every policy query.
- **Target network updates:** Use periodic hard updates (not soft) for DQN phase. Frequency is a hyperparameter.
- **Reward normalization:** Rewards are sparse (+1 win, -1 loss, 0 otherwise) unless explicitly changed.
- **No data leakage:** Future observations must never appear in past transitions.
- **Replay buffer:** Stores `(state, action, reward, next_state, done, legal_mask)`. The legal mask for `next_state` is required for Double DQN's action selection step.
- **Gradient clipping:** Always clip gradients (norm = 10.0) to prevent exploding gradients.

---

## Project Structure

```
QuoridorAI/
├── CLAUDE.md                    ← you are here
├── README.md
├── .gitignore
├── pyproject.toml               ← project dependencies and config
│
├── agents/                      ← RL agent implementations
│   ├── __init__.py
│   └── bot.py                   ← current agent / heuristic bot
│
├── quoridor/                    ← game engine (environment)
│   ├── __init__.py
│   ├── game.py                  ← core game logic, rules, wall validation
│   └── display.py               ← board rendering / visualization
│
├── scripts/                     ← runnable entry points
│   ├── play.py                  ← play a game interactively
│   └── play_bot.py              ← run bot vs bot or human vs bot
│
├── plan/                        ← project planning and team docs
│   ├── member_tracker.md        ← team roles and progress tracking
│   └── research.md              ← research notes and paper summaries
│
└── Research/                    ← design documentation
    └── Model_Design.md          ← state/action space and architecture decisions
```

Key entry points:
- **Game logic lives in** `quoridor/game.py` — touch this carefully, rules must stay correct
- **Agent logic lives in** `agents/bot.py` — this is where the DRL agent will be built out
- **Run experiments via** `scripts/play_bot.py`

For detailed context, reference explicitly: `@Research/Model_Design.md`, `@plan/research.md`

---

## Key Quoridor Properties — Always Keep In Mind

1. **Large, mixed action space:** 8 pawn moves + up to 128 wall placements. Most walls are illegal at any given state. Action masking is not optional.
2. **Two-player adversarial:** The opponent's policy determines the distribution of states we train on. Self-play introduces non-stationarity — be careful.
3. **Long-horizon planning:** Winning requires planning 20–40 moves ahead. Sparse rewards + long episodes = difficult credit assignment. This is why value functions matter.
4. **Wall symmetry:** Walls are shared across both players. State representation must encode walls neutrally (not "my walls" / "their walls").
5. **Pathfinding constraint:** Wall placements that completely block a player's path are illegal. The environment must validate this (BFS/DFS) on every wall action.

---

## Algorithm Progression — When to Graduate

| Phase | Graduate when... |
|---|---|
| DQN vs heuristic | Win rate > 80% consistently; training is stable; loss curve is smooth |
| PPO self-play | Agent beats DQN agent reliably; understand why PPO is more stable |
| AlphaZero | Team has solid grasp of MCTS; PPO agent is a strong baseline to beat |

Do not rush to the next phase. Understanding the current phase deeply is the goal.

---

## Team Context

- All collaborators are beginners in deep RL (solid Python + ML background).
- Prefer explanatory comments over terse code.
- When creating new modules, add a module-level docstring explaining its role in the system.
- When in doubt, optimize for correctness and clarity, not speed.
