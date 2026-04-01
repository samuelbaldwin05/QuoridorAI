# Quoridor Deep Reinforcement Learning — Project Memory

## Project Overview

We are building a deep RL agent to play Quoridor from scratch, progressing through three algorithmic phases:
1. **DQN** — foundational concept implementation
2. **PPO** — stable adversarial/self-play training (**current phase**)
3. **AlphaZero** — MCTS + neural network, the end goal

This is both a learning project (beginner-level deep RL) and a serious performance goal. Code must be
readable, well-commented, and correct. Clarity is never sacrificed for cleverness.

**Solo developer.** Jeffrey is the only person writing code. All collaborators reviewing the project are
beginners in deep RL with solid Python + ML backgrounds — keep explanations accessible.

---

## Current Status

> ⚠️ **Keep this section up to date.** Stale information here is worse than no information — it will
> actively mislead future sessions. Update performance numbers, known issues, and the best checkpoint
> whenever they change.

**Phase:** Transitioning to PPO — DQN was abandoned after 500k episodes at 0% win rate
**Branch:** `agent_jeff`
**Best checkpoint:** `final_500000.pt` (DQN, 500k episodes — not used for PPO)
**Experiment tracking:** W&B / TensorBoard — actively used

### Performance
| Agent | Opponent | Win Rate | Notes |
|---|---|---|---|
| DQN | Random bot | ~100% | Consistently wins |
| DQN | Heuristic bot | **0%** | Abandoned — fundamental ceiling hit |
| PPO | Heuristic bot | Not yet trained | Four model variants ready to experiment |

### PPO Model Variants (ready to run)
All four variants are drop-in replaceable via `--model` flag in `train_ppo.py`:

| Flag | File | Channels | Backbone | Notes |
|---|---|---|---|---|
| `baseline` | `ppo_model.py` | 4 | 2-conv + BatchNorm | Original — use as control |
| `resnet` | `ppo_model_resnet.py` | 4 | ResNet + LayerNorm | Residual blocks, no BFS |
| `bfs` | `ppo_model_bfs.py` | 6 | 2-conv + LayerNorm | BFS distance maps, no residual |
| `bfs_resnet` | `ppo_model_bfs_resnet.py` | 6 | ResNet + LayerNorm | Full combo — start here |

Run experiments: `python -m scripts.train_ppo --model bfs_resnet --no-wandb`

### Known Issues — Prioritized
1. **0% DQN win rate (root cause: state representation)** — The 4-channel state gives no direct
   signal about path lengths. The model had to infer from wall positions alone whether a wall was
   strategically valuable — effectively impossible with sparse +1/-1 rewards over 30-40 move episodes.
   PPO + BFS channels directly address this.
2. **Wall wastage** — DQN agent placed walls that don't meaningfully block the opponent. Expected to
   improve with BFS channels (model can directly see path-length effects of wall placement).
3. **BC warm-start incompatibility with BFS models** — `bc_pretrained.pt` was generated from
   4-channel observations. Loading it into `ppo_model_bfs.py` or `ppo_model_bfs_resnet.py` with
   `strict=False` silently random-initializes the first conv layer (shape mismatch: 4-ch vs 6-ch).
   The rest of the backbone loads correctly. To properly warm-start BFS models, regenerate BC data
   with `use_bfs=True` in `generate_bc_data.py` and retrain. Not done yet.
4. **BC abandonment undocumented** — The DQN BC pretraining was abandoned but the reason was never
   recorded. Log it in `plan/decisions.md` before it's forgotten.
5. **Slow training throughput** — Episode speed is a bottleneck. Profile before optimizing.

### Behavioral Cloning Status
- `data/bc_data.npz` — 4-channel dataset from HeuristicBot self-play (exists, untracked)
- `checkpoints/bc_pretrained.pt` — 4-channel DQN backbone checkpoint (exists, not actively used)
- Scripts exist and work: `generate_bc_data.py`, `pretrain_bc.py`
- **BC → PPO warm-start:** only safe for `baseline` and `resnet` variants (4-channel match)
- **BC → BFS PPO warm-start:** requires regenerating dataset with `use_bfs=True` (not done)

---

## Stack

- **Language:** Python 3.10+
- **Framework:** PyTorch (not TensorFlow, not JAX)
- **Environment:** Custom Quoridor env built from scratch (Gym-compatible interface)
- **Experiment tracking:** W&B / TensorBoard (active)
- **Config:** `config.py` is the **single source of truth** for all hyperparameters
- **Version control:** Git / GitHub — branch `agent_jeff`

### ⚠️ Config Drift — Known Tech Debt
`config.py` is the intended single source of truth, but two files define constants independently:

- **`quoridor/action_encoding.py:72-77`** — defines `NUM_ACTIONS = 137`, `MOVE_ACTION_COUNT = 8`,
  `FENCE_GRID = 8`, etc. locally instead of importing from config. **This is the highest-risk drift
  point** — if `NUM_ACTIONS` is changed in config, action encoding won't pick it up.
- **`quoridor/game.py:5`** — defines `BOARD_SIZE = 9` locally (lower risk, unlikely to change).
- **`quoridor/game.py:24`** — `walls_left = np.array([10, 10])` is hardcoded with no
  `INITIAL_WALLS_PER_PLAYER` constant. Same value appears as a magic `10.0` divisor for wall
  normalization. **This divisor directly affects the observation fed to the network** — if
  initial wall count ever changes, the normalization silently breaks without a config mismatch error.

Training scripts, agents, and model files all import correctly from config.
**Action item:** Consolidate these before any experiment that touches action space or board dimensions.

---

## Architecture Decisions (Finalized — Do Not Revisit Without Discussion)

| Decision | Choice | Rationale |
|---|---|---|
| State representation | `(4, 9, 9)` tensor — 4 base channels; `(6, 9, 9)` with BFS | CNN-friendly; BFS variant adds direct path-length signal |
| Action space | 137 discrete actions (8 pawn + 128 wall + 1 pass) | Covers all legal Quoridor moves including diagonal jumps |
| Invalid action handling | **Action masking** — zero out logits/Q-values before softmax/argmax | Never sample illegal moves |
| Perspective | **Current-player normalization** — board always from active player's POV | Simplifies learning; avoids perspective confusion |
| Backbone | **CNN with late fusion** for scalar features; **ResNet variant** available | Translational equivariance; residual blocks improve gradient flow |
| Normalization | **LayerNorm** in new PPO models (not BatchNorm) | Consistent train/eval behaviour; PPO switches modes constantly |
| Initial opponent | **Heuristic agent** — train against existing bot before self-play | Avoids non-stationarity of early self-play |

The base 4 state channels are:
- `ch0`: Current player pawn position
- `ch1`: Opponent pawn position
- `ch2`: Horizontal walls
- `ch3`: Vertical walls

BFS variant adds (when `QuoridorEnv(use_bfs=True)`):
- `ch4`: Current player BFS distance map — each cell = normalised steps to goal
- `ch5`: Opponent BFS distance map

---

## Coding Standards

- **Comments:** Explain *why*, not *what*. Non-obvious RL design choices must be commented inline.
- **No magic numbers:** Name constants (`BOARD_SIZE = 9`, `NUM_ACTIONS = 137`, etc.). All hyperparameters live in `config.py`. **Never redefine a constant locally** — always `from config import X`. If you see a locally-defined constant that exists in config, flag it.
- **Type hints:** All function signatures must have type hints.
- **Modularity:** Environment, model, training loop, and replay buffer live in separate files.
- **Reproducibility:** Always seed PyTorch, NumPy, and the environment. Log the seed.
- **Tensor shapes:** Comment tensor shapes at non-obvious points, e.g. `# (B, 4, 9, 9)`

---

## RL-Specific Rules (Enforce These Proactively)

### General (All Phases)
- **Never train on illegal actions.** Action masking must be applied before every policy query.
- **No data leakage:** Future observations must never appear in past transitions.
- **Gradient clipping:** Always clip gradients. Current PPO uses `PPO_GRAD_CLIP = 10.0`.
  A comparison run at 0.5 (tighter, more standard for PPO) is planned but not yet run.
- **Rewards are sparse** (+1 win, -1 loss, 0 otherwise) unless PBRS shaping is enabled (TODO).
- **Model ↔ env pairing:** BFS models (`bfs`, `bfs_resnet`) **must** use `QuoridorEnv(use_bfs=True)`.
  `build_model_and_env()` in `train_ppo.py` enforces this — never construct model and env separately.

### DQN Phase (historical — abandoned)
- **Target network updates:** Periodic hard updates (not soft). Frequency is a hyperparameter.
- **Replay buffer:** Stores `(state, action, reward, next_state, done, legal_mask)`. Legal mask for
  `next_state` is required for Double DQN's action selection step.

### PPO Phase (current)
- **Entropy scheduling is implemented.** `PPO_ENTROPY_COEF_START=0.05` decays to
  `PPO_ENTROPY_COEF_END=0.005` linearly over training. Without decay, the policy either explores
  too randomly (never specializes) or collapses to movement-only (ignores 128 wall actions).
- **Advantage normalization is per-rollout, not per-minibatch.** Normalized once in `main()` after
  `compute_gae()`. With sparse ±1 rewards, per-minibatch normalization amplifies noise from
  near-constant advantage mini-batches into fake gradient signal.
- **Per-action-type entropy is logged every update.** W&B keys: `train/move_prob_mass`,
  `train/wall_prob_mass`, `train/move_entropy`, `train/wall_entropy`. `wall_prob_mass < 0.1`
  is the early warning for wall-action collapse — check this before looking at overall entropy.
- **Do NOT clip value loss.** Use simple MSE + global gradient clipping. Evidence shows value
  loss clipping hurts with sparse rewards.

### PPO — Planned (not yet implemented, see `plan/ppo_techniques.md`)
- **γ=0.999, λ=0.97** — longer effective horizon (~333 steps vs current ~100). Implement after
  the agent can win occasionally — longer horizon doesn't help if there's no winning signal yet.
- **PBRS reward shaping** — Φ(s) = α × (opp_path − my_path). Policy-invariant only in potential-
  difference form. Both potentials must use the acting player's perspective.
- **Value head burn-in** — when BC warm-starting, run supervised regression on value head before
  PPO fine-tuning begins. Skipping this can destroy the pretrained policy via garbage advantages.

---

## Project Structure

```
QuoridorAI/
├── CLAUDE.md                          ← you are here
├── config.py                          ← SINGLE SOURCE OF TRUTH for all hyperparameters
├── pyproject.toml                     ← dependencies
│
├── quoridor/                          ← game engine
│   ├── __init__.py
│   ├── game.py                        ← core rules, wall validation (touch carefully)
│   ├── env.py                         ← Gym-compatible wrapper; use_bfs flag supported
│   ├── action_encoding.py             ← maps action indices ↔ game moves
│   └── display.py                     ← board rendering / visualization
│
├── agents/                            ← agent implementations
│   ├── __init__.py
│   ├── bot.py                         ← heuristic bot (training opponent)
│   ├── dqn_model.py                   ← CNN Q-network (DQN, 4-channel)
│   ├── dqn_bot.py                     ← DQN agent wrapper
│   ├── replay_buffer.py               ← experience replay buffer
│   ├── random_bot.py                  ← random baseline agent
│   ├── ppo_model.py                   ← PPO baseline: 4-ch CNN + BatchNorm
│   ├── ppo_model_resnet.py            ← PPO resnet: 4-ch ResNet + LayerNorm
│   ├── ppo_model_bfs.py               ← PPO bfs: 6-ch CNN + LayerNorm (BFS channels)
│   └── ppo_model_bfs_resnet.py        ← PPO bfs_resnet: 6-ch ResNet + LayerNorm (recommended)
│
├── scripts/                           ← runnable entry points
│   ├── train_dqn.py                   ← DQN training loop (deprecated — moved to PPO)
│   ├── train_ppo.py                   ← PPO training loop; --model selects architecture
│   ├── generate_bc_data.py            ← generate BC data (4-ch only; BFS version not built yet)
│   ├── pretrain_bc.py                 ← BC pretraining for DQNModel only
│   ├── play.py                        ← play interactively
│   └── play_bot.py                    ← bot vs bot / human vs bot evaluation
│
├── checkpoints/                       ← saved model weights (untracked)
│   ├── bc_pretrained.pt               ← 4-ch BC checkpoint; usable for baseline/resnet PPO only
│   ├── best.pt                        ← best DQN checkpoint during training
│   └── final_500000.pt                ← final DQN checkpoint (500k episodes, 0% vs heuristic)
│
├── data/
│   └── bc_data.npz                    ← 4-channel BC dataset (untracked)
│
├── tests/                             ← pytest suite (61 tests, all passing)
│   ├── test_action_encoding.py
│   ├── test_dqn_model.py
│   ├── test_env.py
│   ├── test_legal_mask.py
│   ├── test_observation.py
│   └── test_replay_buffer.py
│
└── plan/                              ← project docs
    ├── plan.md
    ├── decisions.md                   ← architectural decision log
    ├── ppo_techniques.md              ← PPO implementation reference (aspirational — not all implemented)
    ├── research.md                    ← research notes and paper summaries
    ├── member_tracker.md
    └── pytorch_curriculum.md          ← PyTorch learning plan
```

### Key Entry Points
- **Game logic:** `quoridor/game.py` — touch carefully, rules must stay correct
- **BFS distance map:** `game.py:bfs_distance_map(player)` — reverse multi-source BFS, returns `(9,9)` normalised float32
- **Environment wrapper:** `quoridor/env.py` — `QuoridorEnv(use_bfs=True/False)`
- **PPO training:** `scripts/train_ppo.py --model {baseline,resnet,bfs,bfs_resnet}`
- **Model + env factory:** `train_ppo.py:build_model_and_env()` — always use this, never pair model and env manually
- **Evaluation:** `scripts/play_bot.py` — run agent vs opponents
- **Config:** `config.py` — change hyperparameters here, nowhere else

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
| DQN vs heuristic | ~~Win rate > 80% consistently~~ — abandoned; moved to PPO |
| PPO self-play | Agent beats heuristic bot reliably (>80%); understand why PPO is more stable |
| AlphaZero | Jeffrey has a solid grasp of MCTS; PPO agent is a strong baseline to beat |

Do not rush to the next phase. Understanding the current phase deeply is the goal.

---

## Debugging Checklist — When Performance Plateaus

Before adding complexity, verify fundamentals:

1. **Action masking correctness** — Are illegal actions truly never selected? Log action distributions.
2. **State encoding fidelity** — Render a few states from the replay buffer and verify they match the board. Off-by-one errors in wall encoding are subtle and fatal.
3. **Reward signal** — Is the agent actually receiving +1/-1 at episode end? Log raw rewards.
4. **Opponent characterization** — What strategy does the heuristic bot use? How many moves does it typically win in? You can't beat what you don't understand.
5. **Q-value / entropy inspection** — Is entropy collapsing early? Are wall-placement logits systematically lower than move logits? This would explain wall avoidance.
6. **BFS channel sanity** — With `use_bfs=True`, verify ch4 goal row = 0.0 and ch5 reflects opponent distance correctly. A flipped perspective would silently train on wrong signal.
7. **Approx KL in PPO** — Should stay below 0.05. Consistently higher means LR is too large or clip_eps needs reducing.
8. **Exploration schedule** — Is epsilon (DQN) or entropy bonus (PPO) keeping enough exploration, or has it decayed too far?

---

## Experiment Log Convention

When running experiments, log the following in W&B (or `plan/decisions.md` as a fallback):
- **What changed** (one variable at a time when possible)
- **Hypothesis** (why you expect this to help)
- **Result** (win rate, loss curve behavior, qualitative observations)
- **Decision** (keep, revert, or investigate further)

One variable at a time. If you change three things and performance improves, you've learned nothing.
