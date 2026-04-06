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

**Phase:** PPO training — actively running. All infrastructure complete; first serious run in progress.
**Branch:** `agent_jeff`
**Best checkpoint:** `checkpoints/bc_pretrained_bfs_resnet.pt` — BC warm-start for next/current run
**Experiment tracking:** W&B / TensorBoard — actively used

### Performance
| Agent | Opponent | Win Rate | Notes |
|---|---|---|---|
| DQN | Random bot | ~100% | Consistently wins |
| DQN | Heuristic bot | **0%** | Abandoned — fundamental ceiling hit |
| PPO (500k steps) | Heuristic bot | **0%** | Diagnosed: actor_head bug + entropy collapse |
| PPO (fresh restart) | Heuristic bot | Not yet measured | Bugs fixed; first real run underway |

### PPO Model Variants (ready to run)
All four variants are drop-in replaceable via `--model` flag in `train_ppo.py`:

| Flag | File | Channels | Backbone | Notes |
|---|---|---|---|---|
| `baseline` | `ppo_model.py` | 4 | 2-conv + BatchNorm | Original — use as control |
| `resnet` | `ppo_model_resnet.py` | 4 | ResNet + LayerNorm | Residual blocks, no BFS |
| `bfs` | `ppo_model_bfs.py` | 6 | 2-conv + LayerNorm | BFS distance maps, no residual |
| `bfs_resnet` | `ppo_model_bfs_resnet.py` | 6 | ResNet + LayerNorm | **Full combo — primary model** |

**Auxiliary prediction head (implemented, bfs_resnet only):**
`ppo_model_bfs_resnet.py` has a third output from `forward()`: `aux_pred (B, 2)` predicting
`[my_dist_to_goal, opp_dist_to_goal]`. Targets are derived for free by dot-producting the BFS
channel maps (ch4, ch5) with the one-hot pawn channels (ch0, ch1). Adds `AUX_LOSS_COEF * MSE`
to the total loss. W&B key: `train/aux_loss` — should decrease quickly; if it stays high, the
backbone is not learning path distances.
Non-BFS models return 2 values from `forward()`; `_model_forward()` in `train_ppo.py` pads with
`None` so all call sites unpack 3 values uniformly.

### Recommended training command
```bash
python -m scripts.train_ppo \
    --model bfs_resnet \
    --opponent pool \
    --reward-shaping \
    --checkpoint checkpoints/bc_pretrained_bfs_resnet.pt \
    --value-burnin-steps 10000
```

### Known Issues — Prioritized
1. **0% DQN win rate (root cause: state representation)** — The 4-channel state gives no direct
   signal about path lengths. The model had to infer from wall positions alone whether a wall was
   strategically valuable — effectively impossible with sparse +1/-1 rewards over 30-40 move episodes.
   PPO + BFS channels directly address this.
2. **Wall wastage** — DQN agent placed walls that don't meaningfully block the opponent. Expected to
   improve with BFS channels (model can directly see path-length effects of wall placement).
3. **BC actor_head was silently discarded on load (NOW FIXED)** — The checkpoint loader
   previously excluded `actor_head.*` from loading, meaning the BC-trained policy head was
   thrown away and randomly reinitialized every time training started. The backbone loaded but
   the policy started random → 0% eval win rate throughout the entire 500k PPO run.
   Fixed: `_HEAD_PREFIXES` now only excludes `value_head.*`, `aux_head.*`, `fc.*`. For a PPO
   BC checkpoint, `actor_head.*` is preserved. For a DQN checkpoint, `actor_head.*` is absent
   so strict=False handles it as missing (randomly initialized) — no regression.
   Verified: `bc_pretrained_bfs_resnet.pt` now loads 19/19 tensors including actor_head.
4. **PPO entropy collapsed to 0.1–0.3 nats (NOW FIXED via config)** — After 500k steps, the
   policy was near-deterministic (healthy range: 1–3 nats). Root cause: entropy coef of 0.05
   was too weak when the policy found a deterministic strategy that won 40% of training games
   by exploiting epsilon randomness. Fixed: `PPO_ENTROPY_COEF_START = 0.2`.
5. **OpponentPool not compatible with --num-envs > 1** — `OpponentPool._current` is shared
   instance state; interleaved resets from N envs clobber each other's sampled opponent.
   Workaround: always use `--num-envs 1` (the default) when `--opponent pool`.
6. **Slow training throughput** — Episode speed is a bottleneck. Profile before optimizing.

### Behavioral Cloning Status
- `data/bc_data.npz` — 4-channel dataset (exists, untracked)
- `data/bc_data_bfs.npz` — 6-channel dataset for BFS models (exists, untracked)
- `checkpoints/bc_pretrained.pt` — DQN checkpoint with `conv.*` keys (incompatible with bfs_resnet)
- `checkpoints/bc_pretrained_bfs_resnet.pt` — **exists and verified** (19 keys: backbone + actor_head)
- **Checkpoint loader** excludes `value_head.*`, `aux_head.*`, `fc.*` only — actor_head is preserved for PPO BC checkpoints

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
- **Entropy scheduling is implemented.** `PPO_ENTROPY_COEF_START=0.2` decays to
  `PPO_ENTROPY_COEF_END=0.01` linearly over training. Start high — previous value of 0.05 was
  insufficient to prevent collapse to 0.1–0.3 nats. Healthy range mid-training: 1–3 nats.
  Monitor `train/entropy`; if it stays below 0.5 the policy has collapsed regardless of win rate.
- **Advantage normalization is per-rollout, not per-minibatch.** Normalized once in `main()` after
  `compute_gae()`. With sparse ±1 rewards, per-minibatch normalization amplifies noise from
  near-constant advantage mini-batches into fake gradient signal.
- **Per-action-type entropy is logged every update.** W&B keys: `train/move_prob_mass`,
  `train/wall_prob_mass`, `train/move_entropy`, `train/wall_entropy`. `wall_prob_mass < 0.1`
  is the early warning for wall-action collapse — check this before looking at overall entropy.
- **Auxiliary path-length head is active for bfs_resnet.** `forward()` returns 3 values
  `(dist, value, aux_pred)`; non-BFS models return 2. `_model_forward()` in `train_ppo.py`
  normalises this to always 3. `train/aux_loss` in W&B — should trend down quickly.
- **Do NOT clip value loss.** Use simple MSE + global gradient clipping. Evidence shows value
  loss clipping hurts with sparse rewards.

### PPO — Planned (not yet implemented, see `plan/ppo_techniques.md`)
- **γ=0.999, λ=0.97** — longer effective horizon (~333 steps vs current ~100). Implement after
  the agent can win occasionally — longer horizon doesn't help if there's no winning signal yet.
- **PBRS reward shaping** — Φ(s) = α × (opp_path − my_path). Policy-invariant only in potential-
  difference form. Both potentials must use the acting player's perspective.
- **Value head burn-in** — **implemented.** Pass `--value-burnin-steps N --checkpoint path` to
  freeze `actor_head.*` params for N env steps before PPO starts. The frozen BC actor generates
  ~20% win-rate trajectories; only backbone + value head get gradients. After N steps, actor
  unfreezes and full PPO resumes. W&B key: `burnin/value_loss` (tracks value head convergence
  during burn-in). Suggested N: 10000. Only activates when `--checkpoint` is also provided.

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
│   ├── env.py                         ← Gym-compatible wrapper; use_bfs + use_reward_shaping flags
│   ├── vec_env.py                     ← synchronous N-env wrapper for parallel rollout collection
│   ├── action_encoding.py             ← maps action indices ↔ game moves
│   └── display.py                     ← board rendering / visualization
│
├── agents/                            ← agent implementations
│   ├── __init__.py
│   ├── bot.py                         ← HeuristicBot + EpsilonHeuristicBot (ε-greedy curriculum)
│   ├── dqn_model.py                   ← CNN Q-network (DQN, 4-channel)
│   ├── dqn_bot.py                     ← DQN agent wrapper
│   ├── replay_buffer.py               ← experience replay buffer
│   ├── random_bot.py                  ← random baseline agent
│   ├── ppo_bot.py                     ← PPO checkpoint wrapper (for watch scripts / evaluation)
│   ├── ppo_model.py                   ← PPO baseline: 4-ch CNN + BatchNorm
│   ├── ppo_model_resnet.py            ← PPO resnet: 4-ch ResNet + LayerNorm
│   ├── ppo_model_bfs.py               ← PPO bfs: 6-ch CNN + LayerNorm (BFS channels)
│   └── ppo_model_bfs_resnet.py        ← PPO bfs_resnet: 6-ch ResNet + LayerNorm + aux head (primary)
│
├── scripts/                           ← runnable entry points
│   ├── train_dqn.py                   ← DQN training loop (deprecated — moved to PPO)
│   ├── train_ppo.py                   ← PPO training loop; --model selects architecture
│   ├── watch_ppo.py                   ← terminal viewer: watch a PPO checkpoint play live
│   ├── generate_bc_data.py            ← generate BC expert data (--bfs flag for 6-channel)
│   ├── pretrain_bc.py                 ← BC pretraining; --model flag for all variants
│   ├── play.py                        ← play interactively
│   └── play_bot.py                    ← bot vs bot / human vs bot evaluation
│
├── checkpoints/                       ← saved model weights (untracked)
│   ├── bc_pretrained.pt               ← 4-ch BC checkpoint; usable for baseline/resnet PPO only
│   ├── bc_pretrained_bfs_resnet.pt    ← 6-ch BC checkpoint; 19 keys incl. actor_head — use this
│   ├── best.pt                        ← best DQN checkpoint during training
│   └── final_500000.pt                ← final DQN checkpoint (500k episodes, 0% vs heuristic)
│
├── data/
│   ├── bc_data.npz                    ← 4-channel BC dataset (untracked)
│   └── bc_data_bfs.npz               ← 6-channel BC dataset (untracked)
│
├── tests/                             ← pytest suite (136 passing, 2 pre-existing DQN failures)
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
- **Environment wrapper:** `quoridor/env.py` — `QuoridorEnv(use_bfs=True/False, use_reward_shaping=True/False)`
- **Vectorized env:** `quoridor/vec_env.py` — `VecQuoridorEnv(n_envs, bot, use_bfs, use_reward_shaping)`
- **PPO training:** `scripts/train_ppo.py --model {baseline,resnet,bfs,bfs_resnet}`
- **Model + env factory:** `train_ppo.py:build_model_and_env()` — always use this, never pair model and env manually
- **Watch agent play:** `scripts/watch_ppo.py --checkpoint <path> --model bfs_resnet`
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
