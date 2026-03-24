# DQN Phase: Decision Log

This document records every concrete decision made for the DQN implementation phase,
the options that were considered, and the reasoning behind each choice.
Read this before touching any DQN code.

For deeper theoretical background on the algorithms themselves, see `research.md`.
For the implementation roadmap, see `plan.md`.

---

## Table of Contents

1. [Action Space Encoding](#1-action-space-encoding)
2. [CNN Architecture](#2-cnn-architecture)
3. [Feature Fusion Strategy](#3-feature-fusion-strategy)
4. [Q-Network Output Head](#4-q-network-output-head)
5. [DQN Variant](#5-dqn-variant)
6. [Training Opponent](#6-training-opponent)
7. [Evaluation Strategy](#7-evaluation-strategy)
8. [Hardware and Scale Constraints](#8-hardware-and-scale-constraints)
9. [Experiment Tracking](#9-experiment-tracking)
10. [Configuration Management](#10-configuration-management)
11. [Code Ownership and Handoff](#11-code-ownership-and-handoff)
12. [Pass Action](#12-pass-action)

---

## 1. Action Space Encoding

**Decision:** Action indices are assigned in this order (137 total):

| Index     | Action                                               |
|-----------|------------------------------------------------------|
| 0         | Move up         — delta (-1,  0)                     |
| 1         | Move down       — delta (+1,  0)                     |
| 2         | Move left       — delta ( 0, -1)                     |
| 3         | Move right      — delta ( 0, +1)                     |
| 4         | Diagonal NW jump — delta (-1, -1)                    |
| 5         | Diagonal NE jump — delta (-1, +1)                    |
| 6         | Diagonal SW jump — delta (+1, -1)                    |
| 7         | Diagonal SE jump — delta (+1, +1)                    |
| 8 – 71    | Horizontal walls, row-major: `8 + row*8 + col`       |
| 72 – 135  | Vertical walls, row-major: `72 + row*8 + col`        |
| 136       | Pass (reserved, never legal)                         |

**Why moves first (0–7):**
Debuggability. When inspecting Q-values or a replay buffer sample, "action 0 means move up"
is immediately readable. The ordering has no effect on learning — the network treats all
137 indices symmetrically.

**Why row-major encoding for walls:**
`index = offset + row*8 + col` is the most natural 2D → 1D flattening. It is easy to
invert: `row = (index - offset) // 8`, `col = (index - offset) % 8`. Consistent with
how numpy lays out 2D arrays in memory.

**Why this matters:**
Every component in the system — the legal mask, the replay buffer, the network output,
the action selection logic — must use the same index convention. This file defines that
convention. Do not redefine it anywhere else.

See also: **Decision 13** (8-pawn-action expansion) for why we have 8 move actions.

---

## 2. CNN Architecture

**Decision:** 2 convolutional layers, small filter counts, CPU-optimized.

```
Conv2d(4,  32, kernel=3, padding=1) → BatchNorm2d → ReLU
Conv2d(32, 64, kernel=3, padding=1) → BatchNorm2d → ReLU
Flatten
[late fusion with scalars]
Linear(5186, 256) → ReLU
Linear(256, 137)
```

**Options considered:**

| Option | Reason rejected |
|---|---|
| ResNet-style residual blocks | Adds complexity before we know if basic DQN even converges. Harder to debug. |
| 4+ conv layers | Risk of overfitting on sparse reward signal. Slower on CPU. |
| Transformer backbone | Powerful but training-sensitive. Inappropriate for first RL training loop. |

**Why this architecture:**
Padding=1 with kernel=3 preserves the 9×9 spatial dimensions through both conv layers.
This means spatial structure is intact when we flatten — the network can use position
information. Small filter counts (32, 64) keep forward/backward pass fast enough on CPU
to reach meaningful environment steps within a 1-hour training run.

**When to revisit:**
If the agent shows no learning signal after ~500k steps, consider adding a third conv layer
or increasing filter counts before concluding the algorithm is at fault.

---

## 3. Feature Fusion Strategy

**Decision:** Late fusion. Scalar features (wall counts) are concatenated to the flattened
CNN output, not injected as spatial channels.

**Why not early fusion (extra spatial channels):**
Broadcasting a scalar like `walls_left = 7` into a full 9×9 grid is semantically wasteful.
It implies the scalar has spatial meaning at every cell, which it does not. The late fusion
approach concatenates the scalar at the decision-making layer, where it belongs — at the
point where the network weighs strategic options.

**Why not a separate MLP stream for scalars:**
Unnecessary complexity for two scalar inputs. Late concat is simpler and equally expressive.

---

## 4. Q-Network Output Head

**Decision:** Single output head producing 137 Q-values. Legal mask applied before returning.

**Why not separate heads for moves vs. walls:**
The 4 move actions and 128 wall actions are not structurally different to the network.
Splitting them into separate heads would require combining two loss signals and tuning their
relative weights. No benefit for vanilla DQN.

**Why not Dueling architecture (value + advantage streams):**
Dueling networks help in large action spaces, but add implementation complexity.
The correct order is: get vanilla DQN working → evaluate whether instability is an issue
→ only then consider dueling. Adding complexity before confirming a problem exists is
premature.

**Masking implementation:**
```python
q_values[~legal_mask] = -1e9  # before argmax or softmax
```
This ensures illegal actions have effectively zero probability and are never selected.
Apply this masking inside the network's `forward()` call so it is impossible to forget.

---

## 5. DQN Variant

**Decision:** ~~Vanilla DQN~~ → **Double DQN** (upgraded after first training run).

**Why vanilla first:**
The team's primary goal in this phase is understanding how DQN works, not maximizing
performance. Each extension solves a specific identified problem. We should encounter
those problems first, understand them, and then apply the fix.

**Why Double DQN now:**
The first training run (25k steps) produced Q-values peaking at ~1.5 — exceeding the
theoretical maximum of 1.0. This is the textbook signature of vanilla DQN's
maximization bias: using the same network to both *select* and *evaluate* the best
next action inflates Q-value estimates. Double DQN decouples these two operations
(online net selects, target net evaluates), eliminating the bias in 3 lines of code.

**Extension path going forward:**
```
Double DQN  ✅ active
  --> if convergence is unstable: add Dueling architecture
  --> if sample efficiency is the bottleneck: add Prioritized Experience Replay
```

**Replay buffer note:**
Even though we start with vanilla DQN, the replay buffer must store `next_legal_mask`
alongside each transition. This is required for Double DQN's action selection step
and is cheap to include now. Retrofitting it later would invalidate existing buffer contents.

---

## 6. Training Opponent

**Decision:** Always train against the fixed HeuristicBot.

**Why not self-play from the start:**
Self-play introduces non-stationarity: experiences collected early (against a weak agent)
become misleading when replayed later (against a stronger version). For DQN, which relies
on a stable target distribution, this can prevent convergence entirely.

**Why not a random agent:**
A random agent is too weak. The agent could learn to win against random play without
developing any real Quoridor strategy, making the transition to self-play difficult.
The HeuristicBot plays competently and produces a meaningful training signal.

**Graduation trigger:**
Consider introducing self-play when win rate vs HeuristicBot reaches ~70%+. At that
point, the agent has a strong enough policy to generate useful self-play experience.

---

## 7. Evaluation Strategy

**Decision:** Win rate vs HeuristicBot, evaluated every 1,000 training steps over 50 episodes.

**Why win rate over loss curve:**
Loss can decrease while the agent plays worse (e.g., Q-values diverging toward a wrong
but internally consistent solution). Win rate is the ground truth metric.

**Why also track episode length:**
Shorter winning episodes suggest more decisive, efficient play. An agent winning in 30 moves
is stronger than one winning in 55. Tracking both metrics together gives a fuller picture.

**Graduation criterion:**
Win rate > 80% consistently across at least 3 consecutive evaluation checkpoints.

---

## 8. Hardware and Scale Constraints

**Decision:** Target training runs under 1 hour on CPU only. All model and buffer sizes
are chosen to fit this constraint.

**Implications:**

| Parameter | GPU-typical | Our CPU target | Reason for reduction |
|---|---|---|---|
| Replay buffer size | 100,000+ | 50,000 | Memory pressure |
| Batch size | 128–256 | 64 | Forward/backward pass time |
| Network depth | 3–5 layers | 2 conv layers | Iteration speed |
| Total env steps | 1M–10M | ~500k | 1-hour budget |

**What this means for learning:**
Fewer environment steps and smaller buffer = fewer unique experiences = harder credit
assignment. If the agent does not learn within 500k steps, the first thing to try is
a longer run (2–3 hours) before changing the algorithm. Do not prematurely conclude
the architecture is wrong.

---

## 9. Experiment Tracking

**Decision:** Weights & Biases (W&B) for all training metrics.

**What to log:**
- `train/loss` — TD loss per update step
- `train/epsilon` — exploration rate over time
- `train/mean_q_value` — mean Q-value on sampled batches (watch for divergence)
- `train/grad_norm` — gradient norm before clipping (watch for spikes)
- `eval/win_rate` — win rate vs HeuristicBot
- `eval/mean_episode_length` — average episode length during evaluation

**Why mean Q-value matters:**
Unbounded Q-value growth is an early warning sign of instability. Log it from the start.
If Q-values grow past ~10 and keep climbing, something is wrong before the win rate
reflects it.

---

## 10. Configuration Management

**Decision:** Hardcoded named constants in `config.py`. No YAML, no CLI args, no sweep
config for the initial implementation.

**Why:**
The team is in learning mode. When reading someone else's training script, you want to
immediately see what hyperparameters were used without tracing through argparse or
loading an external file. `config.py` is version-controlled, diffable, and trivial to
reason about.

**When to upgrade:**
Once basic training works and the team wants to run hyperparameter sweeps, the constants
in `config.py` can be promoted to a W&B sweep config with minimal refactoring.

---

## 11. Code Ownership and Handoff

**Decision:** Sequential ownership. One team member builds as much as possible,
documents thoroughly, then passes to the next person.

**Handoff protocol:**
1. Push to a feature branch and open a PR
2. PR description covers: what was built, what works, what is left, open questions
3. Update `plan/member_tracker.md` with name, completed work, and next steps
4. Leave a `# HANDOFF:` comment at the top of any file left mid-implementation

**Why thorough documentation matters here:**
The person picking up the work may not have been present for the decisions made during
implementation. Comments explaining *why* a choice was made (not just what the code does)
are what make a successful handoff. Code explains the what. Comments explain the why.

---

## 12. Pass Action

**Decision:** Index 136 is reserved for a pass action. It is always masked as illegal.

**Why include it at all:**
The CLAUDE.md spec defines 137 total actions. Reserving the slot costs nothing and
keeps the action space consistent with all documentation. If future game variants or
debugging scenarios require a pass move, it can be enabled by flipping one mask value.

**Why always illegal:**
Standard Quoridor has no pass move. An agent that learns to "pass" has found a degenerate
strategy. Keep it masked out at all times during DQN training.

---

---

## 13. Pawn Action Count Expansion (4 → 8)

**Decision:** Expand pawn move actions from 4 (cardinal only) to 8 (cardinal + 4 diagonal),
bringing the total action count from 133 to **137**.

**Why 4 is insufficient:**
`get_valid_moves()` can legally return diagonal destinations. This happens during jump
situations: when two pawns are adjacent, the current player may jump over the opponent;
if a wall blocks the straight jump, diagonal moves to either side become legal. These
diagonal destinations cannot be encoded by any combination of the four cardinal direction
indices. An agent that only has indices 0–3 would have no way to select a legal diagonal
jump, effectively making those positions unreachable.

**Why encode direction deltas, not absolute destinations:**
A single "move up" index (delta (-1, 0)) covers both a normal 1-step advance and a 2-step
straight jump over an adjacent opponent — both share the same direction sign. Using deltas
keeps the pawn action count small (8) and delegates the disambiguation to the environment
wrapper, which scans `get_valid_moves()` for any legal destination in the given direction.

**Impact on other decisions:**
- Total action count: 133 → **137**
- Wall offsets shift: H_WALL_OFFSET 4 → **8**, V_WALL_OFFSET 68 → **72**, PASS_ACTION 132 → **136**
- CNN output head: `Linear(256, 137)` → `Linear(256, 137)`
- CLAUDE.md updated accordingly

*Decision recorded: March 17, 2026*

---

*Decisions recorded: March 17, 2026*
*Based on team interview and code review session*

---

## 14. Extra Constants Added to config.py

**Decision:** Added `NUM_CHANNELS = 4` and `NUM_SCALARS = 2` beyond the original spec.

**Why:**
`dqn_model.py` needs these values to define its input layer sizes (`Conv2d(NUM_CHANNELS, ...)` and
`_FC_IN_SIZE = CONV2_FILTERS * BOARD_SIZE * BOARD_SIZE + NUM_SCALARS`). Hardcoding `4` and `2`
directly in the model would be magic numbers. Keeping them in `config.py` ensures they stay in sync
with the state representation documented in CLAUDE.md and `get_observation()`.

*Decision recorded: March 23, 2026*

---

## 15. PyTorch Upgrade Required (2.1.2 → 2.11.0)

**Decision:** Upgraded PyTorch from 2.1.2 to 2.11.0 to fix a NumPy compatibility issue.

**What happened:**
The original environment had PyTorch 2.1.2 (compiled against NumPy 1.x) alongside NumPy 2.2.6.
Both `torch.from_numpy(array)` and `torch.tensor(numpy_array)` raised `RuntimeError` at runtime
because PyTorch could not read NumPy 2.x dtype objects.

**Fix:**
```
pip install --upgrade torch
```
Installed PyTorch 2.11.0 which supports NumPy 2.x.

**Side effect:**
`torchvision 0.16.2` now reports a version conflict (it requires torch==2.1.2). This is harmless
because torchvision is not used anywhere in this project.

**What to watch for:**
If a future team member installs torchvision for any reason, they should upgrade it to a version
compatible with torch 2.11.0 at the same time.

*Decision recorded: March 23, 2026*
