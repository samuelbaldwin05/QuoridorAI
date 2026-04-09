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

## 21. Frozen Self-Play as Curriculum Phase 2 (Replaces Heuristic Direct Training)

**Decision:** Curriculum phase 2 is now `--opponent self-play` (frozen DQNBot) instead of
direct training against HeuristicBot.

**Why direct heuristic training failed:**
The agent never won a single game against HeuristicBot in 131k steps. Q-values collapsed to
~-1.17 (below the theoretical minimum of -1), loss dropped to near zero, and the network
found a stable wrong fixed point: "everything loses." With no positive reward signal, there
is no gradient to escape this collapse.

**Why reward shaping also failed:**
Potential-based path differential shaping (Decision 18) was applied to non-terminal steps.
When the agent is losing every game, the path differential (D_opp - D_agent) is consistently
negative, so shaped rewards were negative on every step. Combined with n-step returns (n=5),
this amplified the all-negative signal and accelerated Q-value collapse.

**Why frozen self-play works:**
- The agent plays against a copy of itself → games are roughly balanced → ~50% win rate
- Guaranteed positive (+1) signal from the start, preventing Q-value collapse
- Teaches general Quoridor patterns (pawn racing, wall use, path optimization) rather than
  random-exploiting habits or heuristic-specific counter-strategies
- Better foundation for the PPO self-play phase

**Non-stationarity management (key DQN concern):**
True simultaneous self-play violates DQN's assumption of a stationary MDP. Instead, the
opponent is a *frozen* copy of the online network (`DQNBot`), updated every
`OPPONENT_UPDATE_FREQ = 10_000` steps. Between syncs the opponent is fixed — stable MDP.

**Evaluation is always vs HeuristicBot:**
Even during self-play training, win rate is measured against HeuristicBot every `EVAL_FREQ`
steps. The graduation criterion (>80% vs heuristic) is unchanged.

**Reward shaping removed:**
Sparse rewards (+1/-1) are restored. Shaping requires occasional wins to anchor Q-values;
without wins it only makes the all-negative collapse worse.

**Usage:**
```bash
python -m scripts.train_dqn --opponent self-play --checkpoint checkpoints/best.pt
```

*Decision recorded: March 24, 2026*
---

## 16. Epsilon Start Reduced for Heuristic Warm-Start Phase

**Decision:** `EPSILON_START_HEURISTIC` reduced from `0.99` to `0.5`.

**Why:**
The heuristic phase warm-starts from `checkpoints/best.pt` (the random-phase graduation checkpoint).
Starting at epsilon=0.99 means the agent acts almost purely randomly for the first ~30k steps,
ignoring the useful weights it just loaded. With epsilon=0.5, the learned policy is exploited
from step 1 while still allowing substantial exploration to adapt to the harder opponent distribution.

**Tradeoff considered:**
If the random-trained value function transfers poorly to heuristic games (plausible — the game
distributions are very different), higher epsilon early is less harmful. 0.5 is a middle ground:
enough exploration to adapt, not so much that the warm-start is wasted.

*Decision recorded: March 24, 2026*

---

## 17. Lower Learning Rate for Heuristic Phase

**Decision:** Added `LEARNING_RATE_HEURISTIC = 5e-5` (half of the standard `1e-4`).

**Why:**
When warm-starting, the network already has a reasonable policy. Using the full learning rate
means early noisy gradients from the new opponent distribution can overwrite learned features
before the agent has seen enough heuristic-game experience to form useful replacements. A lower
learning rate makes the update steps smaller and more conservative, preserving the warm-started
weights while still allowing adaptation.

**Implementation:** `train_dqn.py` selects the learning rate based on `--opponent`:
`LEARNING_RATE_HEURISTIC` when `opponent == "heuristic"`, `LEARNING_RATE` otherwise.

*Decision recorded: March 24, 2026*

---

## 18. Potential-Based Reward Shaping Added to env.py

**Decision:** Non-terminal transitions now return a shaped reward instead of `0.0`.

**Shaping function:**
```
F(s, s') = γ · Φ(s') - Φ(s)
where Φ(s) = (D_opponent - D_agent) / 16
```
`D_agent` and `D_opponent` are BFS shortest-path distances to each player's goal row.
Dividing by 16 normalizes the potential so per-step shaping is ~10× smaller than the ±1 terminal reward.

**Why potential-based shaping:**
Ng et al. (1999) proved that shaping of this form is policy-invariant — the optimal policy under
the shaped reward is identical to the optimal policy under the sparse reward. Raw progress bonuses
(e.g., +0.1 per row advanced) do not have this guarantee and can distort the learned policy.

**Why path differential (not self-distance only):**
`Φ(s) = -D_agent / 8` would only reward forward progress and give no signal for wall placements
that block the opponent. The path differential rewards both advancing and placing blocking walls,
which is essential against a heuristic bot that uses UCS pathfinding.

**Why not shape terminal transitions:**
Terminal rewards (+1.0 win, -1.0 loss) are kept clean so the win-rate graduation metric remains
meaningful. Shaping terminal rewards would conflate game outcome with positional advantage.

**When to remove:**
Remove shaping before any self-play phase (PPO). The policy-invariance guarantee assumes a fixed
opponent; with a non-stationary self-play opponent the guarantee no longer holds.

*Decision recorded: March 24, 2026*

---

## 19. N-Step Returns (n=5) Added to Training Loop

**Decision:** Replaced 1-step TD targets with 5-step discounted returns.

**How it works:**
Instead of storing `(s_t, a_t, r_t, s_{t+1})` in the replay buffer, a rolling deque accumulates
5 transitions. Once full, the oldest transition is stored with:
```
reward = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + γ⁴·r_{t+4}
next_state = s_{t+5}
```
At episode boundaries, remaining transitions are flushed with truncated (< 5 step) returns.

**Why n=5 for Quoridor vs heuristic:**
Heuristic-phase games are ~12 steps. With 1-step TD, the terminal ±1 reward takes 12 gradient
updates to propagate from the final step to the first. With n=5, it reaches the first step in
just 3 update passes. This is critical when the agent never wins early and the only learning
signal is the -1 loss at the end of each short episode.

**Tradeoff:**
N-step returns introduce off-policy bias when using experience replay (transitions were collected
under an older policy). In practice this bias is well-tolerated for n ≤ 5 in DQN, and the
variance reduction from faster signal propagation outweighs it for sparse-reward games.

*Decision recorded: March 24, 2026*

---

## 20. HeuristicBot UCS Performance Optimization

**Decision:** Pre-compute a 9×9 fence-distance grid once per `_ucs_path` call instead of
calling `_min_fence_distance` per neighbor expansion.

**What was slow:**
`_min_fence_distance(game, row, col)` scans all 64 fence grid positions — O(64) per call.
It was called inside `_ucs_path` for every neighbor of every expanded node: up to 81 nodes × 4
neighbors = ~324 calls per UCS search. This produced ~20k Python operations per bot move.
The bot was observed hanging during training evaluation (KeyboardInterrupt in `_min_fence_distance`)
due to the accumulated cost over thousands of evaluation games.

**Fix:**
Added `_build_fence_distance_grid(game)` which collects all fence-affected squares once in
O(FENCE_GRID² × 4) = O(256), then computes the 9×9 distance grid in O(81 × n_fence_squares).
`_ucs_path` now calls this once before the search and does O(1) grid lookups per neighbor.
Total reduction: ~O(83k) → ~O(7k) operations per UCS call (~12× improvement).

**Behavior unchanged:** `_min_fence_distance` and `_fence_penalty` are retained for any
external callers. `_ucs_path` output is identical — only the computation path differs.

*Decision recorded: March 24, 2026*
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

---

## 16. PPOBot Self-Play Flip Bug — Action Coordinates Not Unflipped

**Status:** Known bug — NOT yet fixed. Must be fixed before starting PPO self-play.

**Affects:** PPO self-play only. Does NOT affect PPO vs heuristic bot (PPOBot is not involved in that scenario).

**Where:** `scripts/train_ppo.py`, `PPOBot.choose_action()`, lines ~104–132.

**What the bug is:**

`PPOBot.choose_action()` calls `game.get_observation(use_bfs=..., flip=(game.turn==1))` — correctly receiving a flipped (P1 perspective) observation when it is P1's turn. The PPO model selects an action index, which is decoded into a direction delta `(dr, dc)` for moves or a fence row `r` for wall placements.

These decoded values are in the **flipped observation's coordinate space** — but are applied directly to the **actual (unflipped) board coordinates**. This means:

- **Pawn moves:** Action 0 encodes `dr = -1` ("move up toward goal" in flipped space). In actual coordinates, P1's goal is row 8 (increasing row), so applying `dr = -1` moves P1 *away* from its goal.
- **Fence placement:** Fence row `r` in flipped space corresponds to actual row `FENCE_GRID - 1 - r = 7 - r`. Placing the fence at actual row `r` instead of `7 - r` mirrors it to the wrong half of the board.

**Why it doesn't affect PPO vs heuristic:**
During PPO vs heuristic training, the PPO model plays as P0 (turn=0, flip=False). `PPOBot` (the frozen self-play surrogate) is never instantiated in that scenario. The heuristic bot uses raw `QuoridorState` coordinates directly.

**The fix (apply before self-play phase):**

After decoding the action index into `(dr, dc)` for moves or `(r, c, ori)` for fences, conditionally un-flip the coordinates:

```python
flip = (game.turn == 1)

# For pawn moves:
if flip:
    dr = -dr  # columns are not flipped

# For fence placements:
if flip:
    r = FENCE_GRID - 1 - r  # mirror back to actual row
```

**Why not fix it now:**
The fix is straightforward but touches action resolution logic that is not exercised until self-play begins. Deferring avoids accidentally re-introducing the bug if the action encoding is modified during Stage 4. Fix this as the first task when self-play is introduced.

*Decision recorded: March 31, 2026*
