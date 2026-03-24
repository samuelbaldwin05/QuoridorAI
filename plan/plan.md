# DQN Implementation Plan

**Deadline:** ~April 7–14, 2026 (3–4 weeks from March 17, 2026)
**Hardware:** CPU only — all design choices account for this constraint
**Goal:** Vanilla DQN agent achieving >80% win rate vs HeuristicBot

---

## Overview

The game engine and heuristic bot are complete. The DQN phase is split into four sequential
stages, each building on the last. A team member picks up where the previous person left off
using the handoff notes at the bottom of each module.

```
Stage 1: Infrastructure   (action encoding, observation, legal mask, Gym wrapper)  ✅ COMPLETE
Stage 2: DQN Core         (neural network, replay buffer, config)                  ← START HERE
Stage 3: Training Loop    (train script, W&B logging, checkpointing)
Stage 4: Evaluation       (win rate tracking, tuning, handoff docs)
```

---

## Stage 1 — Infrastructure ✅ COMPLETE

All four pieces are implemented and committed. The training loop can be built on top of
this infrastructure now.

### 1.1 `quoridor/action_encoding.py` ✅

**Status:** Done. Note — the final implementation expanded the pawn action count from 4
to 8 to support diagonal jump moves. All constants shifted accordingly.

**Constants (as implemented):**
- `NUM_ACTIONS = 137`
- `MOVE_ACTION_COUNT = 8`  ← 4 cardinal + 4 diagonal jump directions
- `H_WALL_OFFSET = 8`
- `V_WALL_OFFSET = 72`
- `PASS_ACTION = 136`
- `FENCE_GRID = 8`

**Action layout (as implemented):**

| Index range | Type | Encoding formula |
|---|---|---|
| 0 | Move up | fixed delta (-1, 0) |
| 1 | Move down | fixed delta (+1, 0) |
| 2 | Move left | fixed delta (0, -1) |
| 3 | Move right | fixed delta (0, +1) |
| 4 | Diagonal NW | fixed delta (-1, -1) — diagonal jump only |
| 5 | Diagonal NE | fixed delta (-1, +1) — diagonal jump only |
| 6 | Diagonal SW | fixed delta (+1, -1) — diagonal jump only |
| 7 | Diagonal SE | fixed delta (+1, +1) — diagonal jump only |
| 8 – 71 | Horizontal wall | `8 + row*8 + col` |
| 72 – 135 | Vertical wall | `72 + row*8 + col` |
| 136 | Pass (never legal) | fixed |

**Why 8 pawn actions:** Cardinal moves encode direction *deltas*, covering both 1-step
moves and 2-step straight jumps (same sign). Diagonal moves arise only during jump
situations when a straight jump is wall-blocked. 4 actions would make those destinations
unrepresentable. See `action_encoding.py` module docstring for full explanation.

**IMPORTANT for config.py:** Use `NUM_ACTIONS = 137` everywhere, not 133.

---

### 1.2 `get_observation()` on `QuoridorState` ✅

**Status:** Done (`quoridor/game.py`, line 247).

**Channels (as implemented):**

| Channel | Contents | Player perspective |
|---|---|---|
| 0 | Current player pawn | Always "my pawn" |
| 1 | Opponent pawn | Always "their pawn" |
| 2 | Horizontal walls (8×8 in top-left of 9×9) | Flipped vertically for P1 |
| 3 | Vertical walls (8×8 in top-left of 9×9) | Flipped vertically for P1 |

Scalars: `[walls_left[current] / 10.0, walls_left[opponent] / 10.0]`, shape `(2,)`.

P1's board is flipped vertically (row `i` → row `8-i`) so both players always see their
goal at row 0. Wall grids are also flipped to stay aligned with pawn positions.

---

### 1.3 `get_legal_mask()` on `QuoridorState` ✅

**Status:** Done (`quoridor/game.py`, line 305). Returns `(137,)` bool array.

Pawn moves are derived from `get_valid_moves()` by collapsing each destination to its
direction sign (`np.sign`), then looking up `DELTA_TO_INDEX`. Wall actions call `_fence_ok()`.
Pass (index 136) is always False.

---

### 1.4 `quoridor/env.py` — Gym-style wrapper ✅

**Status:** Done. Agent is always P0; HeuristicBot plays P1 internally.

Reward structure (as implemented):
- `+1.0` — P0 (agent) reaches row 0
- `-1.0` — P1 (bot) reaches row 8
- `0.0` — all other transitions

`step()` returns `info={"legal_mask": (137,) bool}`. Terminal steps return `_TERMINAL_MASK`
(all False). Illegal actions raise `ValueError`.

---

## Stage 2 — DQN Core

Depends on: Stage 1 complete.

### 2.1 `config.py`

All hyperparameters as named constants. Written before the model or training loop so
everyone agrees on defaults.

```python
# Network
CONV1_FILTERS    = 32
CONV2_FILTERS    = 64
CONV_KERNEL_SIZE = 3
FC_HIDDEN_SIZE   = 256

# Training
LEARNING_RATE      = 1e-4
GAMMA              = 0.99
BATCH_SIZE         = 64
REPLAY_BUFFER_SIZE = 50_000   # CPU-scaled; would be 100k+ on GPU
TARGET_UPDATE_FREQ = 500       # hard update every N steps
GRADIENT_CLIP_NORM = 10.0      # per CLAUDE.md

# Exploration
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.999          # multiply per episode

# Evaluation
EVAL_FREQ       = 1_000        # evaluate every N training steps
EVAL_EPISODES   = 50
WIN_RATE_TARGET = 0.80

# Reproducibility
SEED = 42
```

---

### 2.2 `agents/dqn_model.py`

**Architecture** (CPU-optimized — kept small deliberately):

```
Input spatial: (B, 4, 9, 9)
→ Conv2d(4,  32, kernel=3, padding=1) → BatchNorm2d → ReLU   # (B, 32, 9, 9)
→ Conv2d(32, 64, kernel=3, padding=1) → BatchNorm2d → ReLU   # (B, 64, 9, 9)
→ Flatten                                                      # (B, 5184)

Input scalars: (B, 2)

→ Concat([flattened, scalars])                                 # (B, 5186)
→ Linear(5186, 256) → ReLU
→ Linear(256, 137)                                             # Q-values, shape (B, 137)
```

**Functions:**
- `forward(spatial, scalars, legal_mask) -> torch.Tensor` — applies mask before returning
- Masking: `q_values[~legal_mask] = -1e9` before returning

---

### 2.3 `agents/replay_buffer.py`

**Stores per transition:**
```
state_spatial   : (4, 9, 9)  float32
state_scalars   : (2,)       float32
action          : int
reward          : float
next_spatial    : (4, 9, 9)  float32
next_scalars    : (2,)       float32
done            : bool
next_legal_mask : (137,)     bool     ← required for Double DQN action selection
```

**Interface:**
```python
class ReplayBuffer:
    def push(self, transition: dict) -> None
    def sample(self, batch_size: int) -> dict
    def __len__(self) -> int
```

---

## Stage 3 — Training Loop

Depends on: Stages 1 and 2 complete.

### 3.1 `scripts/train_dqn.py`

**Loop pseudocode:**
```
seed everything (torch, numpy, env) with SEED
initialize env, online_net, target_net (copy of online), optimizer, replay_buffer
initialize W&B run

for each episode:
    obs, scalars = env.reset()
    legal_mask   = env.get_legal_mask()

    while not done:
        # Switch to eval mode for inference: BatchNorm uses running statistics
        # instead of single-sample batch statistics during action selection.
        online_net.eval()
        with torch.no_grad():
            action = epsilon_greedy(online_net, obs, scalars, legal_mask, epsilon)
        online_net.train()

        next_obs, next_scalars, reward, done, info = env.step(action)
        next_legal_mask = info["legal_mask"]

        replay_buffer.push(transition)

        if len(replay_buffer) >= BATCH_SIZE:
            # online_net is already in train() mode here — BatchNorm uses batch stats
            # target_net must be in eval() mode so its BatchNorm is also stable
            target_net.eval()
            batch = replay_buffer.sample(BATCH_SIZE)
            loss  = compute_td_loss(batch, online_net, target_net)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(online_net.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()

        if steps % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(online_net.state_dict())
            # Keep target_net in eval mode after copying weights
            target_net.eval()

        if steps % EVAL_FREQ == 0:
            online_net.eval()
            win_rate = evaluate(online_net, env, EVAL_EPISODES)
            online_net.train()
            log to W&B

        obs, scalars, legal_mask = next_obs, next_scalars, next_legal_mask

    decay epsilon
    save checkpoint if win_rate improved
```

### 3.2 W&B Logging

Log every `EVAL_FREQ` steps:
- `train/loss`
- `train/epsilon`
- `train/mean_q_value`
- `train/grad_norm`
- `eval/win_rate`
- `eval/mean_episode_length`

---

## Stage 4 — Evaluation and Handoff

- Run 200-episode evaluation vs HeuristicBot; log results
- Save final checkpoint with config as artifact in W&B
- Update `plan/member_tracker.md` with what was completed and what remains
- Annotate anything that was surprising or deviated from the plan in `plan/decisions.md`

---

## Graduation Criterion

> Win rate > 80% consistently (across at least 3 evaluation checkpoints) against HeuristicBot,
> with stable loss curve and no divergence.

Do not move to PPO until this is met. Understanding *why* the agent converged (or failed to)
is as important as the number itself.

---

## Handoff Protocol

When passing work to the next team member:
1. Push all changes to a feature branch and open a PR
2. In the PR description, write: what was built, what works, what is left, and any open questions
3. Update `plan/member_tracker.md` with your name, what you completed, and what the next person should start on
4. Leave a `# HANDOFF:` comment at the top of any file you left mid-implementation

---

*Plan authored: March 17, 2026*
