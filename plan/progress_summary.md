# QuoridorAI — Progress Summary
*Last updated: April 2, 2026*

---

## Project Goal

Build a deep RL agent that plays Quoridor, progressing through three phases:

| Phase | Status |
|---|---|
| **DQN** | ❌ Abandoned — 0% win rate after 500k steps |
| **PPO** | 🔄 Active — 0 wins so far, experiments running |
| **AlphaZero** | ⏳ Future |

**Current honest state: we have never beaten HeuristicBot once.** Getting that first win is the entire focus right now.

---

## Phase 1: Game Engine ✅

Built by Sam Baldwin (~Feb 2026). Jeffrey built the RL interface on top.

**Files:** `quoridor/game.py`, `quoridor/env.py`, `quoridor/action_encoding.py`

**What exists:**
- Full Quoridor rules with BFS-based wall legality validation
- 137-action space: 8 pawn moves (cardinal + diagonal jumps) + 128 wall placements + 1 pass
- Gym-compatible `QuoridorEnv` — agent plays P0, bot plays P1
- 4-channel observation `(4, 9, 9)` + scalars `(2,)`, or 6-channel with BFS distance maps
- Reward shaping support via `use_reward_shaping` flag
- HeuristicBot: UCS pathfinding, opening book, win-in-one detection (12× speedup added March 24)
- RandomBot for curriculum
- **EpsilonHeuristicBot** (new): ε-greedy wrapper around HeuristicBot — takes random legal action with probability ε, otherwise delegates to HeuristicBot. Enables curriculum where the opponent is weaker early, strengthening as the agent wins more.
- **`STEP_PENALTY = -0.01`** added to env.py: small per-step penalty to discourage stalling (non-terminal steps only)

**Locked-in architecture decisions:**

| Decision | Choice | Why |
|---|---|---|
| Action space | 137 discrete (8 pawn + 128 wall + 1 pass) | Includes diagonal jump moves |
| State channels | (4, 9, 9): pawn, opp pawn, h-walls, v-walls | CNN-friendly spatial encoding |
| BFS extension | Optional (6, 9, 9): adds my/opp distance maps | Direct path-length signal for walls |
| Perspective | Current-player-normalized always | Simplifies learning |
| Illegal actions | Masked to −1e9, never sampled | Hard constraint, not penalized |
| Scalars | `[my_walls/10, opp_walls/10]` | Late-fused after CNN flatten |
| Pawn encoding | Direction deltas not absolute positions | 1 index covers both 1-step and 2-step jump |

---

## Phase 2: DQN ❌ Abandoned

**Built:** March 17–24, 2026

**What was built:**
- `DQNModel`: Conv(4→32→64, BatchNorm) → FC(5186→256→137)
- `ReplayBuffer` + `PrioritizedReplayBuffer` (PER, α=0.6, β annealed 0.4→1.0)
- `train_dqn.py`: Double DQN, ε-greedy exploration, curriculum, W&B logging
- N-step returns (n=5) for faster credit propagation
- PBRS reward shaping (Φ = path differential / 16)
- DQN self-play curriculum (frozen `DQNBot` updated every 10k steps)
- BC pretraining pipeline: `generate_bc_data.py` → `pretrain_bc.py` → `bc_pretrained.pt`

**Key experiment results:**

| Experiment | Steps | Result |
|---|---|---|
| DQN vs HeuristicBot (direct) | 131k | 0% — Q-values collapsed to −1.17 (below theoretical min) |
| DQN with reward shaping vs HeuristicBot | ~131k | Made collapse **worse** — shaped rewards were all negative (always losing), amplified by n-step |
| DQN frozen self-play | ~370k | Beat RandomBot ~100%. Still 0% vs HeuristicBot |
| DQN total | 500k | **0% vs HeuristicBot** — abandoned |
| BC pretraining (DQN backbone) | 20 epochs | **20% win rate** greedy vs HeuristicBot ← only positive result |

**Why DQN failed (root cause):**
1. **No winning signal.** Against HeuristicBot, the agent never wins → Q-values see only −1 → all Q(s,a) → −1 → gradient disappears. This is a cold-start problem, not an architecture problem.
2. **Reward shaping backfired.** Without any wins, path differential was always negative → shaped reward reinforced losing. Combined with n-step returns, it accelerated collapse.
3. **BC warmstart couldn't survive RL.** DQN's TD loss overwrote BC features within ~25k steps because DQN has no entropy protection to resist collapse.

---

## Phase 3: PPO 🔄 Active

### Why PPO over DQN

PPO's entropy regularization penalizes the policy for becoming deterministic. Even with 100% losses, the policy is forced to keep exploring — the −1 fixed point collapse is structurally prevented. The BC warmstart has a chance to survive because the policy can't collapse.

### Model Variants (all implemented)

| `--model` | File | Channels | Backbone | BC warmstart | Aux head |
|---|---|---|---|---|---|
| `baseline` | `ppo_model.py` | 4 | 2-conv + BatchNorm | ✅ 14 tensors load | ❌ |
| `resnet` | `ppo_model_resnet.py` | 4 | ResNet + LayerNorm | ❌ 0 tensors (key mismatch) | ❌ |
| `bfs` | `ppo_model_bfs.py` | 6 | 2-conv + LayerNorm | ❌ crashes (shape mismatch) | ❌ |
| `bfs_resnet` | `ppo_model_bfs_resnet.py` | 6 | ResNet + LayerNorm | ✅ `bc_pretrained_bfs_resnet.pt` exists | ✅ |

LayerNorm replaces BatchNorm in all new variants — critical for PPO which constantly switches between `model.train()` and `model.eval()`. BatchNorm uses different statistics in each mode; LayerNorm is identical.

**Auxiliary prediction head (bfs_resnet only):** Third output head predicts `[my_dist_to_goal, opp_dist_to_goal]` from the shared backbone. Targets are extracted for free from BFS channels already in the rollout (dot product of distance map channel with pawn position channel). Forces the backbone to build explicit path-distance representations. Loss weight `AUX_LOSS_COEF = 0.1`.

**BC warmstart for bfs_resnet:** `checkpoints/bc_pretrained_bfs_resnet.pt` was generated by a parallel agent on April 1. This is the 6-channel equivalent of `bc_pretrained.pt` — enables warmstart for the recommended model variant.

### PPO Experiments Run

| Run | Config | Result |
|---|---|---|
| PPO baseline, LR=3e-4, epochs=4 | With BC warmstart | KL=0.27–0.37, entropy collapsed, BC warmstart destroyed in ~2k steps ❌ |
| PPO baseline, LR=3e-5, epochs=2 | With BC warmstart | KL=0.0012 ✅, clip_fraction~0% ⚠️ (possibly too conservative) |
| PPO bfs_resnet + pool self-play + reward shaping | Cold start | `wall_prob_mass` collapsed to 0 repeatedly, oscillated, **0 wins** ❌ |

**Why the pool + reward shaping run failed (W&B analysis):**
- `train/value_loss` spiked to 0.3 on every pool update (new checkpoint enters pool → value estimates suddenly wrong)
- Each value loss spike created garbage advantages → noisy policy gradient → wall_prob_mass crashed to 0
- Reward shaping adds noise: the shaped reward's meaning changes with each pool opponent (PBRS is only policy-invariant vs a fixed opponent)
- Self-play pool of losing agents teaches nothing — nobody in the pool has ever won

### Current PPO Hyperparameters

| Param | Value | Why |
|---|---|---|
| `PPO_LR` | 3e-5 | Reduced 10× from 3e-4 — original destroyed BC warmstart via KL explosion |
| `PPO_EPOCHS` | 2 | Reduced from 4 — same reason |
| `PPO_ENTROPY_COEF_START` | 0.05 | High start — forces wall exploration, prevents 128-action collapse |
| `PPO_ENTROPY_COEF_END` | 0.005 | Low end — permits specialization once agent knows which walls matter |
| `PPO_GRAD_CLIP` | 10.0 | Current; 0.5 comparison experiment running |
| `PPO_GAE_LAMBDA` | 0.95 | ~100-step horizon; 0.97 experiment running |
| `PPO_GAMMA` | 0.99 | ~100-step horizon; 0.999 experiment running |
| `ROLLOUT_STEPS` | 512 | ~17 episodes per rollout |
| `PPO_MINI_BATCH_SIZE` | 64 | 512/64 = 8 updates per epoch |
| `STEP_PENALTY` | -0.01 | Per-step penalty to discourage stalling |

### PPO Techniques Implemented vs Planned

| Technique | Status | Where |
|---|---|---|
| Entropy scheduling (0.05→0.005) | ✅ Implemented | `train_ppo.py` main loop |
| Per-action-type entropy logging | ✅ Implemented | `train_ppo.py`, W&B: `train/wall_prob_mass` etc. |
| Per-rollout advantage normalization | ✅ Implemented | `train_ppo.py` after GAE |
| CLI overrides (--gamma, --lam, --grad-clip) | ✅ Implemented | `train_ppo.py` argparse |
| OpponentPool / Fictitious Self-Play | ✅ Implemented | `train_ppo.py`, `--opponent pool` |
| EpsilonHeuristicBot curriculum | ✅ Implemented | `agents/bot.py`, `--epsilon-curriculum` flag |
| Auxiliary path-length head (bfs_resnet) | ✅ Implemented | `agents/ppo_model_bfs_resnet.py`, `AUX_LOSS_COEF=0.1` |
| BC warmstart for bfs_resnet | ✅ Done | `checkpoints/bc_pretrained_bfs_resnet.pt` |
| γ=0.999, λ=0.97 (longer horizon) | 🔄 Experiment running | batch run `baseline_long_hz` |
| Grad clip 0.5 vs 10.0 comparison | 🔄 Experiment running | batch run `baseline_clip05` |
| Value loss clipping | ✅ Disabled (simple MSE) | Correct per ppo_techniques.md |
| PBRS reward shaping vs fixed opponent | ⚠️ Implemented but not in batch | `env.py` `use_reward_shaping` flag |
| Value head burn-in before PPO fine-tune | ❌ Not done | Needed for warmstart to survive |
| Differential LR (backbone/actor/value) | ❌ Not done | Single Adam LR currently |
| PPOBot flip bug fix | ❌ Not done | Blocks self-play phase |

---

## Batch Experiments (Running Now)

Five runs × 150k steps each, sequential via `scripts/run_batch.sh`:

| Name | Command | Tests |
|---|---|---|
| `baseline_bc` | `--model baseline --checkpoint bc_pretrained.pt` | Control: BC warmstart, 4-ch |
| `resnet_cold` | `--model resnet` | LayerNorm fix, 4-ch, cold |
| `bfs_resnet_cold` | `--model bfs_resnet` | BFS + ResNet, 6-ch, cold |
| `baseline_long_hz` | `--model baseline --checkpoint bc_pretrained.pt --gamma 0.999 --lam 0.97` | Longer credit horizon |
| `baseline_clip05` | `--model baseline --checkpoint bc_pretrained.pt --grad-clip 0.5` | Tighter trust region |

**What to watch in W&B:**
- `train/wall_prob_mass` — must stay above 0.1 or the policy is ignoring walls
- `eval/win_rate` — any non-zero value is progress
- `train/entropy` — should stay above 1.0 early
- `train/value_loss` spikes — indicate policy destabilization

---

## What the Batch is Missing

The batch runs don't include `--reward-shaping`. Given we've never won once, reward shaping against a **fixed HeuristicBot** is likely necessary to generate any gradient signal at all. Next run to add after batch completes:

```bash
python -m scripts.train_ppo \
    --model bfs_resnet \
    --checkpoint checkpoints/bc_pretrained_bfs_resnet.pt \
    --reward-shaping \
    --run-name bfs_resnet_shaped_heuristic
```

This is the highest-priority next experiment — BFS channels give direct path-length feedback, reward shaping gives dense gradient even on losses, BC warmstart via the 6-channel checkpoint, fixed HeuristicBot avoids the non-stationarity that killed the pool run.

**Second priority if still 0 wins:** Add `--epsilon-curriculum` to weaken HeuristicBot until we get our first wins:

```bash
python -m scripts.train_ppo \
    --model bfs_resnet \
    --checkpoint checkpoints/bc_pretrained_bfs_resnet.pt \
    --reward-shaping \
    --epsilon-curriculum \
    --run-name bfs_resnet_epsilon_curriculum
```

---

## Critical Known Issues

| Issue | Severity | Blocks |
|---|---|---|
| **Zero wins ever vs HeuristicBot** | 🔴 Critical | Everything |
| **PPOBot action flip bug** (Decision 16) | 🔴 High | Self-play phase |
| `clip_fraction ≈ 0%` (LR may be too conservative) | 🟡 Medium | Efficient learning |
| Config drift: `action_encoding.py` defines `NUM_ACTIONS` locally | 🟡 Medium | Any action space change |
| Value head burn-in not implemented | 🟡 Medium | BC warmstart survival |
| `game.py` hardcodes `walls_left=[10,10]` and normalization divisor | 🟢 Low | Wall count changes |

### PPOBot Flip Bug (Must Fix Before Self-Play)

When `PPOBot` plays as P1, `get_observation()` returns a flipped board but decoded actions are applied in unflipped coordinates. Pawn moves are mirrored (moving away from goal instead of toward it), and fence rows are placed on the wrong half of the board. Fix:

```python
flip = (game.turn == 1)
if flip:
    dr = -dr               # pawn moves
    r = FENCE_GRID - 1 - r # fence placements
```

Does not affect current PPO-vs-HeuristicBot training (PPOBot is only used in self-play).

---

## File Map

```
config.py                    — all hyperparameters (single source of truth)
CLAUDE.md                    — project memory for Claude sessions

quoridor/
  game.py                    — Quoridor rules, BFS pathfinding, observation
  env.py                     — Gym wrapper; use_bfs + use_reward_shaping flags; STEP_PENALTY
  action_encoding.py         — 137-action ↔ game-move bijection

agents/
  bot.py                     — HeuristicBot (UCS, opening book, win-in-one)
                               EpsilonHeuristicBot (ε-greedy wrapper for curriculum)
  random_bot.py              — RandomBot (curriculum phase 1)
  dqn_model.py               — DQN Q-network (archived)
  dqn_bot.py                 — Frozen DQN opponent for self-play
  replay_buffer.py           — PER replay buffer (DQN only)
  ppo_model.py               — PPO baseline (4-ch, BatchNorm)
  ppo_model_resnet.py        — PPO ResNet (4-ch, LayerNorm, skip connections)
  ppo_model_bfs.py           — PPO BFS (6-ch, LayerNorm)
  ppo_model_bfs_resnet.py    — PPO BFS+ResNet (6-ch, LayerNorm, skip, aux head) ← recommended
  ppo_bot.py                 — Wraps any PPO checkpoint as a bot (has flip bug for P1)

scripts/
  train_ppo.py               — PPO training loop (main entry point)
                               --opponent {heuristic,self_play,pool}
                               --epsilon-curriculum (uses EpsilonHeuristicBot)
                               --value-burnin-steps
                               --gamma, --lam, --grad-clip, --run-name
  train_dqn.py               — DQN training loop (deprecated)
  run_batch.sh               — 5 sequential experiments × 150k steps
  run_bc_pipeline.sh         — BC data generation + pretraining pipeline
  generate_bc_data.py        — HeuristicBot self-play → bc_data.npz
  pretrain_bc.py             — Supervised BC pretraining (DQN backbone)
  watch_ppo.py               — Visualize trained PPO agent
  play_bot.py                — Bot vs bot evaluation

checkpoints/
  bc_pretrained.pt           — BC warmstart for baseline/4-ch models (20% win rate greedy)
  bc_pretrained_bfs_resnet.pt — BC warmstart for bfs_resnet/6-ch model ← use with recommended model

plan/
  decisions.md               — 21 architectural decisions with full rationale
  ppo_techniques.md          — 8 PPO techniques reference (aspirational; partially implemented)
  progress_summary.md        — this file
  research.md                — RL concepts reference
```

---

## Graduation Criteria

| Phase | Criterion |
|---|---|
| PPO vs HeuristicBot | Win rate > 80% consistently |
| PPO self-play | Beats HeuristicBot reliably; PPOBot flip bug fixed |
| AlphaZero | Strong PPO baseline established; Jeffrey comfortable with MCTS |

Do not start self-play until HeuristicBot win rate is reliably above 30%. A pool of all-losing policies teaches nothing.
