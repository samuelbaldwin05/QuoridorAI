# PPO Techniques Reference — Quoridor-Specific

> This document is the authoritative reference for PPO implementation decisions in QuoridorAI.
> All techniques are framed in terms of our specific setup: `(4, 9, 9)` state tensor, 137 discrete
> actions (8 pawn moves + 128 wall placements + 1 pass), sparse +1/−1 terminal rewards, 20–40
> move episodes, CNN backbone with late fusion, action masking.

---

## 1. Entropy Coefficient Scheduling

**Problem it solves:** With 137 actions, the policy collapses to movement-only within early training.
Wall placements (128/137 of the action space) get starved of probability mass and are never explored.
This is the most likely cause of the DQN agent's wall wastage — and PPO will repeat it without
explicit exploration pressure.

**Implementation:**
```python
# In config.py
ENTROPY_COEFF_START = 0.05      # High — forces wall exploration early
ENTROPY_COEFF_END = 0.005       # Low — permits precise strategy late
ENTROPY_DECAY_FRAC = 0.8        # Decay over 80% of total training steps

# In train_ppo.py, called each update
def get_entropy_coeff(step: int, total_steps: int) -> float:
    frac = min(step / (total_steps * ENTROPY_DECAY_FRAC), 1.0)
    return ENTROPY_COEFF_START + frac * (ENTROPY_COEFF_END - ENTROPY_COEFF_START)
```

**Monitoring (W&B):** Log `policy_entropy` every update. Healthy range mid-training: **1.0–3.0 nats**.
- Near 4.9 (max = log(137)) → too random, not learning
- Below 0.5 → collapsed, likely ignoring walls
- Steady decline over training → healthy

**Pitfall:** Decaying too fast before the agent discovers wall value. If win rate is still 0% at 50%
of training, consider resetting or slowing the schedule.

---

## 2. Per-Action-Type Entropy Monitoring

**Problem it solves:** Overall entropy of 3.0 can hide that wall entropy is 0. You need decomposed
metrics to catch this.

**Implementation — logging only (start here):**
```python
# After computing action distribution from policy head
def log_action_type_entropy(logits: torch.Tensor, legal_mask: torch.Tensor) -> dict:
    """Decompose entropy into move vs wall components.
    
    Args:
        logits: (B, 137) raw logits from policy head
        legal_mask: (B, 137) binary mask of legal actions
    
    Returns dict with keys: total_entropy, move_entropy, wall_entropy,
                            move_prob_mass, wall_prob_mass
    """
    # Apply mask, compute probs
    masked_logits = logits.clone()
    masked_logits[~legal_mask.bool()] = -1e8
    probs = F.softmax(masked_logits, dim=-1)  # (B, 137)
    
    # Split: actions 0-7 = moves, 8-135 = walls, 136 = pass
    # ⚠️ VERIFY these indices match action_encoding.py! If the ordering differs
    # (e.g. walls first, then moves), this decomposition is silently wrong.
    # Check against: MOVE_ACTION_COUNT, FENCE_ACTION_COUNT in action_encoding.py
    move_probs = probs[:, :8]          # (B, 8)
    wall_probs = probs[:, 8:136]       # (B, 128)
    
    move_mass = move_probs.sum(dim=-1)  # (B,) — P(choosing any move)
    wall_mass = wall_probs.sum(dim=-1)  # (B,) — P(choosing any wall)
    
    # Conditional entropies (entropy within each type)
    move_ent = -((move_probs / (move_mass.unsqueeze(-1) + 1e-8)) *
                  (move_probs / (move_mass.unsqueeze(-1) + 1e-8) + 1e-8).log()).sum(-1)
    wall_ent = -((wall_probs / (wall_mass.unsqueeze(-1) + 1e-8)) *
                  (wall_probs / (wall_mass.unsqueeze(-1) + 1e-8) + 1e-8).log()).sum(-1)
    
    return {
        "move_entropy": move_ent.mean().item(),      # max ≈ 2.08
        "wall_entropy": wall_ent.mean().item(),      # max ≈ 4.85
        "move_prob_mass": move_mass.mean().item(),   # healthy: 0.1–0.6
        "wall_prob_mass": wall_mass.mean().item(),   # healthy: 0.4–0.9
    }
```

**Warning signs:**
- `wall_prob_mass` → 0: agent ignoring all 128 wall placements (same failure as DQN)
- `wall_entropy` < 1.0 while `move_entropy` is healthy: wall collapse hidden by aggregate
- `move_prob_mass` → 0: agent only walls, never advances (rare but possible)

**Advanced — per-type entropy bonus in loss (only if monitoring shows collapse):**
Replace single entropy bonus with: `loss -= c_type * H_type + c_move * H_move + c_wall * H_wall`
where `c_type = 0.05`, `c_move = 0.02`, `c_wall = 0.01`.

---

## 3. Advantage Normalization — Per-Rollout

**Problem it solves:** Per-minibatch normalization (the CleanRL/SB3 default) amplifies noise with
sparse rewards. Most Quoridor transitions have near-zero advantage; normalizing a minibatch of
near-zeros inflates irrelevant differences.

**Implementation:**
```python
# In train_ppo.py — normalize ONCE before the epoch loop
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

# Then iterate over epochs and minibatches WITHOUT re-normalizing
for epoch in range(PPO_EPOCHS):
    for mb_indices in minibatch_generator(batch_size):
        mb_advantages = advantages[mb_indices]  # already normalized
        # ... compute loss with mb_advantages
```

**Critical:** Rollout buffer must contain **several completed episodes**. With 20–40 move games,
ensure `ROLLOUT_STEPS * NUM_ENVS ≥ 400`. If a rollout has zero completed episodes, all advantages
are bootstrapped noise.

---

## 4. Value Function Clipping — DISABLED

**Do not clip value loss.** Evidence from Andrychowicz et al. (ICLR 2021) shows it hurts performance
across the board. With our ±1 sparse rewards, the value function needs to make large corrections
when the agent first starts winning — clipping prevents this.

**Use instead:**
```python
# Simple MSE loss with coefficient
value_loss = 0.5 * F.mse_loss(value_pred, returns)
total_loss = policy_loss - entropy_coeff * entropy + VF_COEFF * value_loss
```

Add `VF_COEFF = 0.5` to `config.py`. Global gradient clipping at `MAX_GRAD_NORM = 0.5` provides
sufficient stability without slowing value learning.

---

## 5. Gradient Norm Monitoring

**Implementation:**
```python
# In train_ppo.py, after loss.backward()
pre_clip_norm = torch.nn.utils.clip_grad_norm_(
    model.parameters(), max_norm=MAX_GRAD_NORM
)
wandb.log({
    "grad/pre_clip_norm": pre_clip_norm.item(),
    "grad/clipping_active": float(pre_clip_norm > MAX_GRAD_NORM),
    # Per-head norms (optional but useful)
    "grad/policy_head_norm": model.actor_head[-1].weight.grad.norm(2).item(),
    "grad/value_head_norm": model.critic_head[-1].weight.grad.norm(2).item(),
})
```

**Diagnostic guide for Quoridor:**
| Pattern | Meaning | Action |
|---------|---------|--------|
| Pre-clip norm 1–10, decreasing | Healthy | None |
| Persistent spikes >50 | LR too high or degenerate batch | Reduce LR |
| Always at clip limit (0.5) | Constant info loss | Increase MAX_GRAD_NORM or reduce LR |
| Near zero | Vanishing gradients, entropy collapsed | Check entropy coeff |
| Value head >> policy head | Critic dominating backbone | Reduce VF_COEFF or separate networks |
| Spike when first wins appear | Expected — value function adjusting | Monitor, should settle in ~200 updates |

---

## 6. GAE Lambda Tuning

**Problem it solves:** With λ=0.95 and γ=0.99, the effective horizon is ~17 steps. Quoridor games
are 20–40 moves. Early-game wall placements get almost no gradient signal from the terminal reward.

**Our setup:**
```python
# In config.py
GAMMA = 0.999       # Near-zero discounting (game is finite, all moves matter equally)
GAE_LAMBDA = 0.97   # Higher than default — ensures terminal reward reaches move 1

# Effective horizon = 1 / (1 - γλ) = 1 / (1 - 0.969) ≈ 32 steps ✓
```

**Tuning grid if 0.97 doesn't work:**
- If win rate stuck near random → increase to λ=0.99 (more Monte Carlo, better credit to early moves)
- If training is noisy/unstable → decrease to λ=0.95 (more bootstrap, smoother but shorter horizon)
- If using reward shaping (technique #7) → can drop to λ=0.95 since shaping provides dense signal

---

## 7. Potential-Based Reward Shaping (PBRS)

**Problem it solves:** Sparse +1/−1 means the agent gets zero feedback on whether a wall placement
was good until the game ends. PBRS gives dense per-move signal without changing optimal policy.

**Potential function for Quoridor:**
```
Φ(s) = α × (opponent_shortest_path(s) − my_shortest_path(s))
```
Positive when you're ahead (shorter path + opponent blocked), negative when behind.
Shortest paths computed via BFS on the 9×9 board accounting for current walls — **this already
exists in game.py for wall legality validation.**

**Implementation in env.py:**

**⚠️ PERSPECTIVE WARNING:** Our env uses current-player normalization — the board flips after each
move. Both Φ(s) and Φ(s') MUST be computed from the **acting player's** perspective, not
"current player" (which changes after the step). Pin to the player who took the action.

```python
# In config.py
SHAPING_ALPHA = 0.1   # Scale factor — keeps shaping ~10% of terminal reward magnitude

# In QuoridorEnv.step()
def _compute_potential(self, player: int) -> float:
    """Φ(s) from a fixed player's perspective.
    
    Args:
        player: the player whose perspective to use (0 or 1).
                Must be the SAME player for both Φ(s) and Φ(s').
    """
    my_path = self.game.shortest_path_length(player)
    opp_path = self.game.shortest_path_length(1 - player)
    return SHAPING_ALPHA * (opp_path - my_path)

def step(self, action):
    acting_player = self.current_player  # pin BEFORE step
    phi_old = self._compute_potential(acting_player)
    obs, reward, done, info = self._base_step(action)  # executes action, may flip perspective
    
    if done:
        shaped_reward = reward  # terminal: Φ(terminal) = 0
    else:
        phi_new = self._compute_potential(acting_player)  # same player, NOT self.current_player
        shaped_reward = reward + GAMMA * phi_new - phi_old
    
    return obs, shaped_reward, done, info
```

**Four rules that MUST hold:**
1. **Φ(terminal) = 0** — don't add potential to terminal states
2. **γ in shaping formula = γ in PPO** — must match, both from `config.py`
3. **Only use potential-difference form** — ad hoc bonuses like "+0.5 for blocking" are NOT
   policy-invariant and will distort the optimal policy
4. **Both potentials from same player's perspective** — see perspective warning above. With
   current-player normalization, `self.current_player` changes after step. Pin to `acting_player`.

**⚠️ Self-play limitation:** The Ng et al. policy-invariance guarantee assumes a fixed environment
(MDP). Against the heuristic bot, this holds. Once you move to self-play, the opponent is
non-stationary (the MDP changes every update), so the guarantee doesn't strictly apply. PBRS still
provides useful dense signal in self-play, but the theoretical optimality guarantee weakens. Consider
decaying α → 0 as agent strength increases.

**Scaling guide:**
- α = 0.05: gentle nudge, slower learning but safest
- α = 0.1: recommended starting point
- α = 0.2: aggressive, faster learning but may overpower terminal signal if games are short
- Log both `shaped_reward` and `raw_reward` in W&B to monitor the balance

**Performance note:** BFS is O(V+E) = O(81 + ~200) per call, called twice per step. Negligible
compared to neural network forward pass. If profiling shows otherwise, cache and invalidate only
on wall placements.

---

## 8. Behavioral Cloning Warm Start (Revisited)

**Why it failed before (likely):** DQN has a single output head. BC pretrained weights got destroyed
by DQN's different loss signal immediately. Actor-critic architecture solves this because the policy
head can retain BC knowledge while the value head learns independently.

**Pipeline for PPO:**

**Step 1 — Generate better BC data.**
Previous BC data came from the heuristic bot. For a stronger warm start, generate from a deeper
search agent (minimax with alpha-beta, depth 4+, or MCTS 500+ simulations). Target: **100K+
(state, action) pairs**. Store as `data/bc_data_v2.npz`.

**Step 2 — Pretrain backbone + policy head only.**
```python
# Load PPO model, train only backbone + actor_head
# Value head stays randomly initialized
model = PPOModel()
bc_params = list(model.backbone.parameters()) + list(model.actor_head.parameters())
optimizer = Adam(bc_params, lr=1e-3)

for epoch in range(100):
    for states, expert_actions, legal_masks in bc_loader:
        logits = model.get_policy_logits(states)    # (B, 137)
        logits[~legal_masks.bool()] = -1e8          # mask illegal
        loss = F.cross_entropy(logits, expert_actions)
        # Train until top-1 accuracy > 40% (random = 0.7%)
```

**Step 3 — Burn in value head (CRITICAL — this is why it failed before).**
```python
# Freeze backbone + actor_head
# Collect 50-100 rollouts with BC policy (no training on policy)
# Train value head only on returns from those rollouts
for param in model.backbone.parameters():
    param.requires_grad = False
for param in model.actor_head.parameters():
    param.requires_grad = False

value_optimizer = Adam(model.critic_head.parameters(), lr=5e-4)
# 5-10 epochs of value-only training on collected returns
```

**Step 4 — PPO fine-tuning with differential learning rates.**
```python
# In config.py
LR_BACKBONE = 3e-5      # Very slow — preserve BC features
LR_POLICY_HEAD = 1e-4   # Moderate — allow policy refinement
LR_VALUE_HEAD = 5e-4    # Fast — value head still catching up

# In train_ppo.py
optimizer = Adam([
    {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
    {"params": model.actor_head.parameters(), "lr": LR_POLICY_HEAD},
    {"params": model.critic_head.parameters(), "lr": LR_VALUE_HEAD},
])

# Use conservative clip range
CLIP_EPSILON = 0.1  # Tighter than default 0.2 to protect BC knowledge
```

**Optional — KL penalty to BC reference (prevents catastrophic forgetting):**
```python
# Save BC policy logits as reference
bc_reference_logits = model.get_policy_logits(states).detach()

# In PPO loss, add:
kl_loss = F.kl_div(
    F.log_softmax(current_logits, dim=-1),
    F.softmax(bc_reference_logits, dim=-1),
    reduction='batchmean'
)
total_loss += KL_BC_COEFF * kl_loss  # KL_BC_COEFF = 0.01, decay to 0 over training
```

---

## Recommended Implementation Order

1. **Gradient norm monitoring** (#5) — zero-cost diagnostic, add immediately
2. **Per-action-type entropy logging** (#2) — diagnose whether wall collapse is happening
3. **GAE lambda + gamma tuning** (#6) — config.py change, biggest bang for effort
4. **Advantage normalization** (#3) — small code change, fixes sparse reward noise
5. **Entropy scheduling** (#1) — prevents the collapse you'll see in #2
6. **PBRS reward shaping** (#7) — dense signal for wall learning
7. **Disable value clipping** (#4) — verify it's not active, remove if so
8. **BC warm start** (#8) — most complex, do after everything else is stable

---

## Config.py Additions

```python
# === PPO Hyperparameters ===
GAMMA = 0.999
GAE_LAMBDA = 0.97
CLIP_EPSILON = 0.2           # 0.1 if using BC warm start
PPO_EPOCHS = 4
NUM_MINIBATCHES = 4
VF_COEFF = 0.5
MAX_GRAD_NORM = 0.5

# === Entropy Scheduling ===
ENTROPY_COEFF_START = 0.05
ENTROPY_COEFF_END = 0.005
ENTROPY_DECAY_FRAC = 0.8     # fraction of total steps to decay over

# === Reward Shaping ===
USE_REWARD_SHAPING = True
SHAPING_ALPHA = 0.1

# === BC Warm Start ===
USE_BC_WARMSTART = False      # flip to True when ready
BC_CHECKPOINT = "checkpoints/bc_pretrained_v2.pt"
LR_BACKBONE = 3e-5
LR_POLICY_HEAD = 1e-4
LR_VALUE_HEAD = 5e-4
KL_BC_COEFF = 0.01           # decay to 0 over training
