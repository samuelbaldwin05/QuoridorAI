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
Kept it small (2 conv layers, 32->64 filters) since we're training on CPU/MPS.
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
Win rate vs the training opponent every 5k steps, 20 episodes each eval.
Graduation threshold: 80% win rate.

## PyTorch version
Had to upgrade from 2.1.2 to 2.11.0 because old torch couldn't handle numpy 2.x.
torchvision now shows a version warning but we don't use it.

---

## Training log

### Run 1 — baseline (2026-03-26)

Config: lr=1e-4, target_update=500, reward=sparse (+1/-1/0), eval=50 games every 1k steps.

Results:
- Step 1k-8k: win rate climbed from 40% to 62%, episode length dropped to ~150
- Step 10k+: win rate collapsed to 22-34%, never recovered through step 24k
- Step 20k: episode length spiked to 613 — agent was stuck placing walls instead of moving

The eval overhead was massive (50 games every 1k steps), so we also weren't
training fast enough between evaluations.

### Run 2 — reduced eval frequency (2026-03-26)

Changed eval from 50 games/1k steps to 20 games/5k steps. Same config otherwise.

Results:
- Step 5k: 35% win rate
- Step 10k: jumped to 70% (best we've seen), episode length 120
- Step 15k: collapsed to 5%, episode length 265
- Step 20k: 20%, episode length 280

Same collapse pattern as run 1, just more visible with the wider eval spacing.
Training was much faster between evals though — the reduced eval overhead helped.

### Diagnosis

Both runs show the same failure mode: the agent learns early to move toward
the goal, then "discovers" wall placement and gets stuck in a loop of placing
walls instead of advancing. With purely sparse reward (+1/-1 only at game end,
0 for everything else), the agent has no signal that stalling is bad.

The collapse around step 10-15k consistently happens as epsilon drops below
~0.43. At higher epsilon the random exploration forces enough pawn moves to
accidentally win. As exploration decreases, the agent's learned preference
for wall-spamming takes over.

Additional concern: the target network updates every 500 steps may be too
frequent, causing the TD targets to shift too fast and destabilize learning.
And lr=1e-4 may be overshooting once the network starts to converge.

### Run 3 — step penalty + stability fixes (2026-03-26)

Changes made:
1. **Step penalty: -0.01 per non-terminal step.** This is the big one. The agent
   now has a reason to prefer shorter games. Wall-spamming means more steps,
   which means more negative reward. Moving toward the goal and winning fast
   is now strictly better than stalling.
2. **Learning rate: 1e-4 -> 5e-5.** Slower updates to reduce the chance of
   overshooting good Q-values once the network starts learning.
3. **Target network update: 500 -> 200 steps.** More frequent updates keep
   the target closer to the online net. With a lower learning rate, the risk
   of instability from frequent updates is reduced.

Rationale: The step penalty directly addresses the observed failure mode. The
LR and target update changes are stability improvements that should help the
network hold onto good policies once it finds them, instead of oscillating
past them.

Results:
- Step 5k: 60% win rate, ep length 168 — stronger start than any previous run
- Step 10-30k: bounced between 30-50%, ep length 100-250 — similar plateau
- Step 40-60k: wall-spam phase again (10-15% win rate, ep length 400-676)
  BUT step penalty prevented full collapse — episodes stayed shorter than
  run 1's 613+ lengths, and the agent kept getting negative reward for stalling
- **Step 65k: 80% win rate, ep length 56 — GRADUATED**

The agent went through the same wall-spamming phase but came out the other side.
The step penalty made wall-spam a losing strategy long-term: every non-terminal
step costs -0.01, so a 600-step game of wall-spam costs -6.0 even if you win (+1.0).
The network eventually learned that fast pawn movement to the goal is the only
way to get positive net reward.

Key insight: the "U-shaped" learning curve (good -> bad -> good) seems inherent
to this problem. The agent first learns basic movement, then discovers wall
placement and over-indexes on it, then finally learns to balance both. The step
penalty didn't prevent the dip but ensured recovery.

Checkpoint saved: `checkpoints/best.pt` (80% win rate at step 65k)

### Next: curriculum phase 2 — HeuristicBot

Now warm-starting from the RandomBot checkpoint and training against the harder
HeuristicBot. Command:
```
python scripts/train_dqn.py --opponent heuristic --checkpoint checkpoints/best.pt --no-wandb
```
