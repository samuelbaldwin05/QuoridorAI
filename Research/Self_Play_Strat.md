# Quoridor Deep RL: Self-Play Training Strategy

A technical record of the self-play design decisions for Phase 2+ training, covering frozen opponent methodology, epsilon annealing schedules, and league play expansion. Written for after the DQN vs. HeuristicBot phase is complete.

---

## Table of Contents

1. [What Is Self-Play and Why It Matters](#1-what-is-self-play-and-why-it-matters)
2. [The Non-Stationarity Problem](#2-the-non-stationarity-problem)
3. [Phase 1: Frozen Opponent Self-Play](#3-phase-1-frozen-opponent-self-play)
4. [Epsilon Annealing in Self-Play](#4-epsilon-annealing-in-self-play)
5. [Phase 2: League Play](#5-phase-2-league-play)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Algorithm Roadmap](#7-algorithm-roadmap)
8. [References](#8-references)

---

## 1. What Is Self-Play and Why It Matters

In the current training setup, the DQN agent plays against a fixed opponent. First the RandomBot, then the HeuristicBot. The opponent never changes. This makes the environment **stationary**: the agent learns a policy against a fixed target.

**Self-play** replaces the fixed opponent with a version of the agent itself. The agent generates its own training signal by competing against a copy of its own weights. This is important because the heuristic bot has a skill ceiling. Once the DQN agent consistently beats the HeuristicBot (~70-80% win rate), the heuristic opponent no longer provides useful gradient signal and every game will look simialar. The agent will stop learning. There is of evidence that self play works. AlphaGo Zero, for example, learned solely through self-play with no human data and defeated every previous version of AlphaGo [Silver et al. 2017].


---

## 2. The Non-Stationarity Problem

The central challenge of self-play is **non-stationarity**. When both the agent and its opponent are learning simultaneously, the opponent's policy shifts under the agent at every update step. Transitions collected 10,000 steps ago were generated against a weaker version of the opponent. When sampled from the replay buffer later, those transitions no longer accurately reflect the current opponent policy.

This breaks a core assumption of DQN: that the experience distribution is stable enough for the Bellman update to converge.

```
Step 1000:  agent plays against opponent_v1 (weak)
            stores (s, a, r, s') in replay buffer

Step 50000: agent samples that old transition
            but now faces opponent_v50 (much stronger)
            the stored reward r is now misleading
            --> gradient update is corrupted
```

The standard solution is to **freeze the opponent** for a period of steps, so that all transitions in the buffer at any given time were collected against an opponent of approximately consistent strength. This is our Phase 1 self-play approach.

---

## 3. Phase 1: Frozen Opponent Self-Play

Rather than updating both agents simultaneously, we maintain two networks:

```
Online network   theta       <-- receives gradient updates every step
Frozen opponent  theta_k     <-- fixed checkpoint, never updated during run
```

The frozen opponent is loaded from a saved checkpoint of the online network. It is evaluated using a fixed epsilon value and its weights never change during the training run. Periodically, if the online agent has improved sufficiently, the frozen opponent is replaced with a new checkpoint.


AlphaGo Zero used a 55% win rate threshold before promoting the new policy to become the next opponent [Silver et al. 2017]. We adopt the same principle:

```
Every EVAL_FREQ steps:
    win_rate = evaluate(online_net, n=200, opponent=frozen_opponent)

    if win_rate >= FROZEN_PROMOTION_THRESHOLD (default: 0.55):
        save checkpoint
        reload frozen_opponent from new checkpoint
        reset win rate tracking
```

This ensures the opponent difficulty scales with the agent. The agent is never trained against an opponent it has already fully solved, nor against one so strong that it cannot learn from the interaction.

### Why Not Pure Self-Play?

Pure self-play (both agents update simultaneously) is theoretically appealing but has a few issues. 

**Catastrophic forgetting** — the agent may learn to beat the current opponent policy, then the opponent updates and the agent's winning strategy becomes invalid. The agent then relearns, only for the cycle to repeat. Training can oscillate indefinitely without converging.

**Replay buffer corruption** — the DQN replay buffer accumulates transitions from past policies. With simultaneous updates, the buffer contains transitions from many different opponent strengths, and the Bellman targets computed from them are inconsistent.

Frozen opponent sidesteps both issues by making the environment stationary for the duration of each training phase between promotions.

---

## 4. Epsilon Annealing in Self-Play

### What Opponent Epsilon Controls

In the current training loop, epsilon controls the agent's exploration. In self-play, we also assign an epsilon to the frozen opponent. This controls how often the opponent ignores its Q-values and picks a random legal action. A high opponent epsilon produces a weaker, more unpredictable opponent. A low opponent epsilon produces a near-greedy, stronger opponent.

This is distinct from simulated annealing in the classical optimization sense, but the analogy holds: high temperature (high epsilon) = more random exploration of the game tree and low temperature = more exploitation of learned strategy.

### The Problem With a Fixed Opponent Epsilon

Choosing a single fixed epsilon for the opponent is brittle. A value that is appropriate early in training (when the agent is weak and needs an easy opponent) may be too easy later, and vice versa.

Research on adaptive exploration [Tokic 2010] confirms that fixed epsilon values require environment-specific tuning and that the optimal value varies significantly across training stages. No single value generalizes.

### Proposed Experiment: Epsilon Schedule Comparison

We propose running three training configurations and comparing their win rate curves against the HeuristicBot as the evaluation benchmark:

| Run | Opponent Epsilon | Description |
|-----|-----------------|-------------|
| A   | Fixed 0.5       | High randomness throughout |
| B   | Fixed 0.2       | Moderate, near-greedy throughout |
| C   | Decaying 0.5→0.05 | Annealed on same schedule as agent |

**Hypothesis:** Run C will produce the highest final win rate against the HeuristicBot because the opponent difficulty scales naturally with the agent's improving policy. Early in training the agent faces an explorable opponent; late in training it faces a near-optimal one.

**Secondary experiment:** If Run C confirms the hypothesis, a second round of runs tests the effect of the decay rate:

| Run | Decay Schedule | Notes |
|-----|---------------|-------|
| C1  | Fast decay (matches agent epsilon decay) | Opponent hardens quickly |
| C2  | Slow decay (half the rate of agent) | Opponent stays easier longer |
| C3  | Step decay (drops at promotion events) | Hardness tied to promotions |

The best-performing schedule from this round becomes the default for all subsequent self-play phases.

### Implementation

The frozen opponent is loaded and called as follows. The epsilon value is passed in at evaluation time rather than baked into the checkpoint, so the same saved weights can be re-evaluated at any epsilon:

```python
def frozen_opponent_action(model, state, epsilon, device):
    """Run frozen opponent with a fixed or scheduled epsilon."""
    spatial, scalars = state.get_observation()
    legal_mask = state.get_legal_mask()

    if random.random() < epsilon:
        legal_indices = np.where(legal_mask)[0]
        return int(np.random.choice(legal_indices))

    model.eval()
    with torch.no_grad():
        spatial_t  = torch.tensor(spatial).unsqueeze(0).to(device)
        scalars_t  = torch.tensor(scalars).unsqueeze(0).to(device)
        mask_t     = torch.tensor(legal_mask).unsqueeze(0).to(device)
        q = model(spatial_t, scalars_t, mask_t)
    return int(q.argmax(dim=1).item())
```

The opponent epsilon is tracked separately from the agent epsilon and updated according to whichever schedule is under test.

---

## 5. Phase 2: League Play

### Motivation

Frozen opponent self-play has a known weakness: the agent can overfit to the specific strategy of its most recent checkpoint. If the current frozen opponent consistently plays a wall-heavy defensive style (because that happened to be the policy when the checkpoint was saved), the agent may develop a counter-strategy that is brittle against other playstyles.

**League play** addresses this by maintaining a pool of past checkpoints at different skill levels and sampling opponents from the pool during training. The agent is exposed to diverse strategies simultaneously.

This approach is directly inspired by AlphaStar's league training [Vinyals et al. 2019], which maintained a population of agents at different training stages and matched them against each other to prevent strategy collapse.

### Pool Construction

Checkpoints are saved at every promotion event (Section 3). The league pool at any given time contains:

```
pool = [
    checkpoint_1,    # oldest, weakest
    checkpoint_2,
    ...
    checkpoint_k,    # most recent, strongest (current frozen opponent)
]
```

At the start of each episode, the opponent is sampled from the pool according to a weighted distribution that favors recent checkpoints but does not exclude older ones:

```python
# exponential weighting: recent checkpoints sampled more often
weights = [exp(i * LEAGUE_RECENCY_BIAS) for i in range(len(pool))]
weights = [w / sum(weights) for w in weights]
opponent_checkpoint = random.choices(pool, weights=weights)[0]
```

`LEAGUE_RECENCY_BIAS` is a hyperparameter controlling how strongly to favor recent opponents. A value of 0 produces uniform sampling; a value of 1.0 heavily weights the most recent checkpoint.

### When to Introduce League Play

League play is computationally more expensive than frozen opponent training because multiple checkpoints must be loaded and managed. We introduce it only after the agent has demonstrated stable improvement under frozen opponent self-play — concretely, after three consecutive promotion events without regression.

```
Phase 1: Frozen opponent only
          --> agent promotes 3 times without regressing

Phase 2: League play introduced
          --> pool begins accumulating checkpoints
          --> opponent sampled from pool each episode

Phase 3: Evaluate vs HeuristicBot + RandomBot periodically
          --> verify generalization is holding across opponent types
```

---

## 6. Evaluation Protocol

All evaluation runs use the HeuristicBot as the benchmark opponent regardless of what the agent is currently training against. This provides a consistent external reference point across all training phases.

```
Every EVAL_FREQ steps during self-play:
    win_rate_vs_heuristic = evaluate(online_net, n=200, opponent=HeuristicBot)
    win_rate_vs_frozen    = evaluate(online_net, n=200, opponent=frozen_opponent)
    mean_episode_length   = average game length over eval episodes

Log all three metrics to W&B.
```

Logging win rate against the frozen opponent tracks self-play progress. Logging win rate against the HeuristicBot tracks whether self-play is producing genuine generalization or just overfitting to the current frozen checkpoint.

If win rate vs. HeuristicBot *drops* while win rate vs. frozen opponent *rises*, that is a signal of overfitting to the self-play opponent — the agent is learning to exploit specific weaknesses of the frozen checkpoint rather than improving at Quoridor generally. The response is to increase opponent pool diversity (move to league play earlier) or raise the frozen opponent's epsilon temporarily.

---

## 7. Algorithm Roadmap

| Phase | Opponent | Epsilon Schedule | Trigger to Advance |
|-------|----------|-----------------|-------------------|
| 0 (done) | RandomBot | Agent decays 1.0→0.05 | 80% win rate |
| 1 (done) | HeuristicBot | Agent decays, opponent fixed | 70% win rate |
| 2 (next) | Frozen self (epsilon experiments) | Per Section 4 | 3 promotions, stable |
| 3 | League pool | Best schedule from Phase 2 | Time permitting |
| 4 | PPO / self-play | N/A — different algorithm | Final evaluation |

---

## 8. References

- Silver et al. (2017). *Mastering Chess and Shogi by Self-Play with a General
  Reinforcement Learning Algorithm.* arXiv:1712.01815.
  https://arxiv.org/abs/1712.01815

- Silver et al. (2017). *Mastering the Game of Go without Human Knowledge.*
  Nature 550, 354–359.
  https://www.nature.com/articles/nature24270

- Vinyals et al. (2019). *Grandmaster level in StarCraft II using multi-agent
  reinforcement learning.* Nature 575, 350–354.
  (AlphaStar — source of league play concept)

- Tokic, M. (2010). *Adaptive ε-greedy Exploration in Reinforcement Learning
  Based on Value Differences.* KI 2010: Advances in Artificial Intelligence.
  (Motivation for adaptive rather than fixed opponent epsilon)

- Heinrich & Silver (2016). *Deep Reinforcement Learning from Self-Play in
  Imperfect-Information Games.* NIPS Deep RL Workshop.
  (Neural Fictitious Self-Play — theoretical foundation for frozen opponent pool)

---

*Quoridor Deep RL Project — Self-Play Phase — March 2026*
*Author: Dreshta Boghra*