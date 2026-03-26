# Quoridor Deep RL: Design Decisions and Rationale

A beginner-friendly record of the key architectural choices made before writing any code.
This document covers the theory, the options we considered, and why we made each decision.

---

## Table of Contents

1. [What Problem Are We Solving?](#1-what-problem-are-we-solving)
2. [Key Concepts](#2-key-concepts)
3. [State Representation](#3-state-representation)
4. [Action Space](#4-action-space)
5. [Network Architecture](#5-network-architecture)
6. [The DQN Algorithm](#6-the-dqn-algorithm)
7. [Training Curriculum](#7-training-curriculum)
8. [Algorithm Roadmap](#8-algorithm-roadmap)

---

## 1. What Problem Are We Solving?

In supervised machine learning, you train on labeled data where someone already knows the right answer.
In Quoridor, no labeled dataset exists. The only feedback the agent ever receives is whether it won
or lost, sometimes 50 moves after the decisive moment.

**Reinforcement Learning (RL)** is the framework built for exactly this kind of delayed, sparse feedback.

### The Credit Assignment Problem

A game lasts roughly 50 moves. The agent must figure out which of those moves caused the outcome.
Was it the wall placed on move 12 that funneled the opponent? Or the aggressive pawn move on move 34?
This is called the **credit assignment problem** and it is the central challenge of RL.

### How the Agent Learns

```
Agent observes board state s
        |
        v
Neural network picks action a
        |
        v
Environment returns new state s' and reward r
        |
        v
Use (s, a, r, s') to improve the network
        |
        v
Repeat millions of times
```

---

## 2. Key Concepts

### Value Function V(s)

A neural network that predicts win probability from any board position. Instead of waiting until
the game ends to assign credit, we use changes in this prediction after each move as a per-move
training signal.

- Move increases V(s): probably a good move
- Move decreases V(s): probably a bad move

This idea is called **Temporal Difference (TD) learning**.

```
Before move:  V(s)  = 0.45  (45% chance of winning)
After move:   V(s') = 0.61  (61% chance of winning)

TD error = 0.61 - 0.45 = +0.16  --> good move
```

### Q-Values Q(s, a)

An extension of the value function. Instead of valuing a board state alone, Q(s, a) values a
specific action taken from a specific state.

| Concept | Question it answers | Limitation |
|---|---|---|
| V(s) | How good is this board position? | Cannot act without simulating future states |
| Q(s, a) | How good is this action from this position? | None: just pick argmax Q(s, a) |

The critical advantage of Q(s, a): the agent can pick the best move by simply choosing the
highest Q-value. No simulation of future states required. This is called **model-free** learning.

### The Bellman Equation

Q-values are defined recursively. The value of taking action a in state s equals the immediate
reward plus the discounted best future value:

```
Q(s, a) = r + gamma * max Q(s', a')
                       a'
```

- `r` is the immediate reward
- `gamma` (0.99 for Quoridor) is the discount factor, weighting future rewards slightly less than immediate ones
- `s'` is the resulting next state
- The `max` selects the best available action in the next state

For Quoridor, almost every move has r = 0. Only the final winning move has r = +1 and the
losing move has r = -1. The Bellman equation propagates this signal backwards through training.

---

## 3. State Representation

The neural network cannot see a board directly. It processes tensors of numbers. How the board
is encoded is one of the most consequential design decisions in the entire project. A poor
representation makes learning slow or impossible regardless of algorithm quality.

### Decision: Stacked 2D Channels

**Rejected: Flat 1D Vector**

Flattening the board into a list of numbers destroys spatial structure. Adjacent cells on the
board may end up far apart in the vector. The network must re-learn spatial relationships from
scratch, which is slow and wastes model capacity.

**Chosen: (4, 9, 9) Tensor**

Four stacked 9x9 grids, one per type of information. Think of it like an image with multiple
color channels (e.g. RGB), except our channels encode board features instead of colors.

```
Channel 0:  Current player pawn position    (1 where pawn is, 0 elsewhere)
Channel 1:  Opponent pawn position          (1 where pawn is, 0 elsewhere)
Channel 2:  Horizontal wall placements      (1 where wall exists)
Channel 3:  Vertical wall placements        (1 where wall exists)

Scalar features passed separately:
  - Current player walls remaining          (e.g. 0.7 for 7 out of 10)
  - Opponent walls remaining
```

A convolutional filter sliding across all 4 channels simultaneously can detect patterns like
"there is a horizontal wall directly above my pawn and the opponent is nearby," a genuinely
strategic signal that spans multiple channels.

### Decision: Current Player Perspective

**Rejected: Fixed Absolute Perspective**

Player 1 always at the bottom, Player 2 always at the top, with a "whose turn" flag. The same
strategic position looks completely different depending on which player is moving. The network
must learn all patterns twice, once per player orientation. This wastes capacity and slows training.

**Chosen: Always Normalize to Current Player**

The board is flipped vertically when it is Player 2's turn. Channel 0 always means "my pawn"
and Channel 1 always means "their pawn," regardless of which player is moving. The network
learns one universal set of strategic patterns.

```python
# Pseudocode for perspective normalization
if current_player == 2:
    board = flip_vertical(board)
    swap(channel_0, channel_1)          # swap pawn channels
    swap(walls_remaining_p1, walls_remaining_p2)
```

### Decision: Late Fusion for Scalar Features

Scalar information like walls remaining does not belong in a spatial channel. Filling an entire
9x9 grid with one repeated number is wasteful and semantically awkward.

**Chosen: Late Fusion Architecture**

Process spatial features and scalar features through separate pipelines, then concatenate
before the final output layers.

```
Board (4, 9, 9)  -->  CNN layers  -->  Flatten  -->  256-dim vector  --+
                                                                        +--> Concat --> Dense --> output
Scalars [w1, w2] -->  Dense layer -->  32-dim vector  -----------------+
```

---

## 4. Action Space

On each turn a player can move their pawn or place a wall.

```
5    pawn moves  (up, down, left, right, plus jump over adjacent opponent)
64   horizontal wall placements  (8 rows of edges x 8 column positions)
64   vertical wall placements    (same logic rotated)
---
133  total discrete actions
```

This is significantly larger than most classic RL benchmarks, which typically have fewer than
20 actions. The mixed nature of the space (movement vs. placement) adds additional complexity.

### Decision: Action Masking

At any given turn, many of those 133 actions are illegal. Two options:

| Option | Approach | Problem |
|---|---|---|
| Dynamic output size | Only output probabilities for legal moves | Output size changes every step, complicates architecture |
| Action masking | Output all 133, zero out illegal ones | Simple, stable, standard practice |

**Chosen: Action Masking**

Always output 133 probabilities. Before selecting an action, set illegal action logits to
negative infinity. After softmax, illegal actions have probability zero. Simple to implement
and compatible with all downstream algorithms.

---

## 5. Network Architecture

### Decision: CNN Backbone

A Convolutional Neural Network (CNN) slides small filters across the board, detecting local
spatial patterns at each layer. Deeper layers detect combinations of patterns, building up to
higher-level strategic concepts.

```
Input (4, 9, 9)
      |
Conv Layer 1 --> detects: walls near pawns, open cells, adjacency
      |
Conv Layer 2 --> detects: corridors, blocked paths, pawn proximity to goal
      |
Conv Layer 3 --> detects: strategic formations, trap setups, race conditions
      |
Flatten --> Late fusion with scalars --> Dense layers --> output
```

**Why not a Transformer?**

Transformers (used in LLMs) can attend to the entire board simultaneously, potentially
capturing long-range strategic dependencies more directly. However, they are harder to train,
more sensitive to hyperparameters, and would significantly complicate debugging during the
first RL training loop.

Upgrade path if needed later: CNN --> CNN with global pooling --> Transformer backbone.

---

## 6. The DQN Algorithm

Deep Q-Network (DQN) combines Q-learning with a neural network and two key innovations that
make training stable.

### The Core Loop

```
Initialize online network Q(s, a; theta) with random weights
Initialize target network Q(s, a; theta_minus) as a copy
Initialize empty replay buffer D

Loop:
  1. Observe state s
  2. With probability epsilon: take random action  (exploration)
     Otherwise: take argmax Q(s, a; theta)         (exploitation)
  3. Execute action, observe reward r and next state s'
  4. Store (s, a, r, s') in replay buffer D
  5. Sample random mini-batch from D
  6. Compute target:
       terminal state:     target = r
       non-terminal:       target = r + gamma * max Q(s', a'; theta_minus)
  7. Gradient descent on loss = (Q(s, a; theta) - target)^2
  8. Every N steps: copy theta --> theta_minus
```

### Innovation 1: Experience Replay

Rather than training on experiences in the order they occur, every transition (s, a, r, s')
is stored in a replay buffer (typically 100,000+ entries). Mini-batches are sampled randomly
during training.

**Why this matters for Quoridor:** Sequential experiences from the same game are correlated.
If moves 1 through 20 all occur on the left side of the board, gradient updates get skewed
toward patterns from that region. Random sampling ensures each mini-batch contains diverse,
uncorrelated board positions from many different games.

### Innovation 2: Target Network

Two copies of the network are maintained:

```
Online network  theta       <-- receives gradient updates every step
Target network  theta_minus <-- frozen, copied from online every N steps
```

**Why this matters:** If one network does both jobs, every gradient update shifts the training
targets simultaneously. The loss function ends up chasing a moving target. The frozen target
network provides a stable reference point for N steps at a time, analogous to measuring
something with a ruler that does not change length mid-measurement.

### Epsilon-Greedy Exploration

The agent starts with fully random actions and gradually shifts toward greedy exploitation:

```
Early training:   epsilon = 1.0   -->  pure exploration
Mid training:     epsilon = 0.5   -->  mix of exploring and exploiting
Late training:    epsilon = 0.05  -->  mostly exploit, small amount of random
```

This is conceptually identical to simulated annealing: controlled randomness that decreases
as the agent builds confidence. Without exploration, the agent follows its initial noisy
Q-values forever and never discovers better strategies.

---

## 7. Training Curriculum

### Decision: Train Against Fixed Heuristic Agent First

**The non-stationarity problem:** In self-play, both agents learn simultaneously. Experiences
collected early against a weak version of the opponent become misleading when sampled later,
because the opponent has since improved. This instability can prevent convergence.

The team has an existing heuristic-based agent that plays competently. We use it as the
initial training opponent.

**Benefits:**
- Environment becomes stationary (opponent does not change)
- Wins require genuine Quoridor skill, producing a meaningful reward signal
- All replay buffer experiences are against consistent opponent quality

**Risk:** The agent may overfit to the heuristic's specific weaknesses and fail against
human players or other agents.

**Mitigation:** Graduate to self-play once the agent achieves roughly 70 percent win rate
against the heuristic.

```
Phase 1: Train against fixed heuristic
          --> agent learns basic competent Quoridor
          --> achieves consistent wins against heuristic

Phase 2: Introduce self-play
          --> agent now generates meaningful self-play experience
          --> learns strategies beyond the heuristic's limitations

Phase 3: Evaluate against heuristic + humans periodically
          --> verify generalization is holding
```

---

## 8. Algorithm Roadmap

We do not jump straight to the most powerful algorithm. Each phase teaches something concrete
and fixes a real limitation of the previous approach.

| Phase | Algorithm | What It Teaches |
|---|---|---|
| Phase 1 (Now) | DQN | Q-values, experience replay, target networks, large action spaces |
| Phase 2 (Next) | PPO | Policy gradients, stable adversarial training |
| Phase 3 (Goal) | AlphaZero | MCTS + neural network, genuine long-horizon planning |

### Why This Progression?

**DQN** is a natural starting point. It directly learns Q-values and the replay buffer
and target network concepts are foundational to understanding all of RL.

**PPO** fixes DQN's core weakness in adversarial settings. DQN assumes a stationary
environment. Two agents training against each other creates non-stationarity that DQN
handles poorly. PPO's policy gradient approach is more stable in these settings.

**AlphaZero** is the end goal. It combines a neural network with Monte Carlo Tree Search
(MCTS) to explicitly plan ahead. This is what allows world-class play in Go, Chess, and
Shogi. Understanding why DQN and PPO fall short in long-horizon planning is what will
make AlphaZero's design make sense when we get there.

---

*Quoridor Deep RL Project -- Design Phase -- February 2026*