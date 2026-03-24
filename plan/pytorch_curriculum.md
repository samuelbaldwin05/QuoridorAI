# PyTorch Learning Curriculum — QuoridorAI Edition

> **How to use this file**
> Open it in any IDE with an AI assistant (Cursor, VS Code + Copilot, etc.) and work
> through each module in order. Ask the AI to help you run the exercises, explain
> concepts further, or generate additional practice problems. Every exercise uses
> Quoridor-shaped tensors so the numbers you see here are the same numbers you'll
> see in the real codebase.
>
> **Prerequisites:** Python 3.10+, PyTorch installed (`pip install torch`).

---

## Module Map

| # | Module | Key Concepts |
|---|--------|-------------|
| 1 | Tensors | Creation, shape, dtype, bool masks, indexing |
| 2 | Building a Neural Network | nn.Module, Sequential, Conv2d, BatchNorm2d, ReLU, Flatten, Linear |
| 3 | Forward Pass & Tensor Ops | torch.cat, masked_fill, argmax, mean, detach |
| 4 | Training Loop Mechanics | train/eval mode, no_grad, backward, grad, Adam, clip_grad_norm_ |
| 5 | Reproducibility & Checkpointing | manual_seed, Generator, state_dict, load_state_dict, allclose |

---

---

## Module 1 — Tensors

### Concept

A **tensor** is PyTorch's core data structure — a multi-dimensional array of numbers,
similar to a NumPy array, but with two critical extras: it can live on a GPU, and it
tracks the operations performed on it so gradients can be computed automatically.

Everything in a neural network — inputs, weights, outputs, gradients — is a tensor.
Understanding how to create, inspect, and index tensors is the foundation for
everything else in this curriculum.

---

### 1.1 Creating Tensors

```python
import torch

# torch.zeros — all-zero tensor of a given shape
# Used in replay_buffer.py to preallocate storage for legal masks
legal_mask = torch.zeros(4, 137, dtype=torch.bool)   # shape (4, 137), all False

# torch.rand — uniform random values in [0, 1)
# Used in tests to build fake board observations
spatial = torch.rand(4, 4, 9, 9)   # shape (B=4, channels=4, H=9, W=9)
scalars = torch.rand(4, 2)         # shape (B=4, 2 scalar features)

# torch.randint — random integers in [low, high)
n_legal = torch.randint(1, 137, (1,)).item()  # random count, converted to Python int

# torch.randperm — a random permutation of integers 0..n-1
# Used in tests to randomly select which actions are legal
perm = torch.randperm(137)[:n_legal]   # n_legal random action indices

# torch.arange — a range tensor (like Python's range(), but returns a tensor)
action_indices = torch.arange(137)     # tensor([0, 1, 2, ..., 136])
```

**Quoridor connection:**
- `torch.zeros(..., dtype=torch.bool)` creates the legal action mask used throughout the env and replay buffer.
- `torch.rand(B, 4, 9, 9)` is the shape of every board observation fed into the DQN.

---

### 1.2 Shape and dtype

```python
t = torch.rand(4, 4, 9, 9)

print(t.shape)        # torch.Size([4, 4, 9, 9])
print(t.shape[0])     # 4  — batch size
print(t.dtype)        # torch.float32 (default)

mask = torch.zeros(4, 137, dtype=torch.bool)
print(mask.dtype)     # torch.bool
```

Shape and dtype are the two most common things to check when debugging. When a tensor
operation fails, mismatched shapes or dtypes are almost always the cause.

---

### 1.3 Boolean Masks and Indexing

```python
import torch

q_values = torch.tensor([1.5, -2.0, 0.8, 3.1, -0.5])
legal    = torch.tensor([True, False, True, True, False])

# Select only legal Q-values
print(q_values[legal])      # tensor([1.5, 0.8, 3.1])

# Select only illegal Q-values (note the ~ negation operator)
print(q_values[~legal])     # tensor([-2.0, -0.5])

# Check: are all illegal Q-values at the penalty value?
penalty = -1e9
q_values[~legal] = penalty
print((q_values[~legal] == penalty).all())   # tensor(True)
```

**Quoridor connection:**
In `test_dqn_model.py`, the tests verify that all illegal actions have Q-values of
exactly `-1e9` using this same boolean indexing pattern.

---

### Exercise 1

Build a fake batch of Quoridor observations and a random legal mask:

1. Create `spatial` of shape `(4, 4, 9, 9)` with random float values.
2. Create `scalars` of shape `(4, 2)` with random float values in `[0, 1]` (representing
   wall fractions).
3. Create `legal_mask` of shape `(4, 137)`, all `False` initially.
4. For each of the 4 batch items, randomly set between 1 and 10 entries in that row to `True`.
5. Print the shape, dtype, and number of legal actions per batch item.

**Expected:** shapes match above, dtype is `torch.bool`, each row count is between 1 and 10.

---

### Checkpoint Question 1

> `torch.zeros(50_000, 4, 9, 9, dtype=torch.float32)` — how many bytes does this tensor
> occupy in memory? Show your calculation. Why does the replay buffer comment in
> `config.py` matter for CPU-only training?

---

**Further reading:** https://pytorch.org/docs/stable/tensors.html

---
---

## Module 2 — Building a Neural Network

### Concept

PyTorch builds neural networks from **modules** — composable building blocks that each
know how to: (1) do a forward computation, and (2) store their own learnable parameters.

`nn.Module` is the base class every network inherits from. You define the layers in
`__init__` and the data flow through them in `forward`. PyTorch automatically tracks
all parameters (weights, biases) registered inside the module.

---

### 2.1 nn.Module

```python
import torch
import torch.nn as nn

class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()          # REQUIRED: initializes the Module bookkeeping
        self.fc = nn.Linear(10, 5)  # registers 'fc' as a parameter-bearing layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

net = TinyNet()
x   = torch.rand(3, 10)   # batch of 3 inputs, each of size 10
out = net(x)              # calls forward() automatically
print(out.shape)           # torch.Size([3, 5])
```

**Quoridor connection:**
`DQNModel` in `agents/dqn_model.py` inherits from `nn.Module` and defines `__init__`
(architecture) and `forward` (inference + action masking).

---

### 2.2 nn.Sequential

```python
# nn.Sequential chains modules in order — output of one feeds into the next.
# Use it to group layers that always run together.

encoder = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
)

x   = torch.rand(4, 10)
out = encoder(x)
print(out.shape)   # torch.Size([4, 16])
```

**Quoridor connection:**
`DQNModel` uses two `nn.Sequential` blocks: `self.conv` (the spatial encoder) and
`self.fc` (the late-fusion head).

---

### 2.3 nn.Conv2d

```python
# nn.Conv2d(in_channels, out_channels, kernel_size, padding)
# Slides a small filter over a 2D grid, detecting local spatial patterns.
# padding=1 with kernel_size=3 preserves spatial dimensions: (9-3+2)/1+1 = 9

conv = nn.Conv2d(4, 32, kernel_size=3, padding=1)

x   = torch.rand(4, 4, 9, 9)   # (B, in_channels, H, W)
out = conv(x)
print(out.shape)                 # torch.Size([4, 32, 9, 9])  — H and W unchanged
```

**Quoridor connection:**
The DQN uses two Conv2d layers to detect spatial patterns in wall and pawn positions.
Both use `padding=1` so the 9×9 board dimensions are preserved through both layers.

---

### 2.4 nn.BatchNorm2d

```python
# BatchNorm2d normalizes the output of a conv layer across the batch.
# Stabilizes training by keeping activations in a well-behaved range.
# Must receive the same channel count as the preceding conv layer.

bn  = nn.BatchNorm2d(32)
x   = torch.rand(4, 32, 9, 9)
out = bn(x)
print(out.shape)   # torch.Size([4, 32, 9, 9])  — shape unchanged

# IMPORTANT: BatchNorm behaves differently in train vs eval mode.
# See Module 4 for details on model.train() and model.eval().
```

---

### 2.5 nn.ReLU, nn.Flatten, nn.Linear

```python
relu    = nn.ReLU()
flatten = nn.Flatten()
fc      = nn.Linear(5186, 256)

# ReLU: max(0, x) — kills negative activations, introducing non-linearity
x = torch.tensor([-1.0, 0.0, 2.5])
print(relu(x))   # tensor([0., 0., 2.5])

# Flatten: collapses all dimensions after the batch dimension
x   = torch.rand(4, 64, 9, 9)
out = flatten(x)
print(out.shape)   # torch.Size([4, 5184])   — 64 * 9 * 9 = 5184

# Linear: fully connected layer — matrix multiply + bias
x   = torch.rand(4, 5186)
out = fc(x)
print(out.shape)   # torch.Size([4, 256])
```

**Quoridor connection:**
These three are used back-to-back inside `DQNModel.conv`: Flatten converts the
`(B, 64, 9, 9)` conv output into `(B, 5184)`, then scalar wall counts are
concatenated to make `(B, 5186)`, and Linear maps that to Q-values.

---

### Exercise 2

Build a minimal version of the DQN spatial encoder as a standalone `nn.Sequential`:

1. Create `encoder = nn.Sequential(...)` with:
   - Conv2d: 4 → 32 channels, kernel 3, padding 1
   - BatchNorm2d(32)
   - ReLU
   - Conv2d: 32 → 64 channels, kernel 3, padding 1
   - BatchNorm2d(64)
   - ReLU
   - Flatten
2. Pass a random input of shape `(4, 4, 9, 9)` through it.
3. Print the output shape.

**Expected:** `torch.Size([4, 5184])`

---

### Checkpoint Question 2

> Why does `super().__init__()` have to be called in every `nn.Module` subclass?
> What breaks if you forget it?

---

**Further reading:** https://pytorch.org/docs/stable/nn.html

---
---

## Module 3 — Forward Pass & Tensor Operations

### Concept

With layers defined, the forward pass is where input data actually flows through the
network. Three operations are central to the DQN's forward pass:

1. **`torch.cat`** — fuses the spatial encoding with scalar wall counts (late fusion)
2. **`.masked_fill`** — zeroes out illegal actions before the agent can select them
3. **`.argmax`** — selects the action with the highest Q-value (greedy policy)

---

### 3.1 torch.cat

```python
import torch

# torch.cat concatenates tensors along a given dimension.
# dim=1 means "stack along the feature dimension" (batch dim=0 stays intact).

spatial_enc = torch.rand(4, 5184)   # flattened conv output
wall_counts = torch.rand(4, 2)      # [my_walls/10, opp_walls/10]

fused = torch.cat([spatial_enc, wall_counts], dim=1)
print(fused.shape)   # torch.Size([4, 5186])
```

**Quoridor connection:**
`dqn_model.py:111` — this is the late fusion step. The CNN output and scalar features
are concatenated here before the final FC layers produce Q-values.

---

### 3.2 .masked_fill

```python
# .masked_fill(mask, value) — replaces positions where mask is True with value.
# The mask argument selects WHERE to fill, so ~legal_mask selects illegal positions.

q_values   = torch.tensor([[2.1, -0.5, 1.8, 0.3, 3.0]])
legal_mask = torch.tensor([[True, False, True, False, True]])

masked_q = q_values.masked_fill(~legal_mask, -1e9)
print(masked_q)
# tensor([[ 2.1000e+00, -1.0000e+09,  1.8000e+00, -1.0000e+09,  3.0000e+00]])

# Now argmax will never select an illegal action
print(masked_q.argmax(dim=1))   # tensor([4])  — highest legal Q-value
```

**Why `-1e9` and not `-inf`?**
`-inf` can cause NaN when used inside loss computations. `-1e9` is small enough to
never be selected in practice, but doesn't produce NaN.

**Quoridor connection:**
`dqn_model.py:117` — every forward pass ends with this mask. It is applied *inside*
the model so callers can never accidentally skip it.

---

### 3.3 .argmax, .mean, .detach

```python
import torch

q = torch.tensor([[1.0, 5.0, -2.0, 3.0]])   # (1, 4) batch of Q-values

# .argmax(dim) — index of the maximum value along a dimension
action = q.argmax(dim=1)
print(action)           # tensor([1])  — action 1 has the highest Q-value

# .mean() — average all values (scalar output)
avg_q = q.mean()
print(avg_q)            # tensor(1.7500)

# .detach() — creates a new tensor that shares data but is excluded from the
# computation graph. Use it when you want a value for logging/comparison
# without accidentally backpropagating through it.
q_stopped = q.detach()
print(q_stopped.requires_grad)   # False
```

**Quoridor connection:**
- `.argmax(dim=1)` is greedy action selection in epsilon-greedy policy
- `.mean()` is used in `train/loss` logging (`train/mean_q_value`)
- `.detach()` is used in tests to compare train-mode vs eval-mode Q-values without
  polluting the gradient graph

---

### Exercise 3

Simulate one DQN forward pass from scratch (no `DQNModel` class needed):

1. Create `spatial` `(4, 4, 9, 9)` random, `scalars` `(4, 2)` random.
2. Create a fake encoder: `nn.Sequential(nn.Flatten(), nn.Linear(4*9*9, 5184))` — this
   is not the real architecture but produces the right shape for practice.
3. Pass `spatial` through the encoder, then `torch.cat` the result with `scalars`.
4. Pass the fused tensor through `nn.Linear(5186, 137)` to get raw Q-values.
5. Create a `legal_mask` `(4, 137)` bool tensor where the first 10 actions are legal.
6. Apply `.masked_fill` to block illegal actions.
7. Select the greedy action for each batch item with `.argmax(dim=1)`.
8. Verify all 4 chosen actions are in `[0, 9]`.

---

### Checkpoint Question 3

> Why is masking applied *inside* `DQNModel.forward()` rather than in the training
> loop that calls it? What could go wrong if it were applied outside?

---

**Further reading:** https://pytorch.org/docs/stable/torch.html#torch.cat

---
---

## Module 4 — Training Loop Mechanics

### Concept

Training a neural network means repeatedly:
1. Running a forward pass to get predictions
2. Computing a loss (how wrong the predictions are)
3. Running a backward pass to compute gradients (which direction to nudge each weight)
4. Updating weights with an optimizer

PyTorch makes this explicit — nothing happens automatically. You call `.backward()`,
then the optimizer's `.step()`. This transparency is what makes debugging tractable.

---

### 4.1 model.train() and model.eval()

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.BatchNorm1d(32),
    nn.ReLU(),
    nn.Linear(32, 5),
)

# model.train() — BatchNorm uses *batch* statistics (mean/var computed from the
#                 current mini-batch). Correct during gradient updates.
model.train()
x   = torch.rand(64, 10)   # 64-sample batch
out = model(x)             # BatchNorm normalizes using this batch's statistics

# model.eval() — BatchNorm uses *running* statistics (accumulated across training).
#                Correct during single-sample inference or evaluation.
model.eval()
x_single = torch.rand(1, 10)
with torch.no_grad():
    out_single = model(x_single)   # stable output, no batch-stat noise
```

**Why this matters for DQN:**
During epsilon-greedy action selection, you pass a single observation (batch size = 1).
If the model is still in `.train()` mode, BatchNorm normalizes using only that one
sample, producing very different (noisier) Q-values than it produces for a 64-sample
training batch. Always call `model.eval()` before inference and `model.train()` before
gradient updates.

```python
# Correct DQN action selection pattern:
model.eval()
with torch.no_grad():
    q_values = model(spatial, scalars, legal_mask)
action = q_values.argmax(dim=1).item()
model.train()
```

---

### 4.2 torch.no_grad()

```python
import torch
import torch.nn as nn

model = nn.Linear(4, 2)
x = torch.rand(3, 4)

# Without no_grad: PyTorch builds a computation graph to enable .backward()
out = model(x)
print(out.requires_grad)   # True — graph is being tracked

# With no_grad: no graph is built — faster and uses less memory
with torch.no_grad():
    out = model(x)
print(out.requires_grad)   # False

# Use torch.no_grad() whenever you don't need gradients:
#   - Action selection during gameplay
#   - Computing the target network's Q-values
#   - Evaluation / win rate measurement
```

---

### 4.3 .backward() and .grad

```python
import torch
import torch.nn as nn

model = nn.Linear(5186, 137)
x     = torch.rand(4, 5186)

# Forward pass
q_values = model(x)

# Compute a scalar loss (mean Q-value — not a real DQN loss, just for illustration)
loss = q_values.mean()

# Backward pass: computes d(loss)/d(param) for every parameter in the graph
loss.backward()

# Inspect gradients
for name, param in model.named_parameters():
    print(f"{name}: grad shape = {param.grad.shape}, grad norm = {param.grad.norm():.4f}")
```

**In the DQN training loop**, the loss is the TD error:
```python
# Pseudocode — actual implementation goes in train_dqn.py
target = reward + GAMMA * target_net(next_state).max(dim=1).values * (1 - done)
current_q = online_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
loss = nn.functional.mse_loss(current_q, target.detach())

optimizer.zero_grad()   # clear gradients from previous step
loss.backward()         # compute new gradients
```

---

### 4.4 torch.optim.Adam and clip_grad_norm_

```python
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

model     = nn.Linear(5186, 137)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

x    = torch.rand(4, 5186)
loss = model(x).mean()

optimizer.zero_grad()                          # 1. clear old gradients
loss.backward()                                # 2. compute new gradients
clip_grad_norm_(model.parameters(), max_norm=10.0)  # 3. clip to prevent explosion
optimizer.step()                               # 4. apply update
```

**Why gradient clipping?**
Early in DQN training, Q-value estimates are wildly inaccurate, causing large TD errors
and therefore large gradients. Without clipping, one bad batch can shove the weights
far off course. Clipping caps the maximum gradient norm at `10.0` — large enough to
allow real learning, small enough to prevent catastrophic updates.

---

### Exercise 4

Write a minimal training step:

1. Create a small `nn.Sequential` model: `Linear(5186, 256) → ReLU → Linear(256, 137)`.
2. Create an `Adam` optimizer with `lr=1e-4`.
3. Create a fake batch: `x = torch.rand(64, 5186)`, `target = torch.rand(64, 137)`.
4. Compute predictions, then compute MSE loss vs the target using `nn.functional.mse_loss`.
5. Run the full update loop: `zero_grad → backward → clip_grad_norm_(norm=10.0) → step`.
6. After the update, verify that every parameter has a non-None `.grad`.
7. Run the step a second time and verify the loss changes.

---

### Checkpoint Question 4

> Why must `optimizer.zero_grad()` be called before `loss.backward()`?
> What happens to gradient values if you forget it?

---

**Further reading:** https://pytorch.org/docs/stable/optim.html

---
---

## Module 5 — Reproducibility & Checkpointing

### Concept

Two things every RL experiment needs:
1. **Reproducibility** — the same seed produces the same result, making bugs debuggable
   and results comparable across runs
2. **Checkpointing** — saving and loading model weights so you don't lose progress and
   can resume from the best checkpoint

---

### 5.1 torch.manual_seed() and torch.Generator

```python
import torch

# torch.manual_seed — sets the global PyTorch RNG seed.
# Call this once at the start of every training run.
torch.manual_seed(42)
a = torch.rand(3)

torch.manual_seed(42)
b = torch.rand(3)

print(torch.allclose(a, b))   # True — same seed, same values

# torch.Generator — a local, independent RNG.
# Use when you want reproducible randomness in one place without affecting the global seed.
rng = torch.Generator()
rng.manual_seed(99)
x = torch.rand(3, generator=rng)
y = torch.rand(3, generator=rng)   # different from x (advances same generator)
```

**Quoridor connection:**
`config.py` defines `SEED = 42`. The training script seeds `torch`, `numpy`, and the
environment with this value at startup. The tests use `torch.Generator` to produce
deterministic fake inputs without disturbing the global seed.

---

### 5.2 model.state_dict() and model.load_state_dict()

```python
import torch
import torch.nn as nn

model = nn.Linear(5186, 137)

# state_dict() returns an ordered dict of all parameter tensors (weights + biases)
sd = model.state_dict()
for key, tensor in sd.items():
    print(f"{key}: {tensor.shape}")
# weight: torch.Size([137, 5186])
# bias:   torch.Size([137])

# Save to disk
torch.save(sd, "checkpoint.pt")

# Load from disk
new_model = nn.Linear(5186, 137)
new_model.load_state_dict(torch.load("checkpoint.pt"))

# Copying weights between two identical models (used for target network initialization)
online_net = nn.Linear(5186, 137)
target_net = nn.Linear(5186, 137)
target_net.load_state_dict(online_net.state_dict())
target_net.eval()   # target net stays in eval mode permanently
```

**Quoridor connection:**
In the DQN training loop, `target_net.load_state_dict(online_net.state_dict())` is
called every `TARGET_UPDATE_FREQ` steps to perform a hard target network update.
The same pattern is used to save the best checkpoint when win rate improves.

---

### 5.3 torch.allclose

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([1.0, 2.0, 3.0])
c = torch.tensor([1.0, 2.0, 3.0001])

print(torch.allclose(a, b))                      # True — exact match
print(torch.allclose(a, c))                      # False — small but above default tol
print(torch.allclose(a, c, atol=1e-3))           # True — within tolerance
```

**Quoridor connection:**
`test_dqn_model.py` uses `torch.allclose` to verify that the model produces identical
Q-values in train mode vs eval mode *after* the running statistics have been warmed up
(i.e., once BatchNorm's running mean/var have been estimated from enough batches).

---

### Exercise 5

Implement the target network update cycle:

1. Create `online_net = nn.Sequential(nn.Linear(5186, 256), nn.ReLU(), nn.Linear(256, 137))`.
2. Create `target_net` with the same architecture.
3. Copy `online_net` weights into `target_net` using `load_state_dict`.
4. Verify the weights are identical: iterate over both `state_dict()`s and use
   `torch.allclose` on each parameter tensor.
5. Run one gradient update on `online_net` (fake loss: `online_net(torch.rand(4, 5186)).mean()`).
6. Verify the weights are now *different* (the update changed `online_net` but not `target_net`).
7. Save `online_net.state_dict()` to `"test_checkpoint.pt"`, then load it into a fresh
   `nn.Sequential` with the same architecture and verify weights match.

---

### Checkpoint Question 5

> After calling `target_net.load_state_dict(online_net.state_dict())`, why must
> `target_net.eval()` be called immediately after? What would happen during the next
> TD target computation if you forgot?

---

**Further reading:** https://pytorch.org/docs/stable/notes/serialization.html

---
---

## Quick Reference — All PyTorch Items in This Repo

| Item | Module | Where used in repo |
|------|--------|--------------------|
| `torch.rand` | 1 | `tests/test_dqn_model.py` — fake observations |
| `torch.zeros` | 1 | `replay_buffer.py` — preallocate arrays; `tests/` — zero legal masks |
| `torch.randint` | 1 | `tests/test_dqn_model.py` — random action counts |
| `torch.randperm` | 1 | `tests/test_dqn_model.py` — random legal action selection |
| `torch.arange` | 1 | `tests/test_dqn_model.py` — sequential action indices |
| `torch.Generator` | 5 | `tests/test_dqn_model.py` — deterministic test RNG |
| `.manual_seed` | 5 | `config.py` (SEED=42); `tests/` — generator seeding |
| `.shape` | 1 | Throughout tests — shape assertions |
| `.dtype` | 1 | Throughout tests — dtype assertions |
| `torch.bool` | 1 | `replay_buffer.py`, `dqn_model.py` — legal mask dtype |
| `nn.Module` | 2 | `dqn_model.py:47` — DQNModel base class |
| `super().__init__()` | 2 | `dqn_model.py:57` |
| `nn.Sequential` | 2 | `dqn_model.py:63, 81` — conv and fc blocks |
| `nn.Conv2d` | 2 | `dqn_model.py:64, 69` — spatial encoder |
| `nn.BatchNorm2d` | 2 | `dqn_model.py:65, 70` — training stabilization |
| `nn.ReLU` | 2 | `dqn_model.py:66, 71, 83` — activations |
| `nn.Flatten` | 2 | `dqn_model.py:74` — spatial → vector |
| `nn.Linear` | 2 | `dqn_model.py:82, 86` — late-fusion head |
| `torch.cat` | 3 | `dqn_model.py:111` — late fusion |
| `.masked_fill` | 3 | `dqn_model.py:117` — illegal action masking |
| `.argmax` | 3 | `tests/test_dqn_model.py` — greedy action; `train_dqn.py` (planned) |
| `.mean` | 3 | `tests/` — fake loss; `train_dqn.py` — Q-value logging |
| `.detach` | 3 | `tests/` — prevent unintended gradient flow |
| `model.train()` | 4 | `train_dqn.py` (planned) — before gradient updates |
| `model.eval()` | 4 | `train_dqn.py` (planned) — before inference and evaluation |
| `torch.no_grad()` | 4 | `tests/`, `train_dqn.py` (planned) — inference contexts |
| `.backward()` | 4 | `tests/`, `train_dqn.py` (planned) — backpropagation |
| `.named_parameters()` / `.grad` | 4 | `tests/` — gradient inspection |
| `torch.optim.Adam` | 4 | `train_dqn.py` (planned) — optimizer |
| `clip_grad_norm_` | 4 | `train_dqn.py` (planned) — gradient clipping |
| `model.state_dict()` | 5 | `train_dqn.py` (planned) — target update, checkpointing |
| `model.load_state_dict()` | 5 | `train_dqn.py` (planned) — target update, resume from checkpoint |
| `torch.manual_seed()` | 5 | `train_dqn.py` (planned) — reproducibility |
| `torch.allclose` | 5 | `tests/` — numerical equality checks |

---

*Curriculum authored: March 2026 | QuoridorAI DQN Phase*
