# PyTorch Quick Reference

Notes on the PyTorch concepts we actually use in this project.

## Tensors
- `torch.zeros(B, 137, dtype=torch.bool)` — preallocate masks
- `torch.tensor(np_array)` — convert numpy to torch
- `.unsqueeze(0)` — add batch dim for single inference
- `.to(device)` — move to mps/cpu

## Our model structure
```
Conv2d(4, 32, 3, padding=1) → BatchNorm2d → ReLU
Conv2d(32, 64, 3, padding=1) → BatchNorm2d → ReLU
Flatten → concat scalars → Linear(5186, 256) → ReLU → Linear(256, 137)
```
padding=1 with kernel=3 preserves the 9x9 spatial dims.

## Action masking
```python
q_values = q_values.masked_fill(~legal_mask, -1e9)
```
Sets illegal actions to -inf so argmax never picks them.
Use `masked_fill` instead of direct indexing to keep autograd happy.

## Training loop essentials
- `model.train()` / `model.eval()` — matters because of BatchNorm
- `torch.no_grad()` — skip gradient tracking for inference/targets
- `loss.backward()` → `optimizer.step()` — standard gradient descent
- `clip_grad_norm_(params, 10.0)` — prevent exploding gradients
- `target_net.load_state_dict(online_net.state_dict())` — hard target update

## Double DQN
Online net picks the best next action, target net evaluates it.
Fixes the Q-value overestimation we saw in our first run (Q > 1.0).

## Huber loss
`F.huber_loss(pred, target, delta=1.0)` — linear for large errors,
quadratic for small ones. Prevents the loss spikes we got with MSE.

## Reproducibility
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

## Device
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```
MPS = Apple Silicon GPU. Way faster than CPU for our conv layers.
