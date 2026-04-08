"""
ppo_bot.py — PPO checkpoint wrapper that implements the standard bot interface.

Wraps any trained PPO model variant so it can be used as a drop-in opponent
in watch_ppo.py, play_bot.py, or any evaluation harness that calls
bot.choose_action(game) and bot.reset().

Usage:
    from agents.ppo_bot import PPOBot
    bot = PPOBot("checkpoints/ppo_best.pt", model_type="bfs_resnet")
    action = bot.choose_action(game)   # → ("move", r, c) or ("fence", r, c, orient)
"""

import numpy as np
import torch

from agents.ppo_model import PPOModel
from agents.ppo_model_bfs import PPOModelBFS
from agents.ppo_model_bfs_resnet import PPOModelBFSResNet
from agents.ppo_model_resnet import PPOModelResNet
from quoridor.action_encoding import (
    index_to_action, FENCE_GRID, H_WALL_OFFSET, V_WALL_OFFSET,
)

# Maps flipped-space move action index → actual-space move action index.
# np.flipud negates dr but leaves dc unchanged, so:
#   up(0) ↔ down(1), left(2)/right(3) stay, NW(4)↔SW(6), NE(5)↔SE(7).
_MOVE_FLIP = [1, 0, 2, 3, 6, 7, 4, 5]


def _flip_legal_mask(mask: np.ndarray) -> np.ndarray:
    """Remap a legal mask from actual board coords to flipped (P1) coords."""
    flipped = np.zeros_like(mask)
    for flipped_idx, actual_idx in enumerate(_MOVE_FLIP):
        flipped[flipped_idx] = mask[actual_idx]
    for r in range(FENCE_GRID):
        actual_r = FENCE_GRID - 1 - r
        for c in range(FENCE_GRID):
            flipped[H_WALL_OFFSET + r * FENCE_GRID + c] = (
                mask[H_WALL_OFFSET + actual_r * FENCE_GRID + c]
            )
            flipped[V_WALL_OFFSET + r * FENCE_GRID + c] = (
                mask[V_WALL_OFFSET + actual_r * FENCE_GRID + c]
            )
    return flipped

# Maps --model names to (ModelClass, use_bfs). Mirrors the registry in train_ppo.py.
_MODEL_REGISTRY: dict = {
    "baseline":   (PPOModel,          False),
    "resnet":     (PPOModelResNet,     False),
    "bfs":        (PPOModelBFS,        True),
    "bfs_resnet": (PPOModelBFSResNet,  True),
}


class PPOBot:
    """
    Stateless bot wrapper around a trained PPO actor.

    The PPO policy is memoryless — each choose_action() call is a single
    forward pass with no hidden state. reset() is a no-op but is kept for
    interface compatibility with HeuristicBot and RandomBot.

    Parameters
    ----------
    checkpoint_path : str
        Path to a .pt file saved by train_ppo.py. Supports both full
        training checkpoints (dict with "model_state_dict" key) and
        raw state_dicts (from torch.save(model.state_dict(), path)).
    model_type : str
        Architecture name: "baseline", "resnet", "bfs", or "bfs_resnet".
        Must match the architecture used when the checkpoint was saved.
    device : str or torch.device, optional
        Inference device. Defaults to CPU — adequate for single-game eval.
    greedy : bool, optional
        If True, argmax over action probabilities (deterministic).
        If False, sample from the Categorical distribution (stochastic).
        Default: True — greedy play is standard for evaluation.
    """

    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "bfs_resnet",
        device: str | torch.device = "cpu",
        greedy: bool = True,
    ) -> None:
        if model_type not in _MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from: {list(_MODEL_REGISTRY.keys())}"
            )

        ModelClass, self.use_bfs = _MODEL_REGISTRY[model_type]
        self.device = torch.device(device)
        self.greedy = greedy
        self.model  = ModelClass().to(self.device)

        # Support both full training checkpoints and raw state_dicts.
        ckpt       = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def reset(self) -> None:
        """No per-episode state — PPO policy is stateless."""
        pass

    def choose_action(self, game) -> tuple:
        """
        Select an action for the current player.

        Parameters
        ----------
        game : QuoridorState
            Current game state. The action is chosen for game.turn.
            get_observation() returns a perspective-normalized view (flipped
            for P1). The legal mask must be flipped to match, then the decoded
            action must be un-flipped back to actual board coordinates.

        Returns
        -------
        tuple
            Either ("move", row, col) with absolute board coordinates,
            or ("fence", row, col, orientation).
        """
        flip = (game.turn == 1)

        spatial, scalars = game.get_observation(use_bfs=self.use_bfs)
        legal_mask       = game.get_legal_mask()
        # Flip legal mask to match the flipped observation for P1.
        if flip:
            legal_mask = _flip_legal_mask(legal_mask)

        spatial_t = torch.tensor(spatial).unsqueeze(0).to(self.device)     # (1, C, 9, 9)
        scalars_t = torch.tensor(scalars).unsqueeze(0).to(self.device)     # (1, 2)
        mask_t    = torch.tensor(legal_mask).unsqueeze(0).to(self.device)  # (1, 137)

        # bfs_resnet returns (dist, value, aux_pred); other models return (dist, value).
        # Unpack with * to handle both.
        with torch.no_grad():
            dist, *_ = self.model(spatial_t, scalars_t, mask_t)
            if self.greedy:
                action_idx = int(dist.probs.argmax(dim=-1).item())
            else:
                action_idx = int(dist.sample().item())

        return self._decode(action_idx, game, flip)

    def _decode(self, idx: int, game, flip: bool = False) -> tuple:
        """
        Convert an action index to a concrete game action tuple.

        When flip=True (P1), the model selected in flipped space — move
        directions must be negated (dr → -dr) and fence rows must be
        un-flipped (r → FENCE_GRID-1-r) to get actual board coordinates.
        """
        action = index_to_action(idx)

        if action[0] == "fence":
            _, r, c, ori = action
            if flip:
                r = FENCE_GRID - 1 - r
            return ("fence", r, c, ori)

        # Resolve direction delta → absolute destination.
        dr, dc = action[1], action[2]
        # Flip negates row direction: model's "up" is actual "down" for P1.
        if flip:
            dr = -dr
        cur_r  = int(game.pos[game.turn, 0])
        cur_c  = int(game.pos[game.turn, 1])

        for dest_r, dest_c in game.get_valid_moves():
            if (np.sign(dest_r - cur_r) == np.sign(dr)
                    and np.sign(dest_c - cur_c) == np.sign(dc)):
                return ("move", dest_r, dest_c)

        # Should never reach here if the legal mask is correct.
        raise ValueError(
            f"PPOBot: no valid destination for direction ({dr}, {dc}) "
            f"from ({cur_r}, {cur_c}). Action index {idx} should have been masked illegal."
        )
