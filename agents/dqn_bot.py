"""
dqn_bot.py — DQN-based opponent for frozen self-play training.

Wraps a trained DQNModel and plays greedily (epsilon=0) using its Q-values.
Acts as a drop-in replacement for HeuristicBot in QuoridorEnv.

Design:
    The opponent model's weights are periodically copied from the online
    network during training (every OPPONENT_UPDATE_FREQ steps). Between
    updates the opponent is frozen — giving the online network a stable,
    stationary MDP to learn against. This avoids the non-stationarity of
    true simultaneous self-play, which can cause DQN to diverge.

Usage:
    opponent_model = copy.deepcopy(online_net)
    bot = DQNBot(opponent_model, device)
    env = QuoridorEnv(bot=bot)

    # Periodically sync opponent to latest weights:
    bot.update_weights(online_net.state_dict())
"""

import numpy as np
import torch

from agents.dqn_model import DQNModel
from quoridor.action_encoding import index_to_action
from quoridor.game import QuoridorState


class DQNBot:
    """
    Greedy DQN-based opponent for frozen self-play.

    Plays epsilon=0 (fully greedy) — no exploration. The opponent should
    exploit its current policy maximally so the online network is always
    facing the strongest version of that policy.

    Parameters
    ----------
    model  : DQNModel — opponent network (usually a deepcopy of online_net)
    device : torch.device — must match the device used for online training
    """

    def __init__(self, model: DQNModel, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.eval()  # opponent is never trained directly; always frozen

    def reset(self) -> None:
        """No stateful opening book or history to reset between episodes."""
        pass

    def update_weights(self, state_dict: dict) -> None:
        """
        Copy new weights into the opponent model.

        Called every OPPONENT_UPDATE_FREQ steps in the training loop to keep
        the opponent competitive. Between calls the opponent is frozen so the
        online network faces a stable (stationary) target.

        load_state_dict does not preserve training mode, so we re-apply eval().
        """
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def choose_action(self, game: QuoridorState) -> tuple:
        """
        Select the greedy action using the opponent DQN's Q-values.

        Called by QuoridorEnv during P1's turn, so game.turn == 1.
        get_observation() and get_legal_mask() already return P1's perspective
        (board is flipped so P1 always sees their goal at row 0).

        Returns
        -------
        ("move", row, col) or ("fence", row, col, orientation)
        """
        spatial, scalars = game.get_observation()
        legal_mask = game.get_legal_mask()

        spatial_t = torch.tensor(spatial).unsqueeze(0).to(self.device)    # (1, 4, 9, 9)
        scalars_t = torch.tensor(scalars).unsqueeze(0).to(self.device)    # (1, 2)
        mask_t    = torch.tensor(legal_mask).unsqueeze(0).to(self.device) # (1, 137)

        with torch.no_grad():
            q = self.model(spatial_t, scalars_t, mask_t)  # (1, 137)
        action_idx = int(q.argmax(dim=1).item())

        return self._decode_action(game, action_idx)

    def _decode_action(self, game: QuoridorState, idx: int) -> tuple:
        """
        Convert an action index to a game action tuple.

        Wall actions decode directly from index_to_action(). Pawn move actions
        encode direction deltas, not absolute positions — we scan get_valid_moves()
        for the destination whose direction sign matches the decoded delta.
        This mirrors the logic in QuoridorEnv._apply_index().
        """
        action = index_to_action(idx)

        if action[0] == "fence":
            return action  # ("fence", row, col, orientation) — direct

        # Move: encoded as direction delta (dr, dc). Find the legal destination
        # in that direction among get_valid_moves() (handles 1-step and jump moves).
        dr, dc = action[1], action[2]
        cur_r = int(game.pos[game.turn, 0])
        cur_c = int(game.pos[game.turn, 1])

        for dest_r, dest_c in game.get_valid_moves():
            if (np.sign(dest_r - cur_r) == np.sign(dr) and
                    np.sign(dest_c - cur_c) == np.sign(dc)):
                return ("move", dest_r, dest_c)

        # Fallback: legal_mask should have prevented an illegal action index,
        # but if somehow reached, take the first valid move rather than crashing.
        dest_r, dest_c = game.get_valid_moves()[0]
        return ("move", dest_r, dest_c)
