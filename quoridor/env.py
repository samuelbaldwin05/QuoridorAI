"""
Gym-style wrapper around QuoridorState for single-agent RL training.

The agent is always P0. Bot plays P1 internally after each agent action,
so the training loop sees a standard reset/step MDP interface.

Rewards: +1 agent wins, -1 bot wins, STEP_PENALTY per non-terminal step.
"""

import numpy as np

from quoridor.game import QuoridorState
from quoridor.action_encoding import NUM_ACTIONS, index_to_action
from agents.bot import HeuristicBot
from config import STEP_PENALTY


_TERMINAL_MASK = np.zeros(NUM_ACTIONS, dtype=bool)


class QuoridorEnv:
    """Single-agent Quoridor environment. Agent=P0, bot=P1."""

    def __init__(self, bot=None):
        self.state = QuoridorState()
        self.bot = bot if bot is not None else HeuristicBot()

    def reset(self):
        """Start a new episode. Returns (spatial, scalars)."""
        self.state.reset()
        self.bot.reset()
        return self.state.get_observation()

    def step(self, action: int):
        """
        Execute agent action, then bot responds.
        Returns (spatial, scalars, reward, done, info).
        info always has 'legal_mask'.
        """
        # validate
        legal = self.state.get_legal_mask()
        if not legal[action]:
            raise ValueError(f"Action {action} is illegal in current state")

        # agent moves
        agent_won = self._apply_index(action)
        if agent_won:
            spatial, scalars = self.state.get_observation()
            return spatial, scalars, +1.0, True, {"legal_mask": _TERMINAL_MASK.copy()}

        # bot responds
        bot_action = self.bot.choose_action(self.state)
        self._dispatch_bot_action(bot_action)

        if self.state.done:
            spatial, scalars = self.state.get_observation()
            return spatial, scalars, -1.0, True, {"legal_mask": _TERMINAL_MASK.copy()}

        # non-terminal — small step penalty to discourage stalling
        spatial, scalars = self.state.get_observation()
        return spatial, scalars, STEP_PENALTY, False, {"legal_mask": self.state.get_legal_mask()}

    def get_legal_mask(self):
        """Passthrough to state.get_legal_mask()."""
        return self.state.get_legal_mask()

    def _apply_index(self, idx: int) -> bool:
        """Decode action index and apply to game state. Returns True if won."""
        action = index_to_action(idx)

        if action[0] == "move":
            dr, dc = action[1], action[2]
            cur_r = int(self.state.pos[self.state.turn, 0])
            cur_c = int(self.state.pos[self.state.turn, 1])

            # direction-based: find destination matching this direction sign
            for dest_r, dest_c in self.state.get_valid_moves():
                if (np.sign(dest_r - cur_r) == np.sign(dr)
                        and np.sign(dest_c - cur_c) == np.sign(dc)):
                    return self.state.move_to(dest_r, dest_c)

            raise ValueError(f"No valid destination for direction ({dr}, {dc})")

        else:  # fence
            self.state.place_fence(action[1], action[2], action[3])
            return False

    def _dispatch_bot_action(self, bot_action):
        """Apply a bot action tuple to the game state."""
        if bot_action[0] == "move":
            _, r, c = bot_action
            self.state.move_to(r, c)
        else:
            _, r, c, orientation = bot_action
            self.state.place_fence(r, c, orientation)
