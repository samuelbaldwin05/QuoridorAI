"""
Gym-style single-agent wrapper around QuoridorState.

This module bridges the two-player Quoridor game and the single-agent RL
training loop. From the training loop's perspective, there is only one agent
(P0). The env handles P1's turns internally using a bot, making the full game
transparent — each reset/step cycle looks like a standard MDP.

Invariant: the agent is always P0. P0 moves first after every reset().

Interface mirrors the core Gym API (no Gym dependency required):
    reset() -> (spatial, scalars)
    step(action) -> (spatial, scalars, reward, done, info)
    get_legal_mask() -> legal_mask

Reward structure (sparse, per CLAUDE.md):
    +1.0  agent (P0) reaches row 0
    -1.0  bot   (P1) reaches row 8
     0.0  all other transitions

Opponent:
    Pass any bot instance to __init__(bot=...) to swap opponents.
    Defaults to HeuristicBot for backward compatibility. Use RandomBot
    for curriculum training (easier first opponent before HeuristicBot).
"""

import numpy as np

from quoridor.game import QuoridorState
from quoridor.action_encoding import NUM_ACTIONS, index_to_action
from agents.bot import HeuristicBot
from config import REWARD_SHAPING_OPP_COEF, REWARD_SHAPING_SELF_COEF


# A mask of all False is returned on terminal steps — no moves are legal
# once the game is over.
_TERMINAL_MASK = np.zeros(NUM_ACTIONS, dtype=bool)


class QuoridorEnv:
    """
    Single-agent Quoridor environment.

    The agent always plays as P0 (bottom, goal = row 0). After each agent
    action, the env runs one bot turn as P1, then returns control.
    The training loop never observes P1's turn directly.

    Parameters
    ----------
    bot : optional
        Any bot instance with reset() and choose_action(game) methods.
        Defaults to HeuristicBot(). Pass RandomBot() for curriculum training.
    """

    def __init__(self, bot=None, use_bfs: bool = False, use_reward_shaping: bool = False) -> None:
        self.state = QuoridorState()
        self.bot = bot if bot is not None else HeuristicBot()
        # When True, get_observation() returns 6-channel spatial tensors (adds
        # BFS distance maps as ch4 and ch5). Must match the model's NUM_CHANNELS.
        self.use_bfs = use_bfs
        # When True, step() adds a small dense shaped reward based on the change
        # in each player's shortest path caused by the agent's action. Terminal
        # rewards (+1/-1) remain pure — shaping only applies to non-terminal steps.
        self.use_reward_shaping = use_reward_shaping

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Start a new episode.

        Resets the game state and bot state. P0 always moves first.

        Returns
        -------
        spatial : np.ndarray, shape (4, 9, 9), dtype float32
        scalars : np.ndarray, shape (2,),     dtype float32
        """
        self.state.reset()
        self.bot.reset()
        spatial, scalars = self.state.get_observation(use_bfs=self.use_bfs)  # turn == 0 (P0)
        return spatial, scalars

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        """
        Execute one agent action (P0), then have the bot respond as P1.

        Parameters
        ----------
        action : int
            Integer action index in [0, NUM_ACTIONS). Must be legal according
            to the current get_legal_mask(); raises ValueError otherwise.

        Returns
        -------
        spatial  : np.ndarray, shape (4, 9, 9), dtype float32
        scalars  : np.ndarray, shape (2,),      dtype float32
        reward   : float  — +1.0, -1.0, or 0.0
        done     : bool
        info     : dict   — always contains {"legal_mask": np.ndarray (137,) bool}

        Notes
        -----
        On terminal steps the returned observation is the post-game state and
        the legal_mask is all False (no moves remain once the game ends).
        """
        # ── 1. Validate ──────────────────────────────────────────────────────
        legal = self.state.get_legal_mask()
        if not legal[action]:
            raise ValueError(
                f"Action {action} is illegal in the current state. "
                "Only actions where get_legal_mask()[action] is True are allowed."
            )

        # ── 2. Measure paths BEFORE agent action (reward shaping only) ───────
        # Captured here so we credit the agent's action alone, not the bot's
        # subsequent response. The agent is always P0.
        if self.use_reward_shaping:
            my_path_before  = self.state.shortest_path(0)
            opp_path_before = self.state.shortest_path(1)

        # ── 3. Apply agent action ────────────────────────────────────────────
        agent_won = self._apply_index(action)

        if agent_won:
            # Game over — terminal reward stays pure, no shaping applied.
            spatial, scalars = self.state.get_observation(use_bfs=self.use_bfs)
            return spatial, scalars, +1.0, True, {"legal_mask": _TERMINAL_MASK.copy()}

        # ── 4. Compute shaped reward BEFORE bot responds ──────────────────────
        # Measured after the agent's action but before the bot's response so the
        # reward reflects only what the agent did.
        #
        # shaped_reward = OPP_COEF  * (opp_path_before - opp_path_after)  [good wall]
        #               + SELF_COEF * (my_path_before  - my_path_after)   [good pawn]
        #
        # Both terms are positive when good things happen and negative otherwise.
        if self.use_reward_shaping:
            my_path_after  = self.state.shortest_path(0)
            opp_path_after = self.state.shortest_path(1)
            shaped_reward = (
                REWARD_SHAPING_OPP_COEF  * (opp_path_before - opp_path_after) +
                REWARD_SHAPING_SELF_COEF * (my_path_before  - my_path_after)
            )
        else:
            shaped_reward = 0.0

        # ── 5. Bot responds as P1 ────────────────────────────────────────────
        bot_action = self.bot.choose_action(self.state)
        self._dispatch_bot_action(bot_action)

        bot_won = self.state.done  # place_fence / move_to set state.done internally

        if bot_won:
            # Terminal reward stays pure — bot's win is a clean -1.0 signal.
            spatial, scalars = self.state.get_observation(use_bfs=self.use_bfs)
            return spatial, scalars, -1.0, True, {"legal_mask": _TERMINAL_MASK.copy()}

        # ── 6. Non-terminal transition ───────────────────────────────────────
        spatial, scalars = self.state.get_observation(use_bfs=self.use_bfs)  # turn is back to P0
        next_mask = self.state.get_legal_mask()
        return spatial, scalars, shaped_reward, False, {"legal_mask": next_mask}

    def get_legal_mask(self) -> np.ndarray:
        """
        Convenience passthrough to state.get_legal_mask().

        Returns
        -------
        mask : np.ndarray, shape (137,), dtype bool
        """
        return self.state.get_legal_mask()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_index(self, idx: int) -> bool:
        """
        Decode an action index and apply it to self.state.

        For pawn moves the action encodes a direction delta, not an absolute
        destination. We scan get_valid_moves() for any destination whose
        direction sign matches the decoded delta — this correctly handles both
        1-step advances and 2-step straight jumps (same sign).

        Returns True if the move won the game for the current player.
        """
        action = index_to_action(idx)

        if action[0] == "move":
            dr, dc = action[1], action[2]
            cur_r = int(self.state.pos[self.state.turn, 0])
            cur_c = int(self.state.pos[self.state.turn, 1])

            for dest_r, dest_c in self.state.get_valid_moves():
                if (np.sign(dest_r - cur_r) == np.sign(dr)
                        and np.sign(dest_c - cur_c) == np.sign(dc)):
                    return self.state.move_to(dest_r, dest_c)

            raise ValueError(
                f"No valid destination found for direction ({dr}, {dc}). "
                "This action should have been masked as illegal."
            )

        else:  # "fence"
            # place_fence switches the turn internally; it never wins the game
            # (a fence placement cannot be the winning move in Quoridor).
            self.state.place_fence(action[1], action[2], action[3])
            return False

    def _dispatch_bot_action(self, bot_action: tuple) -> None:
        """Apply a bot action tuple returned by HeuristicBot.choose_action()."""
        if bot_action[0] == "move":
            _, r, c = bot_action
            self.state.move_to(r, c)
        else:  # "fence"
            _, r, c, orientation = bot_action
            self.state.place_fence(r, c, orientation)
