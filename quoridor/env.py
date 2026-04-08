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
import random

from config import (INITIAL_WALLS_PER_PLAYER, REPETITION_PENALTY,
                     REWARD_SHAPING_OPP_COEF, REWARD_SHAPING_SELF_COEF,
                     STEP_PENALTY)


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

    def __init__(self, bot=None, use_bfs: bool = False, use_reward_shaping: bool = False,
                 max_moves: int = 300,
                 reward_self_coef: float = REWARD_SHAPING_SELF_COEF,
                 reward_opp_coef: float = REWARD_SHAPING_OPP_COEF,
                 randomize_start: bool = False,
                 repetition_penalty: float = REPETITION_PENALTY) -> None:
        self.state = QuoridorState()
        self.bot = bot if bot is not None else HeuristicBot()
        # When True, get_observation() returns 6-channel spatial tensors (adds
        # BFS distance maps as ch4 and ch5). Must match the model's NUM_CHANNELS.
        self.use_bfs = use_bfs
        # When True, step() adds a small dense shaped reward based on the change
        # in each player's shortest path caused by the agent's action. Terminal
        # rewards (+1/-1) remain pure — shaping only applies to non-terminal steps.
        self.use_reward_shaping = use_reward_shaping
        # Reward shaping coefficients — overridable via CLI for experimentation.
        self.reward_self_coef = reward_self_coef
        self.reward_opp_coef = reward_opp_coef
        # Maximum plies (half-moves) before the game is truncated as a draw.
        # Prevents infinite loops when two similar policies mirror each other.
        self.max_moves = max_moves
        self._move_count = 0
        # When True, the bot takes the first move in ~50% of episodes, so the
        # agent practises responding to opponent openings (effectively playing P1).
        self.randomize_start = randomize_start
        # Penalty applied per revisit of a pawn position within an episode.
        # Scaled by (visit_count - 1) so the first visit is free. Discourages
        # the back-and-forth oscillation that memoryless policies fall into.
        self.repetition_penalty = repetition_penalty
        # Position visit counts for current episode, keyed by (row, col).
        self._pos_visits: dict[tuple[int, int], int] = {}

    def reset(self):
        """Start a new episode. Returns (spatial, scalars)."""
        self.state.reset()
        self.bot.reset()
        self._move_count = 0
        # Record the agent's starting position as the first visit.
        start_pos = (int(self.state.pos[0, 0]), int(self.state.pos[0, 1]))
        self._pos_visits = {start_pos: 1}

        # Random side switching: 50% of episodes the bot moves first, so the
        # agent sees diverse opening states instead of always going first.
        if self.randomize_start and random.random() < 0.5:
            bot_action = self.bot.choose_action(self.state)
            self._dispatch_bot_action(bot_action)
            self._move_count += 1

        spatial, scalars = self.state.get_observation(use_bfs=self.use_bfs)  # turn == 0 (P0)
        return spatial, scalars

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

        # ── 2. Measure paths BEFORE agent action (reward shaping only) ───────
        # Captured here so we credit the agent's action alone, not the bot's
        # subsequent response. The agent is always P0.
        if self.use_reward_shaping:
            my_path_before  = self.state.shortest_path(0)
            opp_path_before = self.state.shortest_path(1)

        # ── 3. Apply agent action ────────────────────────────────────────────
        self._move_count += 1  # agent's ply
        agent_won = self._apply_index(action)

        # ── 3b. Repetition penalty — track pawn position visits ──────────
        # Only fires on pawn moves (wall placements don't change pawn pos).
        # Penalty scales with revisit count: first visit free, then -0.03, -0.06, ...
        rep_penalty = 0.0
        if self.repetition_penalty != 0.0:
            cur_pos = (int(self.state.pos[0, 0]), int(self.state.pos[0, 1]))
            self._pos_visits[cur_pos] = self._pos_visits.get(cur_pos, 0) + 1
            revisits = self._pos_visits[cur_pos] - 1  # first visit is free
            if revisits > 0:
                rep_penalty = self.repetition_penalty * revisits

        if agent_won:
            # Game over — terminal reward stays pure, no shaping applied.
            spatial, scalars = self.state.get_observation(use_bfs=self.use_bfs)
            return spatial, scalars, +1.0, True, {"legal_mask": _TERMINAL_MASK.copy()}

        # ── 4. Compute shaped reward BEFORE bot responds ──────────────────────
        # Measured after the agent's action but before the bot's response so the
        # reward reflects only what the agent did.
        #
        # shaped_reward = OPP_COEF  * (opp_path_after - opp_path_before)  [good wall]
        #               + SELF_COEF * (my_path_before  - my_path_after)  [good pawn]
        #
        # Both terms are positive when good things happen:
        #   - Good wall increases opp_path → opp_path_after > opp_path_before → positive
        #   - Good pawn move decreases my_path → my_path_before > my_path_after → positive
        # Note: pawn moves never change wall positions, so opp_path_after == opp_path_before
        # for all pawn moves — the OPP_COEF term only fires on wall placements.
        if self.use_reward_shaping:
            my_path_after  = self.state.shortest_path(0)
            opp_path_after = self.state.shortest_path(1)
            # Scale shaping by fraction of walls the agent still has.
            # Teaches wall scarcity: burning walls early yields diminishing
            # shaped reward, nudging the agent to conserve walls for when
            # they matter most.
            wall_frac = self.state.walls_left[0] / INITIAL_WALLS_PER_PLAYER
            shaped_reward = wall_frac * (
                self.reward_opp_coef  * (opp_path_after  - opp_path_before) +
                self.reward_self_coef * (my_path_before  - my_path_after)
            )
        else:
            shaped_reward = 0.0

        # ── 5. Bot responds as P1 ────────────────────────────────────────────
        bot_action = self.bot.choose_action(self.state)
        self._dispatch_bot_action(bot_action)
        self._move_count += 1  # bot's ply

        bot_won = self.state.done  # place_fence / move_to set state.done internally

        if bot_won:
            # Terminal reward stays pure — bot's win is a clean -1.0 signal.
            spatial, scalars = self.state.get_observation(use_bfs=self.use_bfs)
            return spatial, scalars, -1.0, True, {"legal_mask": _TERMINAL_MASK.copy()}

        # ── 6. Truncation check — prevent infinite loops ────────────────────
        if self._move_count >= self.max_moves:
            spatial, scalars = self.state.get_observation(use_bfs=self.use_bfs)
            # Draw: reward based on who is closer to goal. Gives a learning
            # signal even when neither player wins outright.
            my_dist  = self.state.shortest_path(0)
            opp_dist = self.state.shortest_path(1)
            if my_dist < opp_dist:
                truncation_reward = 0.5   # agent was closer — soft win
            elif opp_dist < my_dist:
                truncation_reward = -0.5  # bot was closer — soft loss
            else:
                truncation_reward = 0.0   # true draw
            return spatial, scalars, truncation_reward, True, {"legal_mask": _TERMINAL_MASK.copy()}

        # ── 7. Non-terminal transition ───────────────────────────────────────
        # STEP_PENALTY is a small negative reward applied every non-terminal step
        # to discourage stalling. Applied on top of (or instead of) shaping reward.
        # Terminal rewards remain pure ±1.0.
        spatial, scalars = self.state.get_observation(use_bfs=self.use_bfs)  # turn is back to P0
        next_mask = self.state.get_legal_mask()
        return spatial, scalars, shaped_reward + STEP_PENALTY + rep_penalty, False, {"legal_mask": next_mask}

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
