"""
vec_env.py — Synchronous vectorized Quoridor environment.

Wraps N independent QuoridorEnv instances so the training loop can collect
N parallel rollouts in one Python call instead of N sequential ones. The
envs are synchronous (no threading) — the speedup comes from amortising
Python overhead across N steps per loop iteration, not from parallelism.

Why synchronous instead of subprocess-based?
    QuoridorEnv is pure Python + NumPy. The bottleneck is the Python interpreter,
    not I/O or CPU-bound work. Subprocesses would add IPC serialisation overhead
    that exceeds the env cost at this scale. SyncVectorEnv is always at least as
    fast here, and far simpler to debug.

Sharing the opponent reference:
    All N sub-envs hold a reference to the SAME bot object. This is the correct
    design when the bot is stateless per-action (HeuristicBot, EpsilonHeuristicBot,
    PPOBot). When `epsilon_bot.epsilon` is updated in the training loop, all envs
    immediately pick up the new value with no extra bookkeeping.

    CAVEAT: OpponentPool stores per-episode state in `_current`. With n_envs > 1,
    interleaved calls to pool.reset() from different sub-envs will clobber each
    other's `_current`. Use n_envs=1 (the default) when --opponent=pool.

Auto-reset behaviour:
    When a sub-env reaches `done=True`, `vec_env.step()` immediately resets it
    and returns the FIRST observation of the NEW episode in the output arrays.
    This matches standard gym VectorEnv semantics. The training loop therefore
    never needs to call reset() itself — it only calls it once at the start.
"""

import numpy as np

from quoridor.env import QuoridorEnv


class VecQuoridorEnv:
    """
    Synchronous vectorized wrapper over N QuoridorEnv instances.

    Parameters
    ----------
    n_envs : int
        Number of parallel environments. Use 1 for identical behaviour to the
        single-env case (useful for validating the vec-env code path is correct).
    bot :
        Opponent bot shared across all sub-envs. Must implement reset() and
        choose_action(). All sub-envs hold a reference to the SAME object, so
        attribute updates (e.g. epsilon_bot.epsilon = 0.2) propagate automatically.
    use_bfs : bool
        Passed to each QuoridorEnv. Must match the model's channel count.
    use_reward_shaping : bool
        Passed to each QuoridorEnv. Eval env should always set this to False.
    """

    def __init__(
        self,
        n_envs:             int,
        bot,
        use_bfs:            bool  = False,
        use_reward_shaping: bool  = False,
        reward_self_coef:   float = None,
        reward_opp_coef:    float = None,
        randomize_start:    bool  = False,
        repetition_penalty: float = None,
    ) -> None:
        self.n_envs = n_envs
        # If the bot supports fork() (e.g. OpponentPool), give each env its own
        # fork so they don't clobber each other's per-episode state. The forks
        # share the underlying checkpoint pool, so add() on the parent propagates.
        # For simple bots (HeuristicBot, PPOBot), all envs share the same object.
        # Build optional kwargs for reward shaping coefficients.
        extra_kwargs = {}
        if reward_self_coef is not None:
            extra_kwargs["reward_self_coef"] = reward_self_coef
        if reward_opp_coef is not None:
            extra_kwargs["reward_opp_coef"] = reward_opp_coef
        if repetition_penalty is not None:
            extra_kwargs["repetition_penalty"] = repetition_penalty
        self.envs = []
        for _ in range(n_envs):
            env_bot = bot.fork() if hasattr(bot, "fork") else bot
            self.envs.append(
                QuoridorEnv(bot=env_bot, use_bfs=use_bfs,
                            use_reward_shaping=use_reward_shaping,
                            randomize_start=randomize_start, **extra_kwargs)
            )
        self._bot = bot  # keep a direct reference for the property setter below

    # ------------------------------------------------------------------
    # bot property — mirrors env.bot so the training loop can write
    #   vec_env.bot = PPOBot(frozen_model, device)
    # and have it propagate to all sub-envs (used by self_play / pool).
    # ------------------------------------------------------------------

    @property
    def bot(self):
        return self._bot

    @bot.setter
    def bot(self, value) -> None:
        self._bot = value
        for env in self.envs:
            env.bot = value.fork() if hasattr(value, "fork") else value

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Reset all sub-envs and return stacked initial observations.

        Returns
        -------
        spatial : (N, C, 9, 9) float32
        scalars : (N, 2)       float32
        """
        obs = [env.reset() for env in self.envs]
        spatial  = np.stack([o[0] for o in obs])  # (N, C, 9, 9)
        scalars  = np.stack([o[1] for o in obs])  # (N, 2)
        return spatial, scalars

    def step(
        self,
        actions: np.ndarray,  # (N,) int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Step all N envs with the given actions.

        Done envs are auto-reset: the first observation of the NEW episode is
        returned in the output arrays (not the terminal observation). This is
        consistent with gym VectorEnv semantics and means the training loop can
        feed the output directly back as the next state.

        Parameters
        ----------
        actions : (N,) int array

        Returns
        -------
        next_spatial : (N, C, 9, 9) float32
        next_scalars : (N, 2)       float32
        rewards      : (N,)         float32
        dones        : (N,)         bool
        infos        : list of N dicts (each has 'legal_mask' key)
        """
        results = [env.step(int(a)) for env, a in zip(self.envs, actions)]

        next_spatial = np.stack([r[0] for r in results])   # (N, C, 9, 9)
        next_scalars = np.stack([r[1] for r in results])   # (N, 2)
        rewards      = np.array([r[2] for r in results], dtype=np.float32)  # (N,)
        dones        = np.array([r[3] for r in results])                    # (N,)
        infos        = [r[4] for r in results]

        # Auto-reset done envs. The returned obs is the first obs of the new
        # episode, not the terminal obs — this is what the next rollout step
        # needs to condition on.
        for i, done in enumerate(dones):
            if done:
                sp, sc = self.envs[i].reset()
                next_spatial[i] = sp
                next_scalars[i] = sc

        return next_spatial, next_scalars, rewards, dones, infos

    def get_legal_mask(self) -> np.ndarray:
        """
        Return the legal action mask for all N envs.

        Returns
        -------
        (N, 137) bool
        """
        return np.stack([env.get_legal_mask() for env in self.envs])
