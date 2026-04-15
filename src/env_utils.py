"""Environment wrappers and vectorized environment factory for CarRacing."""

from typing import Callable

import gymnasium as gym
import numpy as np


class NormalizeObservation(gym.ObservationWrapper):
    """Divide pixel observations by 255 to get float32 in [0, 1]."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32) / 255.0


class Float64Action(gym.ActionWrapper):
    """Cast actions to float64 to work around box2d-py SWIG float32 bug on Python 3.13."""

    def action(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=np.float64)


def make_env(seed: int, render_mode: str | None = None) -> Callable[[], gym.Env]:
    """Return a thunk that creates a single wrapped CarRacing environment.

    Wrapper order: GrayScale -> Resize(84,84) -> Normalize(/255) -> FrameStack(4).
    """

    def _thunk() -> gym.Env:
        env = gym.make(
            "CarRacing-v2",
            continuous=True,
            render_mode=render_mode,
        )
        env = Float64Action(env)
        env = gym.wrappers.GrayScaleObservation(env, keep_dim=False)
        env = gym.wrappers.ResizeObservation(env, shape=(84, 84))
        env = NormalizeObservation(env)
        env = gym.wrappers.FrameStack(env, num_stack=4)
        env.reset(seed=seed)
        return env

    return _thunk


def make_vec_env(
    n_envs: int, seed: int = 0, render_mode: str | None = None
) -> gym.vector.VectorEnv:
    """Create an AsyncVectorEnv with n_envs parallel CarRacing instances."""
    env_fns = [make_env(seed + i, render_mode=render_mode) for i in range(n_envs)]
    return gym.vector.AsyncVectorEnv(env_fns)
