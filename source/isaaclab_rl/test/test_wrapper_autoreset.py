# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-mode validation at learning wrapper construction."""

from typing import Any

import pytest

pytestmark = pytest.mark.unit


def _wrap_env(library: str, env: Any) -> Any:
    if library == "rsl_rl":
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        return RslRlVecEnvWrapper(env)
    if library == "rl_games":
        from isaaclab_rl.rl_games import RlGamesVecEnvWrapper

        return RlGamesVecEnvWrapper(env, "cuda:0", 100, 100)
    if library == "sb3":
        from isaaclab_rl.sb3 import Sb3VecEnvWrapper

        return Sb3VecEnvWrapper(env)
    if library == "skrl":
        from isaaclab_rl.skrl import SkrlVecEnvWrapper

        return SkrlVecEnvWrapper(env)
    raise ValueError(f"Unsupported RL library: {library}")


@pytest.mark.parametrize("library", ["rsl_rl", "rl_games", "sb3", "skrl"])
def test_wrapper_rejects_disabled_automatic_resets(library: str) -> None:
    """Training wrappers reject explicit resets before initializing or resetting the environment."""
    import gymnasium as gym

    from isaaclab.envs import ManagerBasedRLEnv

    env = ManagerBasedRLEnv.__new__(ManagerBasedRLEnv)
    env.metadata = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}

    with pytest.raises(ValueError, match="require same-step automatic resets"):
        _wrap_env(library, env)
