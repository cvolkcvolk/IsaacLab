# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-mode compatibility for learning framework wrappers."""

import gymnasium as gym


def _check_autoreset_mode(env) -> None:
    """Reject advertised reset modes that require callers to restart completed episodes."""
    if isinstance(env.unwrapped, gym.Env) and "autoreset_mode" in env.unwrapped.metadata:
        autoreset_mode = env.unwrapped.metadata["autoreset_mode"]
        if autoreset_mode != gym.vector.AutoresetMode.SAME_STEP:
            raise ValueError(
                f"RL wrappers require same-step automatic resets; got {autoreset_mode!r}. "
                "Set cfg.autoreset_mode to gymnasium.vector.AutoresetMode.SAME_STEP."
            )
