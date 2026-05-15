from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import gymnasium as gym
import numpy as np

from .compat import patch_legacy_mario_stack


DEFAULT_ENV_ID = "SuperMarioBros-1-1-v0"
DEFAULT_FRAME_SHAPE = (84, 84)
SIMPLE_WITH_DOWN = [[], ["right"], ["right", "A"], ["right", "B"], ["right", "A", "B"], ["A"], ["left"], ["down"], ["left", "down"], ["right", "down"]]


def preprocess_frame(frame: np.ndarray, size: tuple[int, int] = DEFAULT_FRAME_SHAPE) -> np.ndarray:
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


class MarioGymnasiumEnv(gym.Env):
    """Gymnasium-facing wrapper around gym-super-mario-bros + nes-py."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_id: str = DEFAULT_ENV_ID,
        frame_shape: tuple[int, int] = DEFAULT_FRAME_SHAPE,
        movement: str = "simple_down",
        max_stall_steps: int = 240,
    ) -> None:
        super().__init__()
        patch_legacy_mario_stack()

        import gym_super_mario_bros
        from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT
        from nes_py.wrappers import JoypadSpace

        movements = {
            "right_only": RIGHT_ONLY,
            "simple": SIMPLE_MOVEMENT,
            "simple_down": SIMPLE_WITH_DOWN,
            "complex": COMPLEX_MOVEMENT,
        }
        if movement not in movements:
            raise ValueError(f"Unknown movement '{movement}'. Expected {sorted(movements)}")

        self.env_id = env_id
        self.frame_shape = frame_shape
        self.max_stall_steps = max_stall_steps
        self.buttons = movements[movement]
        self.action_names = ["+".join(action) if action else "NOOP" for action in self.buttons]
        self._env = JoypadSpace(
            gym_super_mario_bros.make(
                env_id,
                apply_api_compatibility=True,
                disable_env_checker=True,
            ),
            self.buttons,
        )
        self.action_space = gym.spaces.Discrete(self._env.action_space.n)
        self.observation_space = gym.spaces.Box(0, 255, shape=(frame_shape[1], frame_shape[0], 3), dtype=np.uint8)
        self._last_raw_frame: np.ndarray | None = None
        self._best_x = 0
        self._stall_steps = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            try:
                self._env.seed(seed)
            except Exception:
                pass
            self.action_space.seed(seed)
        raw = self._env.reset()
        if isinstance(raw, tuple):
            raw = raw[0]
        self._last_raw_frame = raw
        self._best_x = 0
        self._stall_steps = 0
        return preprocess_frame(raw, self.frame_shape), self._info_with_progress({})

    def step(self, action: int):
        step = self._env.step(int(action))
        if len(step) == 5:
            raw, reward, terminated, truncated, info = step
            done = bool(terminated or truncated)
        else:
            raw, reward, done, info = step
            truncated = False
        self._last_raw_frame = raw
        info = self._info_with_progress(info)
        if info["x_pos"] > self._best_x:
            self._best_x = info["x_pos"]
            self._stall_steps = 0
        else:
            self._stall_steps += 1
        stalled = self._stall_steps >= self.max_stall_steps
        return preprocess_frame(raw, self.frame_shape), float(reward), bool(done), bool(truncated or stalled), info

    def render(self):
        if self._last_raw_frame is None:
            return None
        return self._last_raw_frame

    def close(self) -> None:
        self._env.close()

    def _info_with_progress(self, info: dict[str, Any]) -> dict[str, Any]:
        info = dict(info)
        info.setdefault("x_pos", int(info.get("x_pos", 0)))
        info.setdefault("world", int(info.get("world", 1)))
        info.setdefault("stage", int(info.get("stage", 1)))
        info.setdefault("score", int(info.get("score", 0)))
        info.setdefault("coins", int(info.get("coins", 0)))
        info.setdefault("time", int(info.get("time", 400)))
        info.setdefault("status", str(info.get("status", "unknown")))
        info["best_x"] = max(self._best_x, int(info["x_pos"]))
        return info
