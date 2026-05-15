from __future__ import annotations

import numpy as np

from .actions import buttons_to_action_idx, normalize_action_name

ACTION_NAMES = []


class RandomController:
    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, info: dict) -> int:
        return int(self.rng.integers(0, self.n_actions))


class RightBiasedController:
    """Data-collection policy strongly biased toward right+B acceleration."""

    def __init__(self, action_names: list[str], seed: int | None = None, epsilon: float = 0.08) -> None:
        self.action_names = action_names
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        self.noop = pick_action(action_names, ["NOOP"], fallback=0)
        self.right = pick_buttons(action_names, [("right",)], fallback=1)
        self.left = pick_buttons(action_names, [("left",)], fallback=self.noop)
        self.down = pick_buttons(action_names, [("down",)], fallback=self.noop)
        self.right_b = pick_buttons(action_names, [("right", "B"), ("right",)], fallback=self.right)
        self.right_a = pick_buttons(action_names, [("right", "A"), ("right",)], fallback=self.right)
        self.right_ab = pick_buttons(action_names, [("right", "A", "B"), ("right", "A"), ("right", "B"), ("right",)], fallback=self.right)
        self.noise_actions = [self.right_b, self.right_ab, self.right, self.left, self.down]

    def __call__(self, obs, info: dict) -> int:
        x = int(info.get("x_pos", 0))
        time = int(info.get("time", 400))
        if self.rng.random() < self.epsilon and x < 820:
            return int(self.rng.choice(self.noise_actions))
        if x < 40 and self.rng.random() < 0.15:
            return self.noop
        if 830 <= x <= 1010:
            return self.right_ab if self.rng.random() < 0.90 else self.right_b
        if self.rng.random() < 0.45 or time % 37 in (0, 1, 2, 3, 4, 5):
            return self.right_ab
        if self.rng.random() < 0.08:
            return self.right_a
        return self.right_b


class AccelJumpController:
    """Level-1 data policy: run first, then use sustained run+jump windows.

    The learned model was previously overexposed to short, poorly timed jumps.
    This controller creates clearer causal demonstrations: build horizontal speed
    with right+B, then hold right+A+B through known obstacle regions.
    """

    def __init__(self, action_names: list[str], seed: int | None = None, epsilon: float = 0.03) -> None:
        self.action_names = action_names
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon
        self.noop = pick_action(action_names, ["NOOP"], fallback=0)
        self.right_b = pick_buttons(action_names, [("right", "B"), ("right",)], fallback=1)
        self.right_ab = pick_buttons(action_names, [("right", "A", "B"), ("right", "A"), ("right", "B")], fallback=self.right_b)
        self.right = pick_buttons(action_names, [("right",)], fallback=self.right_b)
        self.left = pick_buttons(action_names, [("left",)], fallback=self.noop)
        self.down = pick_buttons(action_names, [("down",), ("left", "down"), ("right", "down")], fallback=self.noop)
        self.noise_actions = [self.right_b, self.right_ab, self.right, self.left, self.down]
        self.jump_remaining = 0
        self.used_windows: set[int] = set()
        self.windows = jitter_windows(
            self.rng,
            [
                (250, 365, 22, 34),
                (410, 520, 14, 24),
                (570, 650, 16, 28),
                (650, 760, 24, 40),
                (825, 980, 50, 75),
                (960, 1120, 32, 56),
                (1120, 1300, 24, 44),
            ],
        )

    def __call__(self, obs, info: dict) -> int:
        x = int(info.get("x_pos", 0))
        if self.rng.random() < self.epsilon and x < 820:
            return int(self.rng.choice(self.noise_actions))
        if self.jump_remaining > 0:
            self.jump_remaining -= 1
            return self.right_ab

        for idx, (start, stop, min_hold, max_hold) in enumerate(self.windows):
            if idx not in self.used_windows and start <= x <= stop:
                self.used_windows.add(idx)
                self.jump_remaining = int(self.rng.integers(min_hold, max_hold + 1))
                return self.right_ab

        return self.right_b


class ScriptedController:
    def __init__(self, actions: list[int], fallback) -> None:
        self.actions = actions
        self.fallback = fallback
        self.idx = 0

    def __call__(self, obs, info: dict) -> int:
        if self.idx < len(self.actions):
            action = self.actions[self.idx]
            self.idx += 1
            return action
        return self.fallback(obs, info)


def controller_from_name(name: str, action_names: list[str], seed: int | None = None, epsilon: float = 0.08):
    normalized = name.lower().strip()
    if normalized == "random":
        return RandomController(len(action_names), seed=seed)
    if normalized in {"right", "right_biased", "heuristic"}:
        return RightBiasedController(action_names, seed=seed, epsilon=epsilon)
    if normalized in {"accel_jump", "run_jump", "obstacle"}:
        return AccelJumpController(action_names, seed=seed, epsilon=epsilon)
    if normalized == "mix":
        return MixedController(action_names, seed=seed, epsilon=epsilon)
    raise ValueError(f"Unknown controller '{name}'")


class MixedController:
    def __init__(self, action_names: list[str], seed: int | None = None, epsilon: float = 0.08) -> None:
        self.random = RandomController(len(action_names), seed=seed)
        self.heuristic = RightBiasedController(action_names, seed=seed, epsilon=epsilon)
        self.accel = AccelJumpController(action_names, seed=seed, epsilon=epsilon)
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs, info: dict) -> int:
        value = self.rng.random()
        if value < 0.10:
            return self.random(obs, info)
        if value < 0.50:
            return self.heuristic(obs, info)
        return self.accel(obs, info)


def jitter_windows(rng: np.random.Generator, windows: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    jittered = []
    for start, stop, min_hold, max_hold in windows:
        shift = int(rng.integers(-12, 13))
        jittered.append((start + shift, stop + shift, min_hold, max_hold))
    return jittered


def pick_action(action_names: list[str], candidates: list[str], fallback: int = 0) -> int:
    for candidate in candidates:
        for idx, name in enumerate(action_names):
            if normalize_action_name(name) == normalize_action_name(candidate):
                return idx
    return fallback


def pick_buttons(action_names: list[str], candidates: list[tuple[str, ...]], fallback: int = 0) -> int:
    for buttons in candidates:
        idx = buttons_to_action_idx(buttons, action_names, fallback=-1)
        if idx >= 0:
            return idx
    return fallback
