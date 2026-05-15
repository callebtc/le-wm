from __future__ import annotations

from collections import deque
from itertools import product
from typing import Iterable

import numpy as np

from .game import ACTION_DELTAS, ACTION_NAMES, DOWN, LEFT, OPPOSITE, RIGHT, UP, SnakeGame


class RandomSnakeController:
    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, game: SnakeGame) -> int:
        actions = game.legal_actions()
        return int(actions[int(self.rng.integers(0, len(actions)))])


class ScriptedSnakeController:
    def __init__(self, actions: Iterable[int | str], fallback: str = "greedy") -> None:
        self.actions = [parse_action(action) for action in actions]
        self.idx = 0
        self.fallback = controller_from_name(fallback)

    def __call__(self, game: SnakeGame) -> int:
        if self.idx < len(self.actions):
            action = self.actions[self.idx]
            self.idx += 1
            return action
        return self.fallback(game)


class GreedySnakeController:
    """BFS to food; if blocked, pick the move with the most free space."""

    def __call__(self, game: SnakeGame) -> int:
        path = shortest_safe_path(game, game.food)
        if path:
            return path[0]
        return safest_action(game)


class EdgeCrossingController:
    """Bias motion toward wrap-around transitions for dataset coverage."""

    def __call__(self, game: SnakeGame) -> int:
        head_x, head_y = game.snake[0]
        preferred = []
        if head_x == 0:
            preferred.append(LEFT)
        if head_x == game.width - 1:
            preferred.append(RIGHT)
        if head_y == 0:
            preferred.append(UP)
        if head_y == game.height - 1:
            preferred.append(DOWN)
        preferred.extend(shortest_wrap_path_to_edge(game))
        for action in preferred:
            if action in game.legal_actions():
                sim = game.copy()
                if not sim.step(action).done:
                    return action
        return safest_action(game)


class MPCSnakeController:
    """Model-predictive controller using the exact Snake simulator as the model.

    This is the control interface that the learned world model can replace. It
    enumerates short action sequences, simulates them, and emits the first action
    from the best-scoring sequence.
    """

    def __init__(self, horizon: int = 4) -> None:
        self.horizon = horizon

    def __call__(self, game: SnakeGame) -> int:
        best_action = safest_action(game)
        best_value = -float("inf")
        legal = game.legal_actions()
        for first_action in legal:
            for suffix in product((UP, RIGHT, DOWN, LEFT), repeat=max(0, self.horizon - 1)):
                candidate = (first_action, *suffix)
                value = score_rollout(game, candidate)
                if value > best_value:
                    best_value = value
                    best_action = first_action
        return best_action


def controller_from_name(name: str, seed: int | None = None, horizon: int = 4):
    normalized = name.lower().strip()
    if normalized == "random":
        return RandomSnakeController(seed=seed)
    if normalized == "greedy":
        return GreedySnakeController()
    if normalized == "mpc":
        return MPCSnakeController(horizon=horizon)
    if normalized == "edge":
        return EdgeCrossingController()
    raise ValueError(f"Unknown controller '{name}'. Expected random, greedy, mpc, or edge.")


def parse_action(action: int | str) -> int:
    if isinstance(action, int):
        return action
    normalized = action.lower().strip()
    if normalized not in ACTION_NAMES:
        raise ValueError(f"Unknown action '{action}'. Expected one of {ACTION_NAMES}.")
    return ACTION_NAMES.index(normalized)


def shortest_safe_path(game: SnakeGame, target: tuple[int, int]) -> list[int]:
    start = game.snake[0]
    body = set(game.snake[:-1])
    queue = deque([(start, [])])
    seen = {start}
    while queue:
        pos, path = queue.popleft()
        if pos == target:
            return path
        for action in game.legal_actions() if not path else (UP, RIGHT, DOWN, LEFT):
            if path and action == OPPOSITE[path[-1]]:
                continue
            dx, dy = ACTION_DELTAS[action]
            nxt = ((pos[0] + dx) % game.width, (pos[1] + dy) % game.height)
            if nxt in seen or nxt in body:
                continue
            seen.add(nxt)
            queue.append((nxt, [*path, action]))
    return []


def safest_action(game: SnakeGame) -> int:
    best_action = game.direction
    best_area = -1
    for action in game.legal_actions():
        sim = game.copy()
        step = sim.step(action)
        if step.done:
            continue
        area = reachable_area(sim)
        if area > best_area:
            best_area = area
            best_action = action
    return best_action


def reachable_area(game: SnakeGame) -> int:
    start = game.snake[0]
    blocked = set(game.snake[:-1])
    queue = deque([start])
    seen = {start}
    while queue:
        x, y = queue.popleft()
        for dx, dy in ACTION_DELTAS.values():
            nxt = ((x + dx) % game.width, (y + dy) % game.height)
            if nxt in seen or nxt in blocked:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return len(seen)


def score_rollout(game: SnakeGame, actions: Iterable[int]) -> float:
    sim = game.copy()
    value = 0.0
    for depth, action in enumerate(actions):
        before_score = sim.score
        step = sim.step(action)
        if step.done:
            value -= 100.0 / (depth + 1)
            break
        if sim.score > before_score:
            value += 50.0 / (depth + 1)

    hx, hy = sim.snake[0]
    fx, fy = sim.food
    distance = toroidal_distance((hx, hy), (fx, fy), sim.width, sim.height)
    value += 0.02 * reachable_area(sim)
    value -= 0.2 * distance
    value += 2.0 * sim.score
    return value


def toroidal_distance(a: tuple[int, int], b: tuple[int, int], width: int, height: int) -> int:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return min(dx, width - dx) + min(dy, height - dy)


def shortest_wrap_path_to_edge(game: SnakeGame) -> list[int]:
    head_x, head_y = game.snake[0]
    distances = [
        (head_y, UP),
        (game.height - 1 - head_y, DOWN),
        (head_x, LEFT),
        (game.width - 1 - head_x, RIGHT),
    ]
    distances.sort(key=lambda item: item[0])
    return [action for _, action in distances]
