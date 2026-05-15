from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

UP, RIGHT, DOWN, LEFT = range(4)
ACTION_NAMES = ("up", "right", "down", "left")
ACTION_DELTAS = {
    UP: (0, -1),
    RIGHT: (1, 0),
    DOWN: (0, 1),
    LEFT: (-1, 0),
}
OPPOSITE = {UP: DOWN, RIGHT: LEFT, DOWN: UP, LEFT: RIGHT}


def action_to_one_hot(action: int) -> np.ndarray:
    one_hot = np.zeros(4, dtype=np.float32)
    one_hot[int(action)] = 1.0
    return one_hot


def one_hot_to_action(action: np.ndarray | Iterable[float] | int) -> int:
    if isinstance(action, (int, np.integer)):
        return int(action)
    return int(np.argmax(np.asarray(action)))


@dataclass
class SnakeStep:
    observation: dict[str, np.ndarray | int | float | bool]
    reward: float
    done: bool
    info: dict[str, int | bool | float]


class SnakeGame:
    """A small deterministic Snake simulator with seedable resets.

    The public interface intentionally mirrors a lightweight gym environment:
    call ``reset(seed=...)`` to initialize and ``step(action)`` to advance.
    Controllers can be keyboard input, scripts, or arbitrary callables.
    """

    def __init__(
        self,
        width: int = 12,
        height: int = 12,
        seed: int | None = None,
        start_length: int = 3,
        max_steps_without_food: int | None = None,
    ) -> None:
        if width < 6 or height < 6:
            raise ValueError("SnakeGame needs at least a 6x6 grid")
        if start_length < 2:
            raise ValueError("start_length must be >= 2")

        self.width = width
        self.height = height
        self.start_length = start_length
        self.max_steps_without_food = max_steps_without_food or width * height * 2
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self.reset(seed=seed)

    def copy(self) -> "SnakeGame":
        return copy.deepcopy(self)

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray | int | float | bool]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._seed = seed

        self.done = False
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.edge_crossings = 0
        self.last_reward = 0.0
        self.last_wrapped = False

        self.direction = int(self.rng.integers(0, 4))
        dx, dy = ACTION_DELTAS[self.direction]

        # Pick a head position that leaves room for the initial body behind it.
        valid_heads = []
        for y in range(self.height):
            for x in range(self.width):
                body = [
                    ((x - i * dx) % self.width, (y - i * dy) % self.height)
                    for i in range(self.start_length)
                ]
                if len(set(body)) == len(body):
                    valid_heads.append((x, y))
        head = valid_heads[int(self.rng.integers(0, len(valid_heads)))]
        self.snake = [
            ((head[0] - i * dx) % self.width, (head[1] - i * dy) % self.height)
            for i in range(self.start_length)
        ]
        self.food = self._spawn_food()
        return self.observation()

    def step(self, action: int | Iterable[float] | np.ndarray) -> SnakeStep:
        if self.done:
            return SnakeStep(self.observation(), 0.0, True, self.info())

        action_idx = one_hot_to_action(action)
        if len(self.snake) > 1 and action_idx == OPPOSITE[self.direction]:
            action_idx = self.direction

        self.direction = action_idx
        dx, dy = ACTION_DELTAS[action_idx]
        head_x, head_y = self.snake[0]
        raw_head = (head_x + dx, head_y + dy)
        new_head = self._wrap(raw_head)
        wrapped = raw_head != new_head
        self.last_wrapped = wrapped
        if wrapped:
            self.edge_crossings += 1

        self.steps += 1
        self.steps_since_food += 1
        reward = -0.01
        ate_food = False

        tail = self.snake[-1]
        body_without_tail = set(self.snake[:-1])
        if new_head in body_without_tail:
            self.done = True
            reward = -1.0
            self.last_reward = reward
            return SnakeStep(
                self.observation(),
                reward,
                True,
                self.info(collision=True, wrapped=wrapped),
            )

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            reward = 1.0
            ate_food = True
            self.steps_since_food = 0
            self.food = self._spawn_food()
        else:
            # Moving into the current tail is allowed because the tail moves away.
            if new_head == tail and tail in body_without_tail:
                self.done = True
                reward = -1.0
            self.snake.pop()

        if self.steps_since_food >= self.max_steps_without_food:
            self.done = True
            reward = -0.5

        self.last_reward = reward
        return SnakeStep(
            self.observation(),
            reward,
            self.done,
            self.info(ate_food=ate_food, wrapped=wrapped),
        )

    def legal_actions(self) -> list[int]:
        if self.done:
            return []
        actions = [UP, RIGHT, DOWN, LEFT]
        if len(self.snake) > 1:
            actions.remove(OPPOSITE[self.direction])
        return actions

    def observation(self) -> dict[str, np.ndarray | int | float | bool]:
        return {
            "state": self.state_channels(),
            "pixels": self.render_pixels(),
            "score": self.score,
            "direction": self.direction,
            "done": self.done,
            "reward": self.last_reward,
        }

    def info(self, **extra: int | bool | float) -> dict[str, int | bool | float]:
        info = {
            "score": self.score,
            "steps": self.steps,
            "length": len(self.snake),
            "direction": self.direction,
            "edge_crossings": self.edge_crossings,
            "wrapped": self.last_wrapped,
        }
        info.update(extra)
        return info

    def state_channels(self) -> np.ndarray:
        state = np.zeros((self.height, self.width, 7), dtype=np.float32)
        for x, y in self.snake[1:]:
            state[y, x, 0] = 1.0
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        state[head_y, head_x, 1] = 1.0
        state[food_y, food_x, 2] = 1.0
        state[:, :, 3 + self.direction] = 1.0
        return state

    def render_pixels(self, cell_size: int = 8, border: int = 2) -> np.ndarray:
        h = self.height * cell_size + 2 * border
        w = self.width * cell_size + 2 * border
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # Thin white retro border.
        img[:border, :, :] = 255
        img[-border:, :, :] = 255
        img[:, :border, :] = 255
        img[:, -border:, :] = 255

        for idx, (x, y) in enumerate(self.snake):
            x0 = border + x * cell_size
            y0 = border + y * cell_size
            pad = 1 if idx else 0
            img[y0 + pad : y0 + cell_size - pad, x0 + pad : x0 + cell_size - pad] = 255

        fx, fy = self.food
        x0 = border + fx * cell_size
        y0 = border + fy * cell_size
        inset = max(2, cell_size // 4)
        img[y0 + inset : y0 + cell_size - inset, x0 + inset : x0 + cell_size - inset] = 255
        return img

    def render_ansi(self) -> str:
        board = [["  " for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.snake[1:]:
            board[y][x] = "[]"
        hx, hy = self.snake[0]
        fx, fy = self.food
        board[hy][hx] = "@@"
        board[fy][fx] = "()"
        border = "+" + "--" * self.width + "+"
        rows = [border]
        rows.extend("|" + "".join(row) + "|" for row in board)
        rows.append(border)
        rows.append(f"score={self.score} length={len(self.snake)} steps={self.steps}")
        return "\n".join(rows)

    def _inside(self, pos: tuple[int, int]) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def _wrap(self, pos: tuple[int, int]) -> tuple[int, int]:
        x, y = pos
        return x % self.width, y % self.height

    def _spawn_food(self) -> tuple[int, int]:
        occupied = set(self.snake)
        free = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]
        if not free:
            self.done = True
            return self.snake[0]
        return free[int(self.rng.integers(0, len(free)))]


def run_episode(
    game: SnakeGame,
    controller: Callable[[SnakeGame], int],
    max_steps: int = 500,
    record: bool = False,
) -> tuple[list[int], list[np.ndarray], dict[str, int | bool | float]]:
    actions: list[int] = []
    frames: list[np.ndarray] = []
    for _ in range(max_steps):
        if record:
            frames.append(game.render_pixels())
        action = int(controller(game))
        actions.append(action)
        step = game.step(action)
        if step.done:
            if record:
                frames.append(game.render_pixels())
            break
    return actions, frames, game.info(done=game.done)


def play_keyboard(game: SnakeGame, fps: float = 8.0) -> None:
    import curses

    key_to_action = {
        curses.KEY_UP: UP,
        curses.KEY_RIGHT: RIGHT,
        curses.KEY_DOWN: DOWN,
        curses.KEY_LEFT: LEFT,
        ord("w"): UP,
        ord("d"): RIGHT,
        ord("s"): DOWN,
        ord("a"): LEFT,
    }

    def _loop(stdscr) -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(0)
        action = game.direction
        while not game.done:
            key = stdscr.getch()
            if key in (ord("q"), 27):
                break
            action = key_to_action.get(key, action)
            game.step(action)
            stdscr.erase()
            stdscr.addstr(0, 0, game.render_ansi())
            stdscr.addstr(game.height + 3, 0, "Use arrows/WASD, q to quit")
            stdscr.refresh()
            time.sleep(1.0 / fps)
        stdscr.nodelay(False)
        stdscr.addstr(game.height + 4, 0, "Game over. Press any key.")
        stdscr.getch()

    curses.wrapper(_loop)
