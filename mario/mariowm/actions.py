from __future__ import annotations

import numpy as np

PREFERRED_BUTTON_ORDER = ("left", "right", "up", "down", "A", "B", "start", "select")


def action_name_to_buttons(name: str) -> tuple[str, ...]:
    if name == "NOOP":
        return ()
    return tuple(part for part in name.split("+") if part and part != "NOOP")


def normalize_action_name(name: str) -> str:
    buttons = action_name_to_buttons(name)
    if not buttons:
        return "NOOP"
    order = {button: idx for idx, button in enumerate(PREFERRED_BUTTON_ORDER)}
    ordered = sorted(buttons, key=lambda button: (order.get(button, len(order)), button))
    return "+".join(ordered)


def button_names_from_action_names(action_names: list[str]) -> list[str]:
    seen = {button for name in action_names for button in action_name_to_buttons(name)}
    order = {button: idx for idx, button in enumerate(PREFERRED_BUTTON_ORDER)}
    return sorted(seen, key=lambda button: (order.get(button, len(order)), button))


def action_index_to_button_vector(action_idx: int, action_names: list[str], button_names: list[str]) -> np.ndarray:
    buttons = set(action_name_to_buttons(action_names[int(action_idx)]))
    return np.asarray([1.0 if button in buttons else 0.0 for button in button_names], dtype=np.float32)


def action_button_matrix(action_names: list[str], button_names: list[str]) -> np.ndarray:
    return np.stack(
        [action_index_to_button_vector(idx, action_names, button_names) for idx in range(len(action_names))],
        axis=0,
    ).astype(np.float32)


def buttons_to_action_idx(buttons: list[str] | tuple[str, ...] | set[str], action_names: list[str], fallback: int = 0) -> int:
    target = normalize_action_name("+".join(buttons) if buttons else "NOOP")
    for idx, name in enumerate(action_names):
        if normalize_action_name(name) == target:
            return idx
    return fallback
