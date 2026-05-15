from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


VIDEOS = [
    ("planner weak seed=0 score=3", Path("visualizations/snake_lewm_planner_weak_seed0.mp4")),
    ("planner median seed=16 score=20", Path("visualizations/snake_lewm_planner_median_seed16.mp4")),
    ("planner strong seed=28 score=23", Path("visualizations/snake_lewm_planner_strong_seed28.mp4")),
    ("policy strong seed=28 score=22", Path("visualizations/snake_lewm_policy_strong_seed28.mp4")),
]


def main() -> None:
    clips = []
    for label, path in VIDEOS:
        frames = imageio.mimread(path)
        clips.append((label, [add_label(frame, label) for frame in frames]))

    max_len = max(len(frames) for _, frames in clips)
    padded = []
    for _, frames in clips:
        if len(frames) < max_len:
            frames = [*frames, *([frames[-1]] * (max_len - len(frames)))]
        padded.append(frames)

    out_frames = []
    for i in range(max_len):
        top = np.hstack([padded[0][i], padded[1][i]])
        bottom = np.hstack([padded[2][i], padded[3][i]])
        out_frames.append(np.vstack([top, bottom]))

    output = Path("visualizations/snake_lewm_showcase.mp4")
    imageio.mimsave(output, out_frames, fps=10, codec="libx264", macro_block_size=1)
    print(f"Wrote {output}")


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    frame = np.asarray(frame).copy()
    banner_h = 16
    banner = np.zeros((banner_h, frame.shape[1], 3), dtype=np.uint8)
    draw_text(banner, 3, 3, label)
    return np.vstack([banner, frame])


def draw_text(img: np.ndarray, x: int, y: int, text: str) -> None:
    # Tiny block font sufficient for video labels without external dependencies.
    cursor = x
    for ch in text.lower():
        glyph = FONT.get(ch, FONT[" "])
        for gy, row in enumerate(glyph):
            for gx, val in enumerate(row):
                if val == "1":
                    img[y + gy : y + gy + 1, cursor + gx : cursor + gx + 1] = 255
        cursor += 4


FONT = {
    " ": ["000", "000", "000", "000", "000"],
    "=": ["000", "111", "000", "111", "000"],
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "010", "010", "010"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    "a": ["010", "101", "111", "101", "101"],
    "c": ["111", "100", "100", "100", "111"],
    "d": ["110", "101", "101", "101", "110"],
    "e": ["111", "100", "111", "100", "111"],
    "g": ["111", "100", "101", "101", "111"],
    "i": ["111", "010", "010", "010", "111"],
    "k": ["101", "101", "110", "101", "101"],
    "l": ["100", "100", "100", "100", "111"],
    "m": ["101", "111", "111", "101", "101"],
    "n": ["101", "111", "111", "111", "101"],
    "o": ["111", "101", "101", "101", "111"],
    "p": ["111", "101", "111", "100", "100"],
    "r": ["110", "101", "110", "101", "101"],
    "s": ["111", "100", "111", "001", "111"],
    "t": ["111", "010", "010", "010", "010"],
    "u": ["101", "101", "101", "101", "111"],
    "w": ["101", "101", "111", "111", "101"],
    "y": ["101", "101", "111", "010", "010"],
}


if __name__ == "__main__":
    main()
