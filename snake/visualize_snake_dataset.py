from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

SNAKE_DIR = Path(__file__).resolve().parent


def main(args: argparse.Namespace) -> None:
    data = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(data, "r") as f:
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        wrapped = f["wrapped"][:, 0] if "wrapped" in f else np.zeros(len(f["pixels"]))
        collision = f["collision"][:, 0] if "collision" in f else np.zeros(len(f["pixels"]))
        scores = f["score"][:, 0]

        selected = select_episodes(ep_len, ep_offset, wrapped, collision, scores, args.count)
        for label, episode in selected:
            start = int(ep_offset[episode])
            end = start + int(ep_len[episode])
            frames = f["pixels"][start:end]
            if len(frames) > args.max_frames:
                idx = np.linspace(0, len(frames) - 1, args.max_frames).astype(int)
                frames = frames[idx]
            path = output_dir / f"snake_data_{label}_ep{episode}.mp4"
            imageio.mimsave(path, frames, fps=args.fps, codec="libx264", macro_block_size=1)
            print(f"wrote {path}")


def select_episodes(
    ep_len: np.ndarray,
    ep_offset: np.ndarray,
    wrapped: np.ndarray,
    collision: np.ndarray,
    scores: np.ndarray,
    count: int,
) -> list[tuple[str, int]]:
    per_episode = []
    for episode, (start, length) in enumerate(zip(ep_offset, ep_len)):
        end = int(start + length)
        ep_wrapped = int(wrapped[start:end].sum())
        ep_collision = bool(collision[start:end].sum())
        final_score = float(scores[end - 1]) if end > start else 0.0
        per_episode.append((episode, ep_wrapped, ep_collision, final_score, int(length)))

    picks: list[tuple[str, int]] = []
    edge = max(per_episode, key=lambda item: item[1])
    picks.append(("edge_crossing", edge[0]))

    failures = [item for item in per_episode if item[2]]
    if failures:
        picks.append(("failure", min(failures, key=lambda item: item[4])[0]))

    picks.append(("high_score", max(per_episode, key=lambda item: item[3])[0]))

    for episode, *_ in per_episode:
        if len(picks) >= count:
            break
        if episode not in {pick[1] for pick in picks}:
            picks.append(("sample", episode))
    return picks[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render example episodes from snake.h5")
    parser.add_argument("--data", default=str(SNAKE_DIR / "data" / "snake.h5"))
    parser.add_argument("--output-dir", default=str(SNAKE_DIR / "visualizations"))
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--fps", type=float, default=12.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
