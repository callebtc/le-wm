from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

MARIO_DIR = Path(__file__).resolve().parent


def main(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.data, "r") as f:
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]
        next_x = f["next_x_pos"][:, 0]
        episodes = select_episodes(ep_len, ep_offset, next_x, args.count)
        for label, ep in episodes:
            start = int(ep_offset[ep])
            end = start + int(ep_len[ep])
            frames = f["pixels"][start:end]
            if len(frames) > args.max_frames:
                idx = np.linspace(0, len(frames) - 1, args.max_frames).astype(int)
                frames = frames[idx]
            path = out / f"mario_data_{label}_ep{ep:03d}.mp4"
            imageio.mimsave(path, frames, fps=args.fps, codec="libx264", macro_block_size=1)
            print(f"wrote {path}")


def select_episodes(ep_len, ep_offset, next_x, count: int):
    rows = []
    for ep, (start, length) in enumerate(zip(ep_offset, ep_len)):
        end = int(start + length)
        rows.append((ep, int(length), float(next_x[start:end].max()) if end > start else 0.0))
    picks = [("longest", max(rows, key=lambda r: r[1])[0]), ("furthest", max(rows, key=lambda r: r[2])[0])]
    for ep, *_ in rows:
        if len(picks) >= count:
            break
        if ep not in {p[1] for p in picks}:
            picks.append(("sample", ep))
    return picks[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render sample Mario dataset episodes")
    parser.add_argument("--data", default=str(MARIO_DIR / "data" / "mario.h5"))
    parser.add_argument("--output-dir", default=str(MARIO_DIR / "visualizations"))
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=500)
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
