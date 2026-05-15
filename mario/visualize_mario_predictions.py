from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

from mariowm import MarioLeWM
from train_mario_lewm import pick_device

MARIO_DIR = Path(__file__).resolve().parent


def main(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    payload = torch.load(args.model, map_location="cpu", weights_only=False)
    model = MarioLeWM(action_dim=int(payload["action_dim"]), policy_dim=int(payload.get("policy_dim", payload["action_dim"])), latent_dim=int(payload.get("latent_dim", 256)), frame_stack=int(payload.get("frame_stack", 1)))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    pred_dx = []
    actual_dx = []
    with h5py.File(args.data, "r") as f, torch.no_grad():
        start = int(f["ep_offset"][args.episode])
        length = min(args.horizon, int(f["ep_len"][args.episode]))
        z = None
        for i in range(length):
            row = start + i
            pixels = stack_frame(f, row, "pixels", model.frame_stack).unsqueeze(0).float().to(device)
            if z is None or args.teacher_forcing:
                z = model.encode(pixels)
            action = torch.from_numpy(f["action"][row]).unsqueeze(0).float().to(device)
            z = model.predict(z, action)
            probe = model.probe(z)
            dx_pred = float(probe["progress"].cpu().item() * 20.0)
            dx_actual = float(f["delta_x"][row, 0])
            pred_dx.append(dx_pred)
            actual_dx.append(dx_actual)
            frames.append(stack_visual_frame(f["pixels"][row], f["next_pixels"][row], dx_pred, dx_actual))
    imageio.mimsave(output, frames, fps=args.fps, codec="libx264", macro_block_size=1)
    mse = float(np.mean((np.asarray(pred_dx) - np.asarray(actual_dx)) ** 2))
    print(f"wrote {output}")
    print(f"delta_x_mse={mse:.3f}")


def stack_visual_frame(current, actual_next, pred_dx: float, actual_dx: float) -> np.ndarray:
    canvas = np.hstack([current, actual_next])
    bar = np.zeros((14, canvas.shape[1], 3), dtype=np.uint8)
    center = canvas.shape[1] // 2
    pred_len = int(np.clip(pred_dx, -20, 20) * 4)
    actual_len = int(np.clip(actual_dx, -20, 20) * 4)
    draw_bar(bar, center, 3, actual_len, (255, 255, 255))
    draw_bar(bar, center, 9, pred_len, (120, 120, 255))
    return np.vstack([canvas, bar])


def stack_frame(f, row: int, key: str, frame_stack: int) -> torch.Tensor:
    frames = []
    for offset in range(frame_stack - 1, -1, -1):
        idx = max(0, row - offset)
        frames.append(torch.from_numpy(f[key][idx]).permute(2, 0, 1).float())
    return torch.cat(frames, dim=0)


def draw_bar(img, center: int, y: int, length: int, color) -> None:
    x0, x1 = sorted((center, center + length))
    img[y : y + 3, max(0, x0) : min(img.shape[1], x1)] = color


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Mario LeWM prediction diagnostics")
    parser.add_argument("--data", default=str(MARIO_DIR / "data" / "mario.h5"))
    parser.add_argument("--model", default=str(MARIO_DIR / "outputs" / "mario_lewm.pt"))
    parser.add_argument("--output", default=str(MARIO_DIR / "visualizations" / "mario_prediction_teacher_forced.mp4"))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--teacher-forcing", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
