from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

from snakewm.model import SnakeDynamicsModel


def main(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model = SnakeDynamicsModel(
        height=int(checkpoint["height"]),
        width=int(checkpoint["width"]),
        action_dim=int(checkpoint.get("action_dim", 4)),
        state_channels=int(checkpoint.get("state_channels", 7)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.data, "r") as f, torch.no_grad():
        ep_len = int(f["ep_len"][args.episode])
        start = int(f["ep_offset"][args.episode])
        horizon = min(args.horizon, ep_len - 1)
        state = torch.from_numpy(f["state"][start]).permute(2, 0, 1).unsqueeze(0).float().to(device)
        frames = []
        exact_matches = []
        head_matches = []
        food_matches = []

        for t in range(horizon):
            action = torch.from_numpy(f["action"][start + t]).unsqueeze(0).float().to(device)
            out = model(state, action)
            pred = torch.sigmoid(out["next_state_logits"])
            actual = torch.from_numpy(f["next_state"][start + t]).permute(2, 0, 1).unsqueeze(0).float()

            pred_np = pred[0].cpu().numpy()
            actual_np = actual[0].numpy()
            exact_matches.append(compare_discrete_state(pred_np, actual_np))
            head_matches.append(argmax_match(pred_np[1], actual_np[1]))
            food_matches.append(argmax_match(pred_np[2], actual_np[2]))
            frames.append(stack_prediction_frame(actual_np, pred_np))

            if args.teacher_forcing:
                state = actual.to(device)
            elif args.project_state:
                state = project_state(pred).to(device)
            else:
                state = pred

    imageio.mimsave(output, frames, fps=args.fps, codec="libx264", macro_block_size=1)
    print(f"wrote {output}")
    print(f"mean_discrete_cell_match={float(np.mean(exact_matches)):.3f}")
    print(f"head_argmax_match={float(np.mean(head_matches)):.3f}")
    print(f"food_argmax_match={float(np.mean(food_matches)):.3f}")


def stack_prediction_frame(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    actual_img = render_state(actual)
    pred_img = render_state(pred)
    divider = np.full((actual_img.shape[0], 4, 3), 80, dtype=np.uint8)
    return np.hstack([actual_img, divider, pred_img])


def render_state(state: np.ndarray, cell_size: int = 8) -> np.ndarray:
    state = state.copy()
    h, w = state.shape[1:]
    img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)
    body = state[0]
    head = state[1]
    food = state[2]
    for y in range(h):
        for x in range(w):
            value = max(body[y, x] * 150, head[y, x] * 255, food[y, x] * 220)
            if value > 0.1:
                img[y * cell_size : (y + 1) * cell_size, x * cell_size : (x + 1) * cell_size] = int(
                    min(255, value)
                )
    return img


def compare_discrete_state(pred: np.ndarray, actual: np.ndarray) -> float:
    pred_binary = pred[:3] > 0.5
    actual_binary = actual[:3] > 0.5
    return float((pred_binary == actual_binary).mean())


def argmax_match(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(int(np.argmax(pred) == np.argmax(actual)))


def project_state(pred: torch.Tensor) -> torch.Tensor:
    projected = torch.zeros_like(pred)
    projected[:, 0] = (pred[:, 0] > 0.5).float()
    for batch in range(pred.shape[0]):
        head_idx = int(torch.argmax(pred[batch, 1]).item())
        food_idx = int(torch.argmax(pred[batch, 2]).item())
        direction_idx = int(torch.argmax(pred[batch, 3:7].mean(dim=(1, 2))).item())
        h, w = pred.shape[-2:]
        projected[batch, 1, head_idx // w, head_idx % w] = 1.0
        projected[batch, 2, food_idx // w, food_idx % w] = 1.0
        projected[batch, 3 + direction_idx] = 1.0
    return projected


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render actual-vs-predicted Snake model rollouts")
    parser.add_argument("--data", default="data/snake.h5")
    parser.add_argument("--model", default="outputs/snake_world_model.pt")
    parser.add_argument("--output", default="visualizations/snake_prediction_rollout.mp4")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--teacher-forcing", action="store_true")
    parser.add_argument("--project-state", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
