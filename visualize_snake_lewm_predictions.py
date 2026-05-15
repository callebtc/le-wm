from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

from snakewm.model import SnakeLeWM


def main(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    payload = torch.load(args.model, map_location="cpu", weights_only=False)
    model = SnakeLeWM(
        height=int(payload["height"]),
        width=int(payload["width"]),
        action_dim=int(payload.get("action_dim", 4)),
        latent_dim=int(payload.get("latent_dim", 192)),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.data, "r") as f, torch.no_grad():
        ep_len = int(f["ep_len"][args.episode])
        start = int(f["ep_offset"][args.episode])
        horizon = min(args.horizon, ep_len - 1)
        pixels = torch.from_numpy(f["pixels"][start]).permute(2, 0, 1).unsqueeze(0).float().to(device)
        z = model.encode(pixels)

        frames = []
        head_acc = []
        food_acc = []
        for t in range(horizon):
            row = start + t
            action = torch.from_numpy(f["action"][row]).unsqueeze(0).float().to(device)
            z_pred = model.predict(z, action)
            probe = model.probe(z_pred)
            pred_state = state_from_probe(probe, payload["height"], payload["width"])
            actual_state = torch.from_numpy(f["next_state"][row]).permute(2, 0, 1).numpy()

            head_acc.append(argmax_match(pred_state[1], actual_state[1]))
            food_acc.append(argmax_match(pred_state[2], actual_state[2]))
            frames.append(stack_frame(actual_state, pred_state))

            if args.teacher_forcing:
                next_pixels = torch.from_numpy(f["next_pixels"][row]).permute(2, 0, 1).unsqueeze(0).float().to(device)
                z = model.encode(next_pixels)
            else:
                z = z_pred

    imageio.mimsave(output, frames, fps=args.fps, codec="libx264", macro_block_size=1)
    print(f"wrote {output}")
    print(f"head_argmax_match={float(np.mean(head_acc)):.3f}")
    print(f"food_argmax_match={float(np.mean(food_acc)):.3f}")


def state_from_probe(probe: dict[str, torch.Tensor], height: int, width: int) -> np.ndarray:
    state = np.zeros((7, height, width), dtype=np.float32)
    body = torch.sigmoid(probe["body_logits"])[0].cpu().numpy()
    head_idx = int(probe["head_logits"].argmax(dim=1).cpu().item())
    food_idx = int(probe["food_logits"].argmax(dim=1).cpu().item())
    direction = int(probe["direction_logits"].argmax(dim=1).cpu().item())
    state[0] = body > 0.5
    state[1, head_idx // width, head_idx % width] = 1.0
    state[2, food_idx // width, food_idx % width] = 1.0
    state[3 + direction] = 1.0
    return state


def stack_frame(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    actual_img = render_state(actual)
    pred_img = render_state(pred)
    divider = np.full((actual_img.shape[0], 4, 3), 80, dtype=np.uint8)
    return np.hstack([actual_img, divider, pred_img])


def render_state(state: np.ndarray, cell_size: int = 8) -> np.ndarray:
    h, w = state.shape[1:]
    img = np.zeros((h * cell_size, w * cell_size, 3), dtype=np.uint8)
    body, head, food = state[0], state[1], state[2]
    for y in range(h):
        for x in range(w):
            value = max(body[y, x] * 130, head[y, x] * 255, food[y, x] * 210)
            if value > 0:
                img[y * cell_size : (y + 1) * cell_size, x * cell_size : (x + 1) * cell_size] = int(value)
    return img


def argmax_match(pred: np.ndarray, actual: np.ndarray) -> float:
    return float(int(np.argmax(pred) == np.argmax(actual)))


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Snake LeWM latent predictions")
    parser.add_argument("--data", default="data/snake.h5")
    parser.add_argument("--model", default="outputs/snake_lewm.pt")
    parser.add_argument("--output", default="visualizations/snake_lewm_prediction_teacher_forced.mp4")
    parser.add_argument("--episode", type=int, default=206)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--teacher-forcing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
