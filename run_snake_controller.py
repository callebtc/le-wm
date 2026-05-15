from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from snakewm import ACTION_NAMES, SnakeGame, ScriptedSnakeController, controller_from_name
from snakewm.game import play_keyboard, run_episode
from snakewm.game import action_to_one_hot
from snakewm.model import SnakeDynamicsModel, SnakeLeWM


def main(args: argparse.Namespace) -> None:
    game = SnakeGame(width=args.width, height=args.height, seed=args.seed)
    if args.controller == "keyboard":
        play_keyboard(game, fps=args.fps)
        return

    if args.controller == "scripted":
        controller = ScriptedSnakeController(args.actions.split(",") if args.actions else [])
    elif args.controller == "model":
        controller = LearnedModelController(Path(args.model), device=args.device)
    elif args.controller == "lewm":
        controller = SnakeLeWMController(Path(args.lewm_model), device=args.device, horizon=args.horizon, mode=args.lewm_mode)
    else:
        controller = controller_from_name(args.controller, seed=args.seed, horizon=args.horizon)
    actions, frames, info = run_episode(game, controller, max_steps=args.steps, record=True)

    video_path = Path(args.video)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(video_path, frames, fps=args.fps, codec="libx264", macro_block_size=1)

    actions_path = Path(args.actions_out)
    actions_path.parent.mkdir(parents=True, exist_ok=True)
    actions_path.write_text("\n".join(ACTION_NAMES[action] for action in actions) + "\n")

    print(
        f"controller={args.controller} score={info['score']} length={info['length']} "
        f"steps={info['steps']} done={info['done']}"
    )
    print(f"Wrote video to {video_path}")
    print(f"Wrote control signal to {actions_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Snake under keyboard or autonomous control")
    parser.add_argument(
        "--controller",
        choices=("keyboard", "random", "greedy", "mpc", "edge", "scripted", "model", "lewm"),
        default="mpc",
    )
    parser.add_argument("--video", default="visualizations/snake_solution.mp4")
    parser.add_argument("--actions-out", default="outputs/snake_actions.txt")
    parser.add_argument("--actions", default="", help="Comma-separated actions for scripted mode")
    parser.add_argument("--model", default="outputs/snake_world_model.pt")
    parser.add_argument("--lewm-model", default="outputs/snake_lewm.pt")
    parser.add_argument("--lewm-mode", choices=("policy", "planner"), default="planner")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--horizon", type=int, default=4, help="MPC rollout horizon")
    return parser.parse_args()


class LearnedModelController:
    def __init__(self, checkpoint: Path, device: str = "auto") -> None:
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.device = device
        self.model = SnakeDynamicsModel(
            height=int(payload["height"]),
            width=int(payload["width"]),
            action_dim=int(payload.get("action_dim", 4)),
            state_channels=int(payload.get("state_channels", 7)),
        )
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(device).eval()

    @torch.no_grad()
    def __call__(self, game: SnakeGame) -> int:
        state = torch.from_numpy(game.state_channels()).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        best_action = game.legal_actions()[0]
        best_value = -float("inf")
        for action in game.legal_actions():
            action_tensor = torch.from_numpy(action_to_one_hot(action)).unsqueeze(0).to(self.device)
            out = self.model(state, action_tensor)
            reward = float(out["reward"].cpu().item())
            done_prob = float(torch.sigmoid(out["done_logits"]).cpu().item())
            probs = torch.sigmoid(out["next_state_logits"])[0].cpu()
            head_idx = int(torch.argmax(probs[1]).item())
            width = game.width
            hx, hy = head_idx % width, head_idx // width
            fx, fy = game.food
            distance = min(abs(hx - fx), game.width - abs(hx - fx)) + min(
                abs(hy - fy), game.height - abs(hy - fy)
            )
            value = 8.0 * reward - 10.0 * done_prob - distance
            if value > best_value:
                best_value = value
                best_action = action
        return best_action


class SnakeLeWMController:
    def __init__(self, checkpoint: Path, device: str = "auto", horizon: int = 5, mode: str = "planner") -> None:
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.device = device
        self.horizon = horizon
        self.mode = mode
        self.model = SnakeLeWM(
            height=int(payload["height"]),
            width=int(payload["width"]),
            action_dim=int(payload.get("action_dim", 4)),
            latent_dim=int(payload.get("latent_dim", 192)),
        )
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(device).eval()

    @torch.no_grad()
    def __call__(self, game: SnakeGame) -> int:
        legal = game.legal_actions()
        pixels = torch.from_numpy(game.render_pixels()).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        z = self.model.encode(pixels)
        policy_logits = self.model.probe(z)["policy_logits"][0]
        if self.mode == "policy":
            ranked = sorted(
                legal,
                key=lambda action: float(policy_logits[action].detach().cpu()),
                reverse=True,
            )
            return first_safe_action(game, ranked)
        return first_safe_action(game, [self.plan(game, z, policy_logits, legal), *legal])

    def plan(self, game: SnakeGame, z: torch.Tensor, policy_logits: torch.Tensor, legal: list[int]) -> int:
        sequences = []
        for first in legal:
            for suffix in product(range(4), repeat=max(0, self.horizon - 1)):
                sequences.append((first, *suffix))
        seq = torch.tensor(sequences, dtype=torch.long, device=self.device)
        z_roll = z.repeat(seq.shape[0], 1)
        best_progress = torch.full((seq.shape[0],), -1e6, device=self.device)
        food_x, food_y = game.food
        food_idx = food_y * game.width + food_x

        for t in range(seq.shape[1]):
            action = torch.nn.functional.one_hot(seq[:, t], num_classes=4).float()
            z_roll = self.model.predict(z_roll, action)
            probe = self.model.probe(z_roll)
            head_idx = probe["head_logits"].argmax(dim=1)
            head_x = head_idx % game.width
            head_y = torch.div(head_idx, game.width, rounding_mode="floor")
            dx = torch.minimum((head_x - food_x).abs(), game.width - (head_x - food_x).abs())
            dy = torch.minimum((head_y - food_y).abs(), game.height - (head_y - food_y).abs())
            distance = dx + dy
            reached = (head_idx == food_idx).float()
            step_score = 100.0 * reached - distance.float() - 0.03 * t
            best_progress = torch.maximum(best_progress, step_score)

        first_action = seq[:, 0]
        prior = policy_logits[first_action]
        score = best_progress + 0.25 * prior
        best_idx = int(score.argmax().detach().cpu().item())
        return int(seq[best_idx, 0].detach().cpu().item())


def first_safe_action(game: SnakeGame, actions: list[int]) -> int:
    for action in actions:
        sim = game.copy()
        if not sim.step(action).done:
            return action
    return actions[0]


if __name__ == "__main__":
    main(parse_args())
