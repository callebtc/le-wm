from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import torch

from mariowm import MarioGymnasiumEnv, MarioLeWM, button_names_from_action_names, controller_from_name
from train_mario_lewm import lewm_action, lewm_macro_action, pick_device

MARIO_DIR = Path(__file__).resolve().parent


def main(args: argparse.Namespace) -> None:
    env = MarioGymnasiumEnv(env_id=args.env_id, movement=args.movement, max_stall_steps=args.max_stall_steps)
    obs, info = env.reset(seed=args.seed)
    if args.controller == "lewm":
        controller = LeWMController(Path(args.model), args.device, args.horizon, args.lewm_mode, args.planner_progress_weight, args.planner_reward_weight, args.planner_policy_weight, args.planner_topk)
    else:
        controller = controller_from_name(args.controller, env.action_names, seed=args.seed, epsilon=args.epsilon)
    frames = []
    actions = []
    total_reward = 0.0
    max_x = int(info.get("x_pos", 0))
    for _ in range(args.steps):
        frames.append(obs)
        action = int(controller(obs, info, env.action_names) if args.controller == "lewm" else controller(obs, info))
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        max_x = max(max_x, int(info.get("x_pos", 0)))
        if terminated or truncated:
            frames.append(obs)
            break
    env.close()

    video_path = Path(args.video)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(video_path, frames, fps=args.fps, codec="libx264", macro_block_size=1)
    actions_path = Path(args.actions_out)
    actions_path.parent.mkdir(parents=True, exist_ok=True)
    actions_path.write_text("\n".join(env.action_names[a] for a in actions) + "\n")
    print(f"controller={args.controller} max_x={max_x} reward={total_reward:.1f} steps={len(actions)} status={info.get('status')}")
    print(f"Wrote video to {video_path}")
    print(f"Wrote actions to {actions_path}")


class LeWMController:
    def __init__(self, checkpoint: Path, device: str, horizon: int, mode: str, progress_weight: float, reward_weight: float, policy_weight: float, topk: int) -> None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.device = pick_device(device)
        self.horizon = horizon
        self.mode = mode
        self.progress_weight = progress_weight
        self.reward_weight = reward_weight
        self.policy_weight = policy_weight
        self.topk = topk
        self.action_names = payload.get("action_names") or []
        self.button_names = payload.get("button_names") or button_names_from_action_names(self.action_names)
        self.model = MarioLeWM(action_dim=int(payload["action_dim"]), policy_dim=int(payload.get("policy_dim", len(self.action_names))), latent_dim=int(payload.get("latent_dim", 256)), frame_stack=int(payload.get("frame_stack", 1)))
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.to(self.device).eval()
        self.frames = []
        self.pending_actions = []

    @torch.no_grad()
    def __call__(self, obs, info, env_action_names):
        self.frames.append(obs)
        self.frames = self.frames[-self.model.frame_stack :]
        names = self.action_names or env_action_names
        button_names = self.button_names or button_names_from_action_names(names)
        if self.mode == "macro":
            if not self.pending_actions:
                action, duration = lewm_macro_action(self.model, self.frames, info, self.device, names, button_names, self.progress_weight, self.reward_weight, self.policy_weight)
                self.pending_actions = [action] * max(1, duration)
            return self.pending_actions.pop(0)
        return lewm_action(self.model, self.frames, self.device, names, button_names, self.horizon, self.mode, self.progress_weight, self.reward_weight, self.policy_weight, self.topk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mario controller and record video")
    parser.add_argument("--controller", choices=("random", "right_biased", "heuristic", "accel_jump", "run_jump", "obstacle", "mix", "lewm"), default="lewm")
    parser.add_argument("--model", default=str(MARIO_DIR / "outputs" / "mario_lewm.pt"))
    parser.add_argument("--lewm-mode", choices=("policy", "planner", "macro"), default="macro")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--planner-progress-weight", type=float, default=20.0)
    parser.add_argument("--planner-reward-weight", type=float, default=4.0)
    parser.add_argument("--planner-policy-weight", type=float, default=0.0)
    parser.add_argument("--planner-topk", type=int, default=4)
    parser.add_argument("--env-id", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--movement", choices=("right_only", "simple", "simple_down", "complex"), default="simple_down")
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-stall-steps", type=int, default=240)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--video", default=str(MARIO_DIR / "visualizations" / "mario_lewm_rollout.mp4"))
    parser.add_argument("--actions-out", default=str(MARIO_DIR / "outputs" / "mario_actions.txt"))
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
