from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from snakewm import (
    ACTION_NAMES,
    GreedySnakeController,
    ScriptedSnakeController,
    SnakeGame,
    action_to_one_hot,
    controller_from_name,
)

CONTROLLER_IDS = {"random": 0, "greedy": 1, "mpc": 2, "edge": 3, "scripted": 4}


def build_dataset(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: dict[str, list[np.ndarray | int | float]] = {
        "pixels": [],
        "next_pixels": [],
        "state": [],
        "next_state": [],
        "action": [],
        "action_idx": [],
        "expert_action": [],
        "expert_action_idx": [],
        "reward": [],
        "done": [],
        "score": [],
        "wrapped": [],
        "edge_crossings": [],
        "collision": [],
        "controller_id": [],
        "episode_idx": [],
        "step_idx": [],
    }
    ep_len = []
    ep_offset = []
    offset = 0

    for episode in range(args.episodes):
        seed = args.seed + episode
        game = SnakeGame(width=args.width, height=args.height, seed=seed)
        controller_name = pick_controller(args, episode)
        if controller_name == "scripted":
            controller = ScriptedSnakeController(args.actions.split(",") if args.actions else [])
        else:
            controller = controller_from_name(controller_name, seed=seed, horizon=args.horizon)
        expert = GreedySnakeController()
        ep_offset.append(offset)
        steps = 0

        for step_idx in range(args.steps):
            obs = game.observation()
            expert_action = int(expert(game))
            action = int(controller(game))
            if args.epsilon > 0 and game.rng.random() < args.epsilon:
                action = int(game.rng.choice(game.legal_actions()))
            result = game.step(action)
            next_obs = result.observation

            rows["pixels"].append(obs["pixels"])
            rows["next_pixels"].append(next_obs["pixels"])
            rows["state"].append(obs["state"])
            rows["next_state"].append(next_obs["state"])
            rows["action"].append(action_to_one_hot(action))
            rows["action_idx"].append(action)
            rows["expert_action"].append(action_to_one_hot(expert_action))
            rows["expert_action_idx"].append(expert_action)
            rows["reward"].append([result.reward])
            rows["done"].append([float(result.done)])
            rows["score"].append([float(next_obs["score"])])
            rows["wrapped"].append([float(result.info.get("wrapped", False))])
            rows["edge_crossings"].append([float(result.info.get("edge_crossings", 0))])
            rows["collision"].append([float(result.info.get("collision", False))])
            rows["controller_id"].append([CONTROLLER_IDS[controller_name]])
            rows["episode_idx"].append([episode])
            rows["step_idx"].append([step_idx])
            steps += 1
            offset += 1

            if result.done:
                break

        ep_len.append(steps)

    with h5py.File(output, "w") as f:
        for key, values in rows.items():
            data = np.asarray(values)
            if key in {"pixels", "next_pixels"}:
                data = data.astype(np.uint8)
                f.create_dataset(key, data=data, compression="gzip", compression_opts=4)
            elif key in {"action_idx", "controller_id", "episode_idx", "expert_action_idx", "step_idx"}:
                f.create_dataset(key, data=data.astype(np.int64))
            else:
                f.create_dataset(key, data=data.astype(np.float32))
        f.create_dataset("ep_len", data=np.asarray(ep_len, dtype=np.int64))
        f.create_dataset("ep_offset", data=np.asarray(ep_offset, dtype=np.int64))
        f.attrs["width"] = args.width
        f.attrs["height"] = args.height
        f.attrs["controller"] = args.controller
        f.attrs["action_names"] = ",".join(ACTION_NAMES)
        f.attrs["controller_ids"] = str(CONTROLLER_IDS)
        f.attrs["seed"] = args.seed

    print(f"Wrote {output} with {offset} transitions across {len(ep_len)} episodes")
    final_scores = []
    scores = np.asarray(rows["score"], dtype=np.float32)[:, 0]
    episode_ids = np.asarray(rows["episode_idx"], dtype=np.int64)[:, 0]
    for episode in range(args.episodes):
        episode_scores = scores[episode_ids == episode]
        if len(episode_scores):
            final_scores.append(float(episode_scores[-1]))
    print(
        f"wrap_events={int(np.asarray(rows['wrapped']).sum())} "
        f"collisions={int(np.asarray(rows['collision']).sum())} "
        f"mean_final_score={float(np.mean(final_scores)):.3f}"
    )


def pick_controller(args: argparse.Namespace, episode: int) -> str:
    if args.controller != "mix":
        return args.controller
    cycle = ("greedy", "random", "edge", "mpc", "random", "edge")
    return cycle[episode % len(cycle)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate snake.h5 for LeWM and Snake world-model training")
    parser.add_argument("--output", default="data/snake.h5")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--height", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--controller", choices=("random", "greedy", "mpc", "edge", "scripted", "mix"), default="greedy")
    parser.add_argument("--actions", default="", help="Comma-separated actions for scripted dataset generation")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Random legal action injection for diversity")
    parser.add_argument("--horizon", type=int, default=4, help="MPC horizon for generated data")
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
