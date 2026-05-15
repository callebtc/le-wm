from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import json
import os
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np

from mariowm import MarioGymnasiumEnv, action_index_to_button_vector, button_names_from_action_names, controller_from_name
from mariowm.controllers import AccelJumpController
from mariowm.actions import normalize_action_name

MARIO_DIR = Path(__file__).resolve().parent
ROW_KEYS = [
    "pixels", "next_pixels", "action", "action_idx", "expert_action", "expert_action_idx",
    "reward", "done", "terminated", "truncated", "x_pos", "next_x_pos", "delta_x",
    "coin_delta", "time_used", "episode_objective", "score", "coins", "time", "world",
    "stage", "level_id", "episode_idx", "step_idx",
]
LOG_FIELDS = [
    "attempt", "accepted_episode", "accepted", "reject_reason", "controller", "seed", "env_id",
    "length", "max_x", "max_score", "objective", "flag_get", "accepted_so_far",
]


def generate(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    base_transition, base_episode, existing_attrs = read_existing_metadata(output, args.append)
    vis_dir = Path(args.visualize_dir) if args.visualize_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    env_ids = [item.strip() for item in args.env_ids.split(",") if item.strip()]
    rows = {key: [] for key in ROW_KEYS}
    ep_len: list[int] = []
    ep_offset: list[int] = []
    offset = 0
    action_names: list[str] | None = None
    button_names: list[str] | None = None
    stats = []
    accepted = 0
    attempts = 0
    filters_enabled = has_acceptance_filter(args)
    max_attempts = args.max_attempts or (args.episodes * 20 if filters_enabled else args.episodes)
    num_workers = resolve_num_workers(args.num_workers)
    worker_config = make_worker_config(args, env_ids, capture_frames=args.visualize_episodes > 0 and num_workers == 1)
    if args.visualize_episodes > 0 and num_workers > 1:
        print("Parallel generation disables per-episode videos; rerun with --num-workers 1 to visualize accepted episodes", flush=True)
    print(f"Generating with num_workers={num_workers} max_attempts={max_attempts} target_accepted={args.episodes}", flush=True)
    log_file = None
    log_writer = None
    if args.log_csv:
        log_path = Path(args.log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a" if args.append_log else "w", newline="")
        log_writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        if not args.append_log or log_path.stat().st_size == 0:
            log_writer.writeheader()

    def process_result(result: dict) -> None:
        nonlocal accepted, attempts, offset, action_names, button_names
        attempts += 1
        if action_names is None:
            action_names = result["action_names"]
            button_names = result["button_names"]
            validate_append_metadata(existing_attrs, args.movement, action_names, button_names)
        elif action_names != result["action_names"] or button_names != result["button_names"]:
            raise ValueError("Worker produced mismatched action/button metadata")

        accepted_episode, reject_reason = accept_episode(args, result["max_x"], result["max_score"], result["objective"])
        global_episode = base_episode + accepted
        if accepted_episode:
            for row in result["rows"]["episode_objective"]:
                row[0] = float(result["objective"])
            for row in result["rows"]["episode_idx"]:
                row[0] = global_episode
            ep_offset.append(base_transition + offset)
            ep_len.append(result["length"])
            for key, values in rows.items():
                values.extend(result["rows"][key])
            offset += result["length"]
            stats.append((global_episode, result["env_id"], result["length"], result["max_x"], result["flag_get"], result["max_score"], result["objective"], result["controller"]))
            if vis_dir and result["frames"]:
                imageio.mimsave(vis_dir / f"mario_data_ep{global_episode:03d}_{result['env_id']}.mp4", result["frames"], fps=args.fps, codec="libx264", macro_block_size=1)
            accepted += 1

        if log_writer:
            log_writer.writerow({
                "attempt": result["attempt"],
                "accepted_episode": global_episode if accepted_episode else "",
                "accepted": int(accepted_episode),
                "reject_reason": reject_reason,
                "controller": result["controller"],
                "seed": result["seed"],
                "env_id": result["env_id"],
                "length": result["length"],
                "max_x": result["max_x"],
                "max_score": result["max_score"],
                "objective": float(result["objective"]),
                "flag_get": int(result["flag_get"]),
                "accepted_so_far": accepted,
            })
            log_file.flush()
        if args.log_every and (attempts % args.log_every == 0 or accepted_episode):
            rate = accepted / attempts if attempts else 0.0
            status = "accepted" if accepted_episode else f"rejected:{reject_reason}"
            print(f"attempts_done={attempts} accepted={accepted}/{args.episodes} rate={rate:.3f} last_attempt={result['attempt']} last={status} max_x={result['max_x']} score={result['max_score']} objective={result['objective']:.1f} controller={result['controller']}", flush=True)

    try:
        if num_workers == 1:
            next_attempt = 0
            while accepted < args.episodes and next_attempt < max_attempts:
                result = generate_episode_attempt(next_attempt, worker_config)
                next_attempt += 1
                process_result(result)
        else:
            executor = futures.ProcessPoolExecutor(max_workers=num_workers)
            pending: dict[futures.Future, int] = {}
            next_attempt = 0
            try:
                for _ in range(min(max_attempts, num_workers * max(1, args.prefetch_factor))):
                    pending[executor.submit(generate_episode_attempt, next_attempt, worker_config)] = next_attempt
                    next_attempt += 1
                while pending and accepted < args.episodes:
                    done, _ = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        process_result(future.result())
                        if accepted < args.episodes and next_attempt < max_attempts:
                            pending[executor.submit(generate_episode_attempt, next_attempt, worker_config)] = next_attempt
                            next_attempt += 1
            finally:
                for future in pending:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
    finally:
        if log_file:
            log_file.close()

    if accepted < args.episodes:
        print(f"Warning: accepted {accepted}/{args.episodes} episodes after {attempts} attempts", flush=True)
    if accepted == 0:
        print(f"No episodes accepted after {attempts} attempts; nothing written", flush=True)
        return

    arrays = rows_to_arrays(rows)
    attrs = {
        "env_ids": json.dumps(env_ids),
        "movement": args.movement,
        "action_names": json.dumps(action_names or []),
        "button_names": json.dumps(button_names or []),
        "action_encoding": "button_multihot_v1",
        "frame_shape": json.dumps([84, 84, 3]),
    }
    if args.append and output.exists():
        append_h5(output, arrays, np.asarray(ep_len, dtype=np.int64), np.asarray(ep_offset, dtype=np.int64), attrs, action_names or [])
    else:
        write_h5(output, arrays, np.asarray(ep_len, dtype=np.int64), np.asarray(ep_offset, dtype=np.int64), attrs, action_names or [])

    max_xs = [s[3] for s in stats]
    run_frac = run_action_fraction(rows["action_idx"], action_names or [])
    left_frac = button_action_fraction(rows["action_idx"], action_names or [], "left")
    down_frac = button_action_fraction(rows["action_idx"], action_names or [], "down")
    verb = "Appended" if args.append and base_transition else "Wrote"
    print(f"{verb} {output} with {offset} new transitions across {len(ep_len)} accepted episodes from {attempts} attempts")
    print(f"mean_max_x={float(np.mean(max_xs)):.1f} max_x={int(np.max(max_xs))} mean_len={float(np.mean(ep_len)):.1f}")
    print(f"run_or_run_jump_action_fraction={run_frac:.3f}")
    print(f"left_action_fraction={left_frac:.3f} down_action_fraction={down_frac:.3f}")


def pick_controller(args: argparse.Namespace, episode: int) -> str:
    if args.controller != "mix":
        return args.controller
    return ("accel_jump", "right_biased", "accel_jump", "right_biased", "random")[episode % 5]


def has_acceptance_filter(args: argparse.Namespace) -> bool:
    return bool(args.min_max_x > 0 or args.min_score > 0 or args.min_objective > 0)


def accept_episode(args: argparse.Namespace, max_x: float, max_score: float, objective: float) -> tuple[bool, str]:
    failed = []
    if args.min_max_x > 0 and max_x < args.min_max_x:
        failed.append("max_x")
    if args.min_score > 0 and max_score < args.min_score:
        failed.append("score")
    if args.min_objective > 0 and objective < args.min_objective:
        failed.append("objective")
    if not failed:
        return True, ""
    return False, "+".join(failed)


def resolve_num_workers(num_workers: int) -> int:
    if num_workers > 0:
        return num_workers
    return max(1, os.cpu_count() or 1)


def make_worker_config(args: argparse.Namespace, env_ids: list[str], capture_frames: bool) -> dict:
    return {
        "env_ids": env_ids,
        "movement": args.movement,
        "steps": args.steps,
        "controller": args.controller,
        "epsilon": args.epsilon,
        "coin_reward": args.coin_reward,
        "step_penalty": args.step_penalty,
        "flag_reward": args.flag_reward,
        "coin_objective": args.coin_objective,
        "time_objective": args.time_objective,
        "flag_objective": args.flag_objective,
        "max_stall_steps": args.max_stall_steps,
        "seed": args.seed,
        "capture_frames": capture_frames,
        "min_max_x": args.min_max_x,
        "min_score": args.min_score,
        "min_objective": args.min_objective,
    }


def generate_episode_attempt(attempt: int, config: dict) -> dict:
    env_ids = config["env_ids"]
    env_id = env_ids[attempt % len(env_ids)]
    seed = int(config["seed"] + attempt)
    env = MarioGymnasiumEnv(env_id=env_id, movement=config["movement"], max_stall_steps=config["max_stall_steps"])
    try:
        obs, info = env.reset(seed=seed)
        action_names = env.action_names
        button_names = button_names_from_action_names(action_names)
        controller_name = pick_controller_from_name(config["controller"], attempt)
        controller = controller_from_name(controller_name, action_names, seed=seed, epsilon=config["epsilon"])
        expert = AccelJumpController(action_names, seed=seed, epsilon=0.0)
        rows = {key: [] for key in ROW_KEYS}
        frames = []
        max_x = int(info.get("x_pos", 0))
        max_score = int(info.get("score", 0))
        start_time = int(info.get("time", 400))

        for step_idx in range(config["steps"]):
            action = int(controller(obs, info))
            expert_action = int(expert(obs, info))
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            x_pos = int(info.get("x_pos", 0))
            next_x = int(next_info.get("x_pos", x_pos))
            coins = int(info.get("coins", 0))
            next_coins = int(next_info.get("coins", coins))
            coin_delta = max(0, next_coins - coins)
            clipped_dx = float(max(-20, min(20, next_x - x_pos)))
            flag_get = bool(next_info.get("flag_get", False))
            custom_reward = clipped_dx + config["coin_reward"] * coin_delta - config["step_penalty"]
            if flag_get:
                custom_reward += config["flag_reward"]
            max_x = max(max_x, next_x)
            max_score = max(max_score, int(next_info.get("score", 0)))

            rows["pixels"].append(obs)
            rows["next_pixels"].append(next_obs)
            rows["action"].append(action_index_to_button_vector(action, action_names, button_names))
            rows["action_idx"].append(action)
            rows["expert_action"].append(action_index_to_button_vector(expert_action, action_names, button_names))
            rows["expert_action_idx"].append(expert_action)
            rows["reward"].append([float(custom_reward)])
            rows["done"].append([float(done)])
            rows["terminated"].append([float(terminated)])
            rows["truncated"].append([float(truncated)])
            rows["x_pos"].append([float(x_pos)])
            rows["next_x_pos"].append([float(next_x)])
            rows["delta_x"].append([clipped_dx])
            rows["coin_delta"].append([float(coin_delta)])
            rows["time_used"].append([float(start_time - int(next_info.get("time", start_time)))])
            rows["episode_objective"].append([0.0])
            rows["score"].append([float(next_info.get("score", 0))])
            rows["coins"].append([float(next_info.get("coins", 0))])
            rows["time"].append([float(next_info.get("time", 0))])
            rows["world"].append([int(next_info.get("world", 1))])
            rows["stage"].append([int(next_info.get("stage", 1))])
            rows["level_id"].append([env_ids.index(env_id)])
            rows["episode_idx"].append([-1])
            rows["step_idx"].append([step_idx])
            if config["capture_frames"]:
                frames.append(obs)

            obs, info = next_obs, next_info
            if done:
                break

        final_coins = int(info.get("coins", 0))
        final_time = int(info.get("time", 0))
        final_flag = bool(info.get("flag_get", False))
        objective = max_x + config["coin_objective"] * final_coins + config["time_objective"] * final_time
        if final_flag:
            objective += config["flag_objective"]
        length = len(rows["action_idx"])
        if not accepts_config_filter(config, max_x, max_score, objective):
            rows = {key: [] for key in ROW_KEYS}
            frames = []
        return {
            "attempt": attempt,
            "seed": seed,
            "env_id": env_id,
            "controller": controller_name,
            "rows": rows,
            "frames": frames,
            "length": length,
            "max_x": max_x,
            "max_score": max_score,
            "objective": float(objective),
            "flag_get": bool(final_flag),
            "action_names": action_names,
            "button_names": button_names,
        }
    finally:
        env.close()


def pick_controller_from_name(controller: str, attempt: int) -> str:
    if controller != "mix":
        return controller
    return ("accel_jump", "right_biased", "accel_jump", "right_biased", "random")[attempt % 5]


def accepts_config_filter(config: dict, max_x: float, max_score: float, objective: float) -> bool:
    if config["min_max_x"] > 0 and max_x < config["min_max_x"]:
        return False
    if config["min_score"] > 0 and max_score < config["min_score"]:
        return False
    if config["min_objective"] > 0 and objective < config["min_objective"]:
        return False
    return True


def read_existing_metadata(output: Path, append: bool) -> tuple[int, int, dict[str, str]]:
    if not append or not output.exists():
        return 0, 0, {}
    with h5py.File(output, "r") as f:
        return len(f["action"]), len(f["ep_len"]), {key: f.attrs[key] for key in f.attrs}


def validate_append_metadata(existing_attrs: dict[str, str], movement: str, action_names: list[str], button_names: list[str]) -> None:
    if not existing_attrs:
        return
    expected = {
        "movement": movement,
        "action_names": json.dumps(action_names),
        "button_names": json.dumps(button_names),
        "action_encoding": "button_multihot_v1",
    }
    for key, value in expected.items():
        if existing_attrs.get(key) != value:
            raise ValueError(f"Cannot append: existing {key}={existing_attrs.get(key)!r} but new {key}={value!r}")


def rows_to_arrays(rows: dict[str, list]) -> dict[str, np.ndarray]:
    arrays = {}
    int_keys = {"action_idx", "expert_action_idx", "world", "stage", "level_id", "episode_idx", "step_idx"}
    for key, values in rows.items():
        data = np.asarray(values)
        if key in {"pixels", "next_pixels"}:
            arrays[key] = data.astype(np.uint8)
        elif key in int_keys:
            arrays[key] = data.astype(np.int64)
        else:
            arrays[key] = data.astype(np.float32)
    return arrays


def write_h5(output: Path, arrays: dict[str, np.ndarray], ep_len: np.ndarray, ep_offset: np.ndarray, attrs: dict[str, str], action_names: list[str]) -> None:
    with h5py.File(output, "w") as f:
        for key, data in arrays.items():
            create_resizable_dataset(f, key, data)
        create_resizable_dataset(f, "ep_len", ep_len)
        create_resizable_dataset(f, "ep_offset", ep_offset)
        for key, value in attrs.items():
            f.attrs[key] = value
        update_fraction_attrs(f, action_names)


def append_h5(output: Path, arrays: dict[str, np.ndarray], ep_len: np.ndarray, ep_offset: np.ndarray, attrs: dict[str, str], action_names: list[str]) -> None:
    tmp = output.with_name(output.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with h5py.File(output, "r") as src, h5py.File(tmp, "w") as dst:
        for key, data in arrays.items():
            copy_and_append_dataset(src, dst, key, data)
        copy_and_append_dataset(src, dst, "ep_len", ep_len)
        copy_and_append_dataset(src, dst, "ep_offset", ep_offset)
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        for key, value in attrs.items():
            dst.attrs[key] = value
        update_fraction_attrs(dst, action_names)
    tmp.replace(output)


def create_resizable_dataset(f: h5py.File, key: str, data: np.ndarray) -> None:
    kwargs = {"maxshape": (None, *data.shape[1:]), "chunks": True}
    if key in {"pixels", "next_pixels"}:
        kwargs.update({"compression": "gzip", "compression_opts": 4})
    f.create_dataset(key, data=data, **kwargs)


def copy_and_append_dataset(src: h5py.File, dst: h5py.File, key: str, data: np.ndarray) -> None:
    old = src[key]
    if old.shape[1:] != data.shape[1:]:
        raise ValueError(f"Cannot append {key}: existing shape {old.shape[1:]} does not match new shape {data.shape[1:]}")
    total_shape = (old.shape[0] + data.shape[0], *old.shape[1:])
    kwargs = {"shape": total_shape, "dtype": old.dtype, "maxshape": (None, *old.shape[1:]), "chunks": True}
    if key in {"pixels", "next_pixels"}:
        kwargs.update({"compression": "gzip", "compression_opts": 4})
    out = dst.create_dataset(key, **kwargs)
    for start in range(0, old.shape[0], 1024):
        end = min(start + 1024, old.shape[0])
        out[start:end] = old[start:end]
    out[old.shape[0] :] = data.astype(old.dtype, copy=False)


def update_fraction_attrs(f: h5py.File, action_names: list[str]) -> None:
    actions = f["action_idx"][:]
    f.attrs["run_action_fraction"] = float(run_action_fraction(actions, action_names))
    f.attrs["left_action_fraction"] = float(button_action_fraction(actions, action_names, "left"))
    f.attrs["down_action_fraction"] = float(button_action_fraction(actions, action_names, "down"))


def run_action_fraction(actions: list[int | np.ndarray], action_names: list[str]) -> float:
    if len(actions) == 0 or not action_names:
        return 0.0
    count = 0
    for action in actions:
        idx = int(np.asarray(action).reshape(-1)[0])
        if idx < len(action_names) and "B" in normalize_action_name(action_names[idx]):
            count += 1
    return count / len(actions)


def button_action_fraction(actions: list[int | np.ndarray], action_names: list[str], button: str) -> float:
    if len(actions) == 0 or not action_names:
        return 0.0
    count = 0
    for action in actions:
        idx = int(np.asarray(action).reshape(-1)[0])
        if idx < len(action_names) and button in normalize_action_name(action_names[idx]).split("+"):
            count += 1
    return count / len(actions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Mario HDF5 pixel/action dataset")
    parser.add_argument("--output", default=str(MARIO_DIR / "data" / "mario.h5"))
    parser.add_argument("--append", action="store_true", help="Append new episodes to an existing HDF5 dataset instead of overwriting it")
    parser.add_argument("--env-ids", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--movement", choices=("right_only", "simple", "simple_down", "complex"), default="simple_down")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--min-max-x", type=float, default=0.0, help="Only keep episodes whose maximum x position reaches this value")
    parser.add_argument("--min-score", type=float, default=0.0, help="Only keep episodes whose maximum score reaches this value")
    parser.add_argument("--min-objective", type=float, default=0.0, help="Only keep episodes whose episode objective reaches this value")
    parser.add_argument("--max-attempts", type=int, default=0, help="Maximum candidate episodes to try; default is episodes, or episodes*20 when filtering")
    parser.add_argument("--num-workers", type=int, default=0, help="Parallel worker processes; 0 uses all available CPU cores")
    parser.add_argument("--prefetch-factor", type=int, default=1, help="Candidate episodes queued per worker")
    parser.add_argument("--controller", choices=("random", "right_biased", "heuristic", "accel_jump", "run_jump", "obstacle", "mix"), default="mix")
    parser.add_argument("--epsilon", type=float, default=0.06)
    parser.add_argument("--coin-reward", type=float, default=25.0)
    parser.add_argument("--step-penalty", type=float, default=0.02)
    parser.add_argument("--flag-reward", type=float, default=250.0)
    parser.add_argument("--coin-objective", type=float, default=25.0)
    parser.add_argument("--time-objective", type=float, default=0.5)
    parser.add_argument("--flag-objective", type=float, default=1000.0)
    parser.add_argument("--max-stall-steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visualize-dir", default=str(MARIO_DIR / "visualizations"))
    parser.add_argument("--visualize-episodes", type=int, default=4)
    parser.add_argument("--log-csv", default=str(MARIO_DIR / "outputs" / "mario_generation_log.csv"))
    parser.add_argument("--append-log", action="store_true", help="Append to --log-csv instead of replacing it")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--fps", type=float, default=20.0)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
