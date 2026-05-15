from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from module import SIGReg
from mariowm import MarioGymnasiumEnv, MarioLeWM, action_button_matrix, button_names_from_action_names

MARIO_DIR = Path(__file__).resolve().parent


class MarioDataset(Dataset):
    def __init__(self, path: Path, indices: np.ndarray, frame_stack: int = 1) -> None:
        self.path = Path(path)
        self.indices = np.sort(indices)
        self.frame_stack = frame_stack
        self._h5 = None
        with h5py.File(self.path, "r") as f:
            self.action_dim = f["action"].shape[1]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        row = int(self.indices[idx])
        return {
            "pixels": self.stack_pixels(row, "pixels"),
            "next_pixels": self.stack_pixels(row, "next_pixels"),
            "action": torch.from_numpy(self._h5["action"][row]).float(),
            "expert_action_idx": torch.tensor(int(self._h5["expert_action_idx"][row]), dtype=torch.long),
            "delta_x": torch.from_numpy(self._h5["delta_x"][row]).float() / 20.0,
            "reward": torch.from_numpy(self._h5["reward"][row]).float() / 20.0,
        }

    def stack_pixels(self, row: int, key: str) -> torch.Tensor:
        frames = []
        for offset in range(self.frame_stack - 1, -1, -1):
            idx = max(0, row - offset)
            frames.append(torch.from_numpy(self._h5[key][idx]).permute(2, 0, 1).float())
        return torch.cat(frames, dim=0)


class InMemoryMarioDataset(Dataset):
    def __init__(self, path: Path, indices: np.ndarray, frame_stack: int = 1) -> None:
        indices = np.sort(indices)
        self.frame_stack = frame_stack
        with h5py.File(path, "r") as f:
            print(f"Preloading {len(indices)} Mario transitions from {path}...", flush=True)
            pixels_all = f["pixels"][:]
            next_pixels_all = f["next_pixels"][:]
            self.pixels = stack_numpy_frames(pixels_all, indices, frame_stack, as_float=False)
            self.next_pixels = stack_numpy_frames(next_pixels_all, indices, frame_stack, as_float=False)
            self.action = torch.from_numpy(f["action"][:][indices]).float().contiguous()
            self.expert_action_idx = torch.from_numpy(f["expert_action_idx"][:][indices]).long().contiguous()
            self.delta_x = torch.from_numpy(f["delta_x"][:][indices]).float().contiguous() / 20.0
            self.reward = torch.from_numpy(f["reward"][:][indices]).float().contiguous() / 20.0
            self.action_dim = self.action.shape[1]

    def __len__(self) -> int:
        return len(self.action)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"pixels": self.pixels[idx], "next_pixels": self.next_pixels[idx], "action": self.action[idx], "expert_action_idx": self.expert_action_idx[idx], "delta_x": self.delta_x[idx], "reward": self.reward[idx]}


def train(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(data_path, "r") as f:
        total = len(f["action"])
        action_dim = f["action"].shape[1]
        action_names = json.loads(f.attrs.get("action_names", "[]"))
        button_names = json.loads(f.attrs.get("button_names", "[]")) or button_names_from_action_names(action_names)
        policy_dim = len(action_names)
    rng = np.random.default_rng(args.seed)
    indices = np.arange(total)
    rng.shuffle(indices)
    if args.max_samples and args.max_samples < len(indices):
        indices = indices[: args.max_samples]
    val_len = max(1, int(len(indices) * args.val_split))
    train_idx = indices[val_len:]
    val_idx = indices[:val_len]

    device = pick_device(args.device)
    report_device(device)
    dataset_cls = InMemoryMarioDataset if args.cache_in_memory else MarioDataset
    train_set = dataset_cls(data_path, train_idx, args.frame_stack)
    val_set = dataset_cls(data_path, val_idx, args.frame_stack)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)

    model = MarioLeWM(action_dim=action_dim, latent_dim=args.latent_dim, frame_stack=args.frame_stack, policy_dim=policy_dim).to(device)
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        print(f"Resumed from {args.resume}", flush=True)
    sigreg = SIGReg(knots=17, num_proj=args.sigreg_proj).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    policy_class_weight = compute_policy_class_weight(data_path, policy_dim).to(device)
    metrics_path = Path(args.metrics_csv) if args.metrics_csv else None
    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        write_metrics_header(metrics_path)
    best_control_x = -float("inf")
    best_output = Path(args.best_output) if args.best_output else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        accs = []
        for step, batch in enumerate(train_loader, start=1):
            batch = to_device(batch, device)
            loss, metrics = loss_fn(model, sigreg, batch, args, policy_class_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            accs.append(metrics["policy_acc"])
            if args.log_every and step % args.log_every == 0:
                print(f"epoch={epoch:03d} step={step:04d}/{len(train_loader)} loss={np.mean(losses[-args.log_every:]):.4f} device={device}", flush=True)
        val = evaluate(model, sigreg, val_loader, args, device, policy_class_weight)
        control = evaluate_control(model, device, action_names, button_names, args) if args.control_eval_episodes else empty_control()
        print(f"epoch={epoch:03d} train_loss={np.mean(losses):.4f} train_policy_acc={np.mean(accs):.3f} val_loss={val['loss']:.4f} val_policy_acc={val['policy_acc']:.3f} val_progress_mse={val['progress_mse']:.4f} control_x={control['x_mean']:.1f} device={device}", flush=True)
        if metrics_path:
            append_metrics(metrics_path, {"epoch": epoch, "train_loss": float(np.mean(losses)), "train_policy_acc": float(np.mean(accs)), "val_loss": val["loss"], "val_policy_acc": val["policy_acc"], "val_progress_mse": val["progress_mse"], "control_x_mean": control["x_mean"], "control_x_max": control["x_max"], "control_reward_mean": control["reward_mean"]})
        if best_output and control["x_mean"] > best_control_x:
            best_control_x = float(control["x_mean"])
            save_checkpoint(model, best_output, action_dim, policy_dim, args.latent_dim, args.frame_stack, action_names, button_names, data_path)
            print(f"Saved best checkpoint to {best_output} with control_x_mean={best_control_x:.1f}", flush=True)

    save_checkpoint(model, output, action_dim, policy_dim, args.latent_dim, args.frame_stack, action_names, button_names, data_path)
    print(f"Saved {output}")


def save_checkpoint(model, output: Path, action_dim: int, policy_dim: int, latent_dim: int, frame_stack: int, action_names: list[str], button_names: list[str], data_path: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model_state_dict": state_dict, "action_dim": action_dim, "policy_dim": policy_dim, "latent_dim": latent_dim, "frame_stack": frame_stack, "action_names": action_names, "button_names": button_names, "action_encoding": "button_multihot_v1", "data": str(data_path)}, output)


def loss_fn(model, sigreg, batch, args, policy_class_weight=None):
    z = model.encode(batch["pixels"])
    z_next = model.encode(batch["next_pixels"])
    z_pred = model.predict(z, batch["action"])
    pred_loss = F.mse_loss(z_pred, z_next)
    sigreg_loss = sigreg(torch.stack([z, z_next], dim=0))
    probe = model.probe(z)
    pred_probe = model.probe(z_pred)
    policy_loss = F.cross_entropy(probe["policy_logits"], batch["expert_action_idx"], weight=policy_class_weight)
    progress_loss = F.mse_loss(pred_probe["progress"], batch["delta_x"])
    reward_loss = F.mse_loss(pred_probe["reward"], batch["reward"])
    loss = pred_loss + args.sigreg_weight * sigreg_loss + args.policy_weight * policy_loss + args.progress_weight * progress_loss + args.reward_weight * reward_loss
    with torch.no_grad():
        acc = (probe["policy_logits"].argmax(dim=1) == batch["expert_action_idx"]).float().mean().item()
    return loss, {"policy_acc": acc, "progress_mse": float(progress_loss.detach().cpu())}


@torch.no_grad()
def evaluate(model, sigreg, loader, args, device, policy_class_weight=None):
    model.eval()
    losses, accs, mses = [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        loss, metrics = loss_fn(model, sigreg, batch, args, policy_class_weight)
        losses.append(float(loss.cpu()))
        accs.append(metrics["policy_acc"])
        mses.append(metrics["progress_mse"])
    return {"loss": float(np.mean(losses)), "policy_acc": float(np.mean(accs)), "progress_mse": float(np.mean(mses))}


@torch.no_grad()
def evaluate_control(model, device, action_names, button_names, args):
    scores, rewards = [], []
    for i in range(args.control_eval_episodes):
        env = MarioGymnasiumEnv(env_id=args.control_env_id, movement=args.movement, max_stall_steps=args.max_stall_steps)
        obs, info = env.reset(seed=args.seed + i)
        total_reward = 0.0
        max_x = 0
        frames = [obs]
        for _ in range(args.control_eval_steps):
            action = lewm_action(
                model,
                frames,
                device,
                action_names,
                button_names,
                args.control_eval_horizon,
                mode=args.control_eval_mode,
                progress_weight=args.planner_progress_weight,
                reward_weight=args.planner_reward_weight,
                policy_weight=args.planner_policy_weight,
                topk=args.planner_topk,
            )
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(obs)
            frames = frames[-model.frame_stack :]
            total_reward += reward
            max_x = max(max_x, int(info.get("x_pos", 0)))
            if terminated or truncated:
                break
        env.close()
        scores.append(max_x)
        rewards.append(total_reward)
    return {"x_mean": float(np.mean(scores)), "x_max": float(np.max(scores)), "reward_mean": float(np.mean(rewards))}


def empty_control():
    return {"x_mean": float("nan"), "x_max": float("nan"), "reward_mean": float("nan")}


def compute_policy_class_weight(path: Path, policy_dim: int) -> torch.Tensor:
    with h5py.File(path, "r") as f:
        labels = f["expert_action_idx"][:]
    counts = np.bincount(labels.astype(np.int64), minlength=policy_dim).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.from_numpy(np.clip(weights, 0.25, 8.0).astype(np.float32))


@torch.no_grad()
def lewm_action(
    model,
    obs_or_frames,
    device,
    action_names,
    button_names,
    horizon: int,
    mode: str,
    progress_weight: float = 20.0,
    reward_weight: float = 4.0,
    policy_weight: float = 0.0,
    topk: int = 4,
) -> int:
    pixels = stack_runtime_frames(obs_or_frames, model.frame_stack).unsqueeze(0).to(device)
    z = model.encode(pixels)
    logits = model.probe(z)["policy_logits"][0]
    if mode == "policy":
        return int(logits.argmax().cpu())
    candidates = range(len(action_names))
    if topk > 0:
        candidates = torch.topk(logits, k=min(topk, len(action_names))).indices.detach().cpu().tolist()
    sequences = torch.tensor(list(product(candidates, repeat=horizon)), device=device)
    z_roll = z.repeat(sequences.shape[0], 1)
    value = torch.zeros(sequences.shape[0], device=device)
    button_lookup = torch.as_tensor(action_button_matrix(action_names, button_names), device=device)
    for t in range(sequences.shape[1]):
        action = button_lookup[sequences[:, t]].float()
        z_roll = model.predict(z_roll, action)
        probe = model.probe(z_roll)
        value += progress_weight * probe["progress"].squeeze(-1) + reward_weight * probe["reward"].squeeze(-1)
    if policy_weight:
        value += policy_weight * logits[sequences[:, 0]]
    return int(sequences[int(value.argmax().cpu()), 0].cpu())


@torch.no_grad()
def lewm_macro_action(
    model,
    obs_or_frames,
    info: dict,
    device,
    action_names,
    button_names,
    progress_weight: float = 20.0,
    reward_weight: float = 4.0,
    policy_weight: float = 0.0,
) -> tuple[int, int]:
    pixels = stack_runtime_frames(obs_or_frames, model.frame_stack).unsqueeze(0).to(device)
    z = model.encode(pixels)
    logits = model.probe(z)["policy_logits"][0]
    button_lookup = torch.as_tensor(action_button_matrix(action_names, button_names), device=device)
    candidates = macro_candidates(action_names, int(info.get("x_pos", 0)))
    if not candidates:
        action = lewm_action(model, obs_or_frames, device, action_names, button_names, 5, "planner", progress_weight, reward_weight, policy_weight, 4)
        return action, 1

    scores = []
    for action_idx, duration, _label in candidates:
        z_roll = z
        action = button_lookup[action_idx].unsqueeze(0).float()
        score = torch.zeros((), device=device)
        for _ in range(duration):
            z_roll = model.predict(z_roll, action)
            probe = model.probe(z_roll)
            score = score + progress_weight * probe["progress"][0, 0] + reward_weight * probe["reward"][0, 0]
        if policy_weight:
            score = score + policy_weight * logits[action_idx]
        scores.append(score)
    best = int(torch.stack(scores).argmax().cpu())
    action_idx, duration, _label = candidates[best]
    return int(action_idx), int(duration)


def macro_candidates(action_names: list[str], x_pos: int) -> list[tuple[int, int, str]]:
    by_name = {name: idx for idx, name in enumerate(action_names)}

    def add(name: str, durations: list[int], out: list[tuple[int, int, str]]) -> None:
        if name in by_name:
            out.extend((by_name[name], duration, f"{name}x{duration}") for duration in durations)

    candidates: list[tuple[int, int, str]] = []
    if x_pos < 230:
        add("right+B", [8, 12, 16, 24], candidates)
        add("right", [4, 8], candidates)
    elif x_pos < 380:
        add("right+A+B", [16, 24, 32], candidates)
        add("right+B", [8, 12], candidates)
    elif x_pos < 650:
        add("right+B", [8, 12, 16, 24], candidates)
        add("right+A+B", [12, 16, 24], candidates)
    elif x_pos < 820:
        add("right+B", [8, 16, 24], candidates)
        add("right+A+B", [16, 24, 32], candidates)
    elif x_pos < 1010:
        add("right+A+B", [32, 48, 64], candidates)
        add("right+B", [8, 16], candidates)
    else:
        add("right+B", [8, 16, 24], candidates)
        add("right+A+B", [12, 24, 32], candidates)
        add("right", [4, 8], candidates)
    return candidates


def stack_runtime_frames(obs_or_frames, frame_stack: int) -> torch.Tensor:
    frames = obs_or_frames if isinstance(obs_or_frames, list) else [obs_or_frames]
    selected = []
    for offset in range(frame_stack - 1, -1, -1):
        idx = max(0, len(frames) - 1 - offset)
        selected.append(torch.from_numpy(frames[idx]).permute(2, 0, 1).float())
    return torch.cat(selected, dim=0)


def pick_device(device: str) -> str:
    if device != "auto":
        return device
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def report_device(device: str) -> None:
    print(f"torch={torch.__version__} mps={torch.backends.mps.is_available()} cuda={torch.cuda.is_available()} device={device}", flush=True)


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def stack_h5_frames(f, indices: np.ndarray, key: str, frame_stack: int) -> torch.Tensor:
    stacked = []
    for offset in range(frame_stack - 1, -1, -1):
        idx = np.maximum(0, indices - offset)
        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]
        unique_idx, inverse = np.unique(sorted_idx, return_inverse=True)
        data = torch.from_numpy(f[key][unique_idx]).permute(0, 3, 1, 2).float()
        restored_sorted = data[inverse]
        inverse_order = np.argsort(order, kind="stable")
        stacked.append(restored_sorted[inverse_order])
    return torch.cat(stacked, dim=1).contiguous()


def stack_numpy_frames(data: np.ndarray, indices: np.ndarray, frame_stack: int, as_float: bool = True) -> torch.Tensor:
    stacked = []
    for offset in range(frame_stack - 1, -1, -1):
        idx = np.maximum(0, indices - offset)
        tensor = torch.from_numpy(data[idx]).permute(0, 3, 1, 2)
        stacked.append(tensor.float() if as_float else tensor)
    return torch.cat(stacked, dim=1).contiguous()


FIELDS = ["epoch", "train_loss", "train_policy_acc", "val_loss", "val_policy_acc", "val_progress_mse", "control_x_mean", "control_x_max", "control_reward_mean"]


def write_metrics_header(path: Path) -> None:
    with path.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDS).writeheader()


def append_metrics(path: Path, row: dict) -> None:
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Mario LeWM-style latent model")
    parser.add_argument("--data", default=str(MARIO_DIR / "data" / "mario.h5"))
    parser.add_argument("--output", default=str(MARIO_DIR / "outputs" / "mario_lewm.pt"))
    parser.add_argument("--best-output", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--frame-stack", type=int, default=1)
    parser.add_argument("--sigreg-weight", type=float, default=0.04)
    parser.add_argument("--sigreg-proj", type=int, default=256)
    parser.add_argument("--policy-weight", type=float, default=0.6)
    parser.add_argument("--progress-weight", type=float, default=0.8)
    parser.add_argument("--reward-weight", type=float, default=0.2)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-in-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--metrics-csv", default=str(MARIO_DIR / "outputs" / "mario_lewm_metrics.csv"))
    parser.add_argument("--movement", choices=("right_only", "simple", "simple_down", "complex"), default="simple_down")
    parser.add_argument("--max-stall-steps", type=int, default=240)
    parser.add_argument("--control-eval-episodes", type=int, default=3)
    parser.add_argument("--control-eval-steps", type=int, default=600)
    parser.add_argument("--control-eval-horizon", type=int, default=3)
    parser.add_argument("--control-eval-mode", choices=("policy", "planner"), default="planner")
    parser.add_argument("--planner-progress-weight", type=float, default=20.0)
    parser.add_argument("--planner-reward-weight", type=float, default=4.0)
    parser.add_argument("--planner-policy-weight", type=float, default=0.0)
    parser.add_argument("--planner-topk", type=int, default=4)
    parser.add_argument("--control-env-id", default="SuperMarioBros-1-1-v0")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
