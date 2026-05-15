from __future__ import annotations

import argparse
import csv
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
from snakewm import SnakeGame
from snakewm.model import SnakeLeWM

SNAKE_DIR = Path(__file__).resolve().parent


class SnakeLeWMDataset(Dataset):
    def __init__(self, path: Path, indices: np.ndarray | None = None) -> None:
        self.path = Path(path)
        self.indices = indices
        self._h5: h5py.File | None = None
        with h5py.File(self.path, "r") as f:
            self.length = len(f["action"])
            self.height = int(f.attrs.get("height", f["state"].shape[1]))
            self.width = int(f.attrs.get("width", f["state"].shape[2]))

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        row = int(self.indices[idx]) if self.indices is not None else int(idx)
        pixels = torch.from_numpy(self._h5["pixels"][row]).permute(2, 0, 1).float()
        next_pixels = torch.from_numpy(self._h5["next_pixels"][row]).permute(2, 0, 1).float()
        state = torch.from_numpy(self._h5["state"][row]).permute(2, 0, 1).float()
        next_state = torch.from_numpy(self._h5["next_state"][row]).permute(2, 0, 1).float()
        action = torch.from_numpy(self._h5["action"][row]).float()
        expert_action_idx = int(self._h5["expert_action_idx"][row]) if "expert_action_idx" in self._h5 else int(self._h5["action_idx"][row])
        return {
            "pixels": pixels,
            "next_pixels": next_pixels,
            "state": state,
            "next_state": next_state,
            "action": action,
            "expert_action_idx": torch.tensor(expert_action_idx, dtype=torch.long),
        }


class InMemorySnakeLeWMDataset(Dataset):
    def __init__(self, path: Path, indices: np.ndarray) -> None:
        path = Path(path)
        sorted_indices = np.sort(indices)
        with h5py.File(path, "r") as f:
            self.height = int(f.attrs.get("height", f["state"].shape[1]))
            self.width = int(f.attrs.get("width", f["state"].shape[2]))
            print(f"Preloading {len(sorted_indices)} transitions from {path} into RAM...", flush=True)
            self.pixels = torch.from_numpy(f["pixels"][sorted_indices]).permute(0, 3, 1, 2).contiguous()
            self.next_pixels = torch.from_numpy(f["next_pixels"][sorted_indices]).permute(0, 3, 1, 2).contiguous()
            self.state = torch.from_numpy(f["state"][sorted_indices]).permute(0, 3, 1, 2).float().contiguous()
            self.next_state = torch.from_numpy(f["next_state"][sorted_indices]).permute(0, 3, 1, 2).float().contiguous()
            self.action = torch.from_numpy(f["action"][sorted_indices]).float().contiguous()
            action_key = "expert_action_idx" if "expert_action_idx" in f else "action_idx"
            self.expert_action_idx = torch.from_numpy(f[action_key][sorted_indices]).long().contiguous()

    def __len__(self) -> int:
        return len(self.action)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "pixels": self.pixels[idx],
            "next_pixels": self.next_pixels[idx],
            "state": self.state[idx],
            "next_state": self.next_state[idx],
            "action": self.action[idx],
            "expert_action_idx": self.expert_action_idx[idx],
        }


def train(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    base = SnakeLeWMDataset(data_path)
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(base))
    rng.shuffle(indices)
    if args.max_samples and args.max_samples < len(indices):
        indices = indices[: args.max_samples]
    val_len = max(1, int(len(indices) * args.val_split))
    val_idx = np.sort(indices[:val_len])
    train_idx = indices[val_len:]

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    report_device(device)

    if args.cache_in_memory:
        train_set = InMemorySnakeLeWMDataset(data_path, train_idx)
        val_set = InMemorySnakeLeWMDataset(data_path, val_idx)
    else:
        train_set = SnakeLeWMDataset(data_path, train_idx)
        val_set = SnakeLeWMDataset(data_path, val_idx)

    pin_memory = device == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)

    model = SnakeLeWM(height=base.height, width=base.width, latent_dim=args.latent_dim).to(device)
    start_epoch = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        start_epoch = int(payload.get("epoch", 0))
        print(f"Resumed model weights from {args.resume} at stored epoch={start_epoch}", flush=True)
    if device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} CUDA GPUs via DataParallel", flush=True)
        model = torch.nn.DataParallel(model)
    elif device == "mps":
        print("Using Apple Metal (MPS). PyTorch exposes Mac GPUs as one logical device: mps:0.", flush=True)
    else:
        print(f"Using device={device}", flush=True)

    sigreg = SIGReg(knots=args.sigreg_knots, num_proj=args.sigreg_proj).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    body_pos_weight = compute_body_pos_weight(data_path).to(device)

    metrics_path = Path(args.metrics_csv) if args.metrics_csv else None
    if metrics_path:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if not metrics_path.exists() or not args.append_metrics:
            write_metrics_header(metrics_path)

    for local_epoch in range(1, args.epochs + 1):
        epoch = start_epoch + local_epoch
        model.train()
        losses = []
        accs = []
        for step, batch in enumerate(train_loader, start=1):
            batch = to_device(batch, device)
            out, loss, metrics = forward_loss(model, sigreg, batch, args, body_pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            accs.append(metrics["policy_acc"])
            if args.log_every and step % args.log_every == 0:
                print(
                    f"epoch={epoch:03d} step={step:04d}/{len(train_loader)} "
                    f"loss={np.mean(losses[-args.log_every:]):.4f} device={device}",
                    flush=True,
                )

        val = evaluate(model, sigreg, val_loader, args, body_pos_weight, device)
        eval_model = model.module if isinstance(model, torch.nn.DataParallel) else model
        control = (
            evaluate_control(
                eval_model,
                device=device,
                seeds=list(range(args.control_eval_seeds)),
                steps=args.control_eval_steps,
                horizon=args.control_eval_horizon,
                mode=args.control_eval_mode,
            )
            if args.control_eval_seeds > 0
            else empty_control_metrics()
        )
        print(
            f"epoch={epoch:03d} train_loss={np.mean(losses):.4f} "
            f"train_policy_acc={np.mean(accs):.3f} val_loss={val['loss']:.4f} "
            f"val_policy_acc={val['policy_acc']:.3f} val_head_acc={val['head_acc']:.3f} "
            f"val_food_acc={val['food_acc']:.3f} control_score={control['score_mean']:.2f} "
            f"control_deaths={control['death_count']} device={device}",
            flush=True,
        )
        if metrics_path:
            append_metrics(
                metrics_path,
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "train_policy_acc": float(np.mean(accs)),
                    "val_loss": val["loss"],
                    "val_policy_acc": val["policy_acc"],
                    "val_head_acc": val["head_acc"],
                    "val_food_acc": val["food_acc"],
                    "control_score_mean": control["score_mean"],
                    "control_score_median": control["score_median"],
                    "control_score_max": control["score_max"],
                    "control_death_count": control["death_count"],
                    "control_steps": args.control_eval_steps,
                    "control_seeds": args.control_eval_seeds,
                    "control_mode": args.control_eval_mode,
                },
            )

    save_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(
        {
            "model_state_dict": save_model.cpu().state_dict(),
            "height": base.height,
            "width": base.width,
            "action_dim": 4,
            "latent_dim": args.latent_dim,
            "data": str(data_path),
            "epoch": start_epoch + args.epochs,
        },
        output,
    )
    print(f"Saved {output}", flush=True)


def report_device(device: str) -> None:
    print(f"torch={torch.__version__}", flush=True)
    print(f"mps_available={torch.backends.mps.is_available()} mps_built={torch.backends.mps.is_built()}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()} cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}", flush=True)
    if device == "mps":
        probe = torch.randn(256, 256, device="mps")
        probe = probe @ probe
        print(f"mps_smoke_device={probe.device}", flush=True)


METRIC_FIELDS = [
    "epoch",
    "train_loss",
    "train_policy_acc",
    "val_loss",
    "val_policy_acc",
    "val_head_acc",
    "val_food_acc",
    "control_score_mean",
    "control_score_median",
    "control_score_max",
    "control_death_count",
    "control_steps",
    "control_seeds",
    "control_mode",
]


def write_metrics_header(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()


def append_metrics(path: Path, row: dict[str, float | int | str]) -> None:
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writerow(row)


def forward_loss(model, sigreg, batch, args, body_pos_weight):
    z = model.encode(batch["pixels"])
    with torch.no_grad() if args.stop_next_grad else torch.enable_grad():
        z_next = model.encode(batch["next_pixels"])
    pred_z = model.predict(z, batch["action"])
    pred_loss = F.mse_loss(pred_z, z_next)
    sigreg_loss = sigreg(torch.stack([z, z_next], dim=0))

    current_probe = model.probe(z)
    next_probe = model.probe(pred_z)
    probe_loss = state_probe_loss(current_probe, batch["state"], body_pos_weight)
    pred_probe_loss = state_probe_loss(next_probe, batch["next_state"], body_pos_weight)
    policy_loss = F.cross_entropy(current_probe["policy_logits"], batch["expert_action_idx"])
    loss = (
        pred_loss
        + args.sigreg_weight * sigreg_loss
        + args.probe_weight * (probe_loss + pred_probe_loss)
        + args.policy_weight * policy_loss
    )
    with torch.no_grad():
        policy_acc = (current_probe["policy_logits"].argmax(dim=1) == batch["expert_action_idx"]).float().mean().item()
        metrics = {"policy_acc": policy_acc}
    return {"z": z, "pred_z": pred_z}, loss, metrics


def state_probe_loss(probe: dict[str, torch.Tensor], state: torch.Tensor, body_pos_weight: torch.Tensor) -> torch.Tensor:
    b, _, h, w = state.shape
    body_loss = F.binary_cross_entropy_with_logits(
        probe["body_logits"], state[:, 0], pos_weight=body_pos_weight
    )
    head_target = state[:, 1].reshape(b, h * w).argmax(dim=1)
    food_target = state[:, 2].reshape(b, h * w).argmax(dim=1)
    direction_target = state[:, 3:7].mean(dim=(2, 3)).argmax(dim=1)
    return (
        body_loss
        + F.cross_entropy(probe["head_logits"], head_target)
        + F.cross_entropy(probe["food_logits"], food_target)
        + F.cross_entropy(probe["direction_logits"], direction_target)
    )


@torch.no_grad()
def evaluate(model, sigreg, loader, args, body_pos_weight, device: str) -> dict[str, float]:
    model.eval()
    losses = []
    policy_accs = []
    head_accs = []
    food_accs = []
    for batch in loader:
        batch = to_device(batch, device)
        _, loss, metrics = forward_loss(model, sigreg, batch, args, body_pos_weight)
        z = model.encode(batch["pixels"])
        probe = model.probe(z)
        b, _, h, w = batch["state"].shape
        head_target = batch["state"][:, 1].reshape(b, h * w).argmax(dim=1)
        food_target = batch["state"][:, 2].reshape(b, h * w).argmax(dim=1)
        losses.append(float(loss.cpu()))
        policy_accs.append(metrics["policy_acc"])
        head_accs.append((probe["head_logits"].argmax(dim=1) == head_target).float().mean().item())
        food_accs.append((probe["food_logits"].argmax(dim=1) == food_target).float().mean().item())
    return {
        "loss": float(np.mean(losses)),
        "policy_acc": float(np.mean(policy_accs)),
        "head_acc": float(np.mean(head_accs)),
        "food_acc": float(np.mean(food_accs)),
    }


@torch.no_grad()
def evaluate_control(
    model: SnakeLeWM,
    device: str,
    seeds: list[int],
    steps: int,
    horizon: int,
    mode: str,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    scores = []
    lengths = []
    deaths = 0
    for seed in seeds:
        game = SnakeGame(seed=seed)
        for _ in range(steps):
            if game.done:
                break
            action = lewm_control_action(model, game, device=device, horizon=horizon, mode=mode)
            game.step(action)
        scores.append(float(game.score))
        lengths.append(float(len(game.snake)))
        deaths += int(game.done)
    if was_training:
        model.train()
    return {
        "score_mean": float(np.mean(scores)),
        "score_median": float(np.median(scores)),
        "score_max": float(np.max(scores)),
        "length_mean": float(np.mean(lengths)),
        "death_count": int(deaths),
    }


def empty_control_metrics() -> dict[str, float | int]:
    return {
        "score_mean": float("nan"),
        "score_median": float("nan"),
        "score_max": float("nan"),
        "length_mean": float("nan"),
        "death_count": 0,
    }


@torch.no_grad()
def lewm_control_action(
    model: SnakeLeWM,
    game: SnakeGame,
    device: str,
    horizon: int,
    mode: str,
) -> int:
    legal = game.legal_actions()
    pixels = torch.from_numpy(game.render_pixels()).permute(2, 0, 1).unsqueeze(0).float().to(device)
    z = model.encode(pixels)
    policy_logits = model.probe(z)["policy_logits"][0]
    if mode == "policy":
        ranked = sorted(legal, key=lambda action: float(policy_logits[action].detach().cpu()), reverse=True)
        return first_safe_action(game, ranked)
    return first_safe_action(game, [lewm_plan_action(model, game, z, policy_logits, legal, horizon), *legal])


def lewm_plan_action(
    model: SnakeLeWM,
    game: SnakeGame,
    z: torch.Tensor,
    policy_logits: torch.Tensor,
    legal: list[int],
    horizon: int,
) -> int:
    sequences = []
    for first in legal:
        for suffix in product(range(4), repeat=max(0, horizon - 1)):
            sequences.append((first, *suffix))
    seq = torch.tensor(sequences, dtype=torch.long, device=z.device)
    z_roll = z.repeat(seq.shape[0], 1)
    best_progress = torch.full((seq.shape[0],), -1e6, device=z.device)
    food_x, food_y = game.food
    food_idx = food_y * game.width + food_x
    for t in range(seq.shape[1]):
        action = torch.nn.functional.one_hot(seq[:, t], num_classes=4).float()
        z_roll = model.predict(z_roll, action)
        probe = model.probe(z_roll)
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
    score = best_progress + 0.25 * policy_logits[first_action]
    best_idx = int(score.argmax().detach().cpu().item())
    return int(seq[best_idx, 0].detach().cpu().item())


def first_safe_action(game: SnakeGame, actions: list[int]) -> int:
    for action in actions:
        sim = game.copy()
        if not sim.step(action).done:
            return action
    return actions[0]


def compute_body_pos_weight(path: Path) -> torch.Tensor:
    with h5py.File(path, "r") as f:
        body = f["state"][:, :, :, 0]
        positives = float(body.sum())
        total = float(body.size)
    negatives = total - positives
    return torch.tensor(min(50.0, max(1.0, negatives / max(1.0, positives))), dtype=torch.float32)


def to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Snake LeWM-style latent world model")
    parser.add_argument("--data", default=str(SNAKE_DIR / "data" / "snake.h5"))
    parser.add_argument("--output", default=str(SNAKE_DIR / "outputs" / "snake_lewm.pt"))
    parser.add_argument("--resume", default="", help="Checkpoint to load before continuing training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=192)
    parser.add_argument("--sigreg-weight", type=float, default=0.05)
    parser.add_argument("--sigreg-proj", type=int, default=256)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--probe-weight", type=float, default=0.15)
    parser.add_argument("--policy-weight", type=float, default=0.5)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-in-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--metrics-csv", default="", help="Append per-epoch metrics to this CSV")
    parser.add_argument("--append-metrics", action="store_true")
    parser.add_argument("--control-eval-seeds", type=int, default=0)
    parser.add_argument("--control-eval-steps", type=int, default=120)
    parser.add_argument("--control-eval-horizon", type=int, default=4)
    parser.add_argument("--control-eval-mode", choices=("policy", "planner"), default="planner")
    parser.add_argument("--stop-next-grad", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
