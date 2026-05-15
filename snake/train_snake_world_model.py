from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from snakewm.model import SnakeDynamicsModel

SNAKE_DIR = Path(__file__).resolve().parent


def load_dataset(path: Path) -> TensorDataset:
    with h5py.File(path, "r") as f:
        state = torch.from_numpy(f["state"][:]).permute(0, 3, 1, 2).float()
        action = torch.from_numpy(f["action"][:]).float()
        next_state = torch.from_numpy(f["next_state"][:]).permute(0, 3, 1, 2).float()
        reward = torch.from_numpy(f["reward"][:]).float()
        done = torch.from_numpy(f["done"][:]).float()
    return TensorDataset(state, action, next_state, reward, done)


def train(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(data_path)
    state0 = dataset.tensors[0]
    _, state_channels, height, width = state0.shape

    generator = torch.Generator().manual_seed(args.seed)
    val_len = max(1, int(len(dataset) * args.val_split))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=generator)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.num_workers,
        pin_memory=device_uses_cuda(args.device),
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.num_workers)

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    model = SnakeDynamicsModel(height=height, width=width, state_channels=state_channels).to(device)
    if device == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} CUDA GPUs via DataParallel")
        model = torch.nn.DataParallel(model)
    elif device == "mps":
        print("Using Apple Metal (MPS). PyTorch exposes this Mac GPU as one device.")
    else:
        print(f"Using device={device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    body_pos_weight = compute_pos_weight(dataset.tensors[2][:, :1]).to(device)
    done_pos_weight = compute_pos_weight(dataset.tensors[4]).to(device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for state, action, next_state, reward, done in train_loader:
            state = state.to(device)
            action = action.to(device)
            next_state = next_state.to(device)
            reward = reward.to(device)
            done = done.to(device)
            out = model(state, action)
            state_loss = snake_state_loss(out["next_state_logits"], next_state, body_pos_weight)
            reward_loss = F.mse_loss(out["reward"], reward)
            done_loss = F.binary_cross_entropy_with_logits(
                out["done_logits"], done, pos_weight=done_pos_weight
            )
            loss = state_loss + reward_loss + done_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch:03d} train_loss={np.mean(train_losses):.4f} val_loss={val_loss:.4f} device={device}"
        )

    save_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(
        {
            "model_state_dict": save_model.cpu().state_dict(),
            "height": height,
            "width": width,
            "state_channels": state_channels,
            "action_dim": 4,
            "data": str(data_path),
        },
        output,
    )
    print(f"Saved {output}")


@torch.no_grad()
def evaluate(model: SnakeDynamicsModel, loader: DataLoader, device: str) -> float:
    model.eval()
    losses = []
    for state, action, next_state, reward, done in loader:
        state = state.to(device)
        action = action.to(device)
        next_state = next_state.to(device)
        reward = reward.to(device)
        done = done.to(device)
        out = model(state, action)
        loss = snake_state_loss(out["next_state_logits"], next_state) + F.mse_loss(
            out["reward"], reward
        ) + F.binary_cross_entropy_with_logits(out["done_logits"], done)
        losses.append(float(loss.cpu()))
    return float(np.mean(losses))


def snake_state_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    body_pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    b, _, h, w = logits.shape
    body_loss = F.binary_cross_entropy_with_logits(
        logits[:, :1],
        target[:, :1],
        pos_weight=body_pos_weight.view(1, 1, 1, 1) if body_pos_weight is not None else None,
    )
    head_target = target[:, 1].reshape(b, -1).argmax(dim=1)
    food_target = target[:, 2].reshape(b, -1).argmax(dim=1)
    direction_target = target[:, 3:7].mean(dim=(2, 3)).argmax(dim=1)
    head_loss = F.cross_entropy(logits[:, 1].reshape(b, h * w), head_target)
    food_loss = F.cross_entropy(logits[:, 2].reshape(b, h * w), food_target)
    direction_loss = F.cross_entropy(logits[:, 3:7].mean(dim=(2, 3)), direction_target)
    return body_loss + head_loss + food_loss + direction_loss


def compute_pos_weight(target: torch.Tensor) -> torch.Tensor:
    reduce_dims = tuple(i for i in range(target.ndim) if i != 1)
    positives = target.sum(dim=reduce_dims)
    total = target.numel() / target.shape[1]
    negatives = total - positives
    return (negatives / positives.clamp_min(1.0)).clamp(1.0, 50.0).float()


def device_uses_cuda(device: str) -> bool:
    return device == "cuda" or (device == "auto" and torch.cuda.is_available() and not torch.backends.mps.is_available())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact action-conditioned Snake world model")
    parser.add_argument("--data", default=str(SNAKE_DIR / "data" / "snake.h5"))
    parser.add_argument("--output", default=str(SNAKE_DIR / "outputs" / "snake_world_model.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
