from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

MARIO_DIR = Path(__file__).resolve().parent


def main(args: argparse.Namespace) -> None:
    with Path(args.metrics).open(newline="") as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]
    train_loss = [float(r["train_loss"]) for r in rows]
    val_loss = [float(r["val_loss"]) for r in rows]
    control_x = [float(r["control_x_mean"]) for r in rows]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].plot(epochs, val_loss, marker="o", label="val")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].plot(epochs, control_x, marker="o", label="mean max x")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("max x position")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"Wrote {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Mario LeWM training curves")
    parser.add_argument("--metrics", default=str(MARIO_DIR / "outputs" / "mario_lewm_metrics.csv"))
    parser.add_argument("--output", default=str(MARIO_DIR / "visualizations" / "mario_training_curves.png"))
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
