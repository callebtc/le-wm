from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main(args: argparse.Namespace) -> None:
    rows = read_rows(Path(args.metrics))
    if not rows:
        raise ValueError(f"No rows found in {args.metrics}")

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    food_score = [float(row["control_score_mean"]) for row in rows]
    food_score_median = [float(row["control_score_median"]) for row in rows]
    food_score_max = [float(row["control_score_max"]) for row in rows]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(epochs, train_loss, marker="o", label="train loss")
    axes[0].plot(epochs, val_loss, marker="o", label="val loss")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Snake LeWM continued training")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, food_score, marker="o", label="mean food score")
    axes[1].plot(epochs, food_score_median, marker="o", label="median food score")
    axes[1].plot(epochs, food_score_max, marker="o", label="max food score", alpha=0.7)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("food eaten")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output, dpi=160)
    print(f"Wrote {output}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Snake LeWM loss and control score over epochs")
    parser.add_argument("--metrics", default="outputs/snake_lewm_continue_metrics.csv")
    parser.add_argument("--output", default="visualizations/snake_lewm_training_curves.png")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
