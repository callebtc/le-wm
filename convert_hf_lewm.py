import argparse
import json
from pathlib import Path

import stable_pretraining as spt
import stable_worldmodel as swm
import torch

from jepa import JEPA
from module import ARPredictor, Embedder, MLP


def strip_hydra(cfg):
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Directory with config.json and weights.pt")
    parser.add_argument("--out", required=True, help="Run name relative to STABLEWM_HOME, e.g. tworooms/lewm")
    args = parser.parse_args()

    src = Path(args.src).expanduser()
    out = Path(swm.data.utils.get_cache_dir(), f"{args.out}_object.ckpt")
    cfg = json.loads((src / "config.json").read_text())

    encoder_cfg = strip_hydra(cfg["encoder"])
    encoder = spt.backbone.utils.vit_hf(**encoder_cfg)

    def mlp(name):
        mlp_cfg = strip_hydra(cfg[name])
        mlp_cfg["norm_fn"] = torch.nn.BatchNorm1d
        return MLP(**mlp_cfg)

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**strip_hydra(cfg["predictor"])),
        action_encoder=Embedder(**strip_hydra(cfg["action_encoder"])),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )

    state_dict = torch.load(src / "weights.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, out)
    print(f"Converted checkpoint saved to: {out}")


if __name__ == "__main__":
    main()
