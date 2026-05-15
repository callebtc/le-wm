from __future__ import annotations

import torch
from torch import nn


class MarioLeWM(nn.Module):
    """Compact LeWM-style pixel latent model for Mario frames."""

    def __init__(self, action_dim: int, latent_dim: int = 256, hidden_dim: int = 768, frame_stack: int = 1, policy_dim: int | None = None) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.policy_dim = policy_dim if policy_dim is not None else action_dim
        self.latent_dim = latent_dim
        self.frame_stack = frame_stack
        self.encoder = nn.Sequential(
            nn.Conv2d(3 * frame_stack, 32, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.projector = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.predictor = nn.Sequential(
            nn.LayerNorm(2 * latent_dim),
            nn.Linear(2 * latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.policy_head = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, self.policy_dim))
        self.progress_head = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.reward_head = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.float()
        if pixels.max() > 2:
            pixels = pixels / 255.0
        return self.projector(self.encoder(pixels))

    def predict(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_emb = self.action_encoder(action.float())
        return self.predictor(torch.cat([latent, action_emb], dim=-1))

    def probe(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "policy_logits": self.policy_head(latent),
            "progress": self.progress_head(latent),
            "reward": self.reward_head(latent),
        }
