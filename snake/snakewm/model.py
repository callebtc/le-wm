from __future__ import annotations

import torch
from torch import nn


class SnakeDynamicsModel(nn.Module):
    """Small action-conditioned world model for Snake state channels."""

    def __init__(
        self,
        height: int,
        width: int,
        action_dim: int = 4,
        hidden_dim: int = 256,
        state_channels: int = 7,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.action_dim = action_dim
        self.state_channels = state_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(state_channels, 32, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, padding_mode="circular"),
            nn.ReLU(),
            nn.Flatten(),
        )
        encoded_dim = 64 * height * width
        self.trunk = nn.Sequential(
            nn.Linear(encoded_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.next_state = nn.Linear(hidden_dim, state_channels * height * width)
        self.reward = nn.Linear(hidden_dim, 1)
        self.done = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        if state.ndim != 4:
            raise ValueError("state must be BCHW")
        x = self.encoder(state.float())
        x = torch.cat([x, action.float()], dim=-1)
        h = self.trunk(x)
        return {
            "next_state_logits": self.next_state(h).view(-1, self.state_channels, self.height, self.width),
            "reward": self.reward(h),
            "done_logits": self.done(h),
        }


class SnakeLeWM(nn.Module):
    """LeWM-style latent world model for Snake pixels.

    The core model is reward-free: an encoder maps pixels to a compact latent,
    and an action-conditioned predictor predicts the next latent. Probe and
    policy heads are intentionally thin; they make the learned representation
    measurable and controllable without turning the core objective into pixel
    reconstruction.
    """

    def __init__(
        self,
        height: int = 12,
        width: int = 12,
        action_dim: int = 4,
        latent_dim: int = 192,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
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
        self.policy_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )
        self.body_head = nn.Linear(latent_dim, height * width)
        self.head_head = nn.Linear(latent_dim, height * width)
        self.food_head = nn.Linear(latent_dim, height * width)
        self.direction_head = nn.Linear(latent_dim, action_dim)

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        pixels = pixels.float()
        if pixels.max() > 2:
            pixels = pixels / 255.0
        features = self.backbone(pixels)
        return self.projector(features)

    def predict(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action_emb = self.action_encoder(action.float())
        return self.predictor(torch.cat([latent, action_emb], dim=-1))

    def probe(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "body_logits": self.body_head(latent).view(-1, self.height, self.width),
            "head_logits": self.head_head(latent),
            "food_logits": self.food_head(latent),
            "direction_logits": self.direction_head(latent),
            "policy_logits": self.policy_head(latent),
        }

    def forward(self, pixels: torch.Tensor, action: torch.Tensor) -> dict[str, torch.Tensor]:
        latent = self.encode(pixels)
        pred_latent = self.predict(latent, action)
        return {"latent": latent, "pred_latent": pred_latent, **self.probe(latent)}
