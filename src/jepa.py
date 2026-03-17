"""JEPA architecture: Context Encoder, Condition Encoder, Predictor, and Target Encoder."""

import copy

import torch
import torch.nn as nn

from src.env import CircularTrack

STATE_SIZE = CircularTrack.STATE_SIZE
CONDITION_SIZE = CircularTrack.CONDITION_SIZE
LATENT_SIZE = 16


class ContextEncoder(nn.Module):
    """MLP: STATE_SIZE → 64 → LATENT_SIZE. Maps track state to latent vector."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, LATENT_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionEncoder(nn.Module):
    """Linear: CONDITION_SIZE → LATENT_SIZE."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(CONDITION_SIZE, LATENT_SIZE)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.linear(a)


class Predictor(nn.Module):
    """MLP: 2*LATENT_SIZE → 64 → LATENT_SIZE."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * LATENT_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, LATENT_SIZE),
        )

    def forward(self, s: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, c], dim=-1))


class JEPA(nn.Module):
    """Unified JEPA module."""

    def __init__(self) -> None:
        super().__init__()
        self.context_encoder = ContextEncoder()
        self.condition_encoder = ConditionEncoder()
        self.predictor = Predictor()
        self.target_encoder = copy.deepcopy(self.context_encoder)
        self.tau = 0.99

        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def forward(
        self, x_t: torch.Tensor, a_t: torch.Tensor, x_next: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (predicted_latent, target_latent)."""
        s_t = self.context_encoder(x_t)
        c_t = self.condition_encoder(a_t)
        predicted = self.predictor(s_t, c_t)

        with torch.no_grad():
            target = self.target_encoder(x_next)

        return predicted, target

    def ema_update(self) -> None:
        """φ ← τφ + (1−τ)θ for all parameter pairs."""
        for p_target, p_context in zip(
            self.target_encoder.parameters(), self.context_encoder.parameters()
        ):
            p_target.data.mul_(self.tau).add_(p_context.data, alpha=1.0 - self.tau)
