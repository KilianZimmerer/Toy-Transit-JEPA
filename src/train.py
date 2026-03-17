"""Self-supervised training loop with MSE loss in latent space."""

import math

import torch
import torch.nn.functional as F

from src.env import CircularTrack
from src.jepa import JEPA


def train(
    num_epochs: int = 5000,
    batch_size: int = 256,
    lr: float = 1e-4,
) -> JEPA:
    """Train JEPA model. Returns trained model."""
    model = JEPA()

    optimizer = torch.optim.Adam(
        list(model.context_encoder.parameters())
        + list(model.condition_encoder.parameters())
        + list(model.predictor.parameters()),
        lr=lr,
    )

    for epoch in range(num_epochs):
        x_t, a_t, x_next = CircularTrack.generate_batch(batch_size)

        predicted, target = model(x_t, a_t, x_next)
        loss = F.mse_loss(predicted, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.ema_update()

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print(f"WARNING: Loss is {loss.item()} at epoch {epoch}. Stopping early.")
            break

    return model
