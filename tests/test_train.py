"""Tests for the training loop (train.py)."""

import torch
import torch.nn.functional as F

from src.env import CircularTrack
from src.jepa import JEPA
from src.train import train


def test_loss_decreases():
    model = JEPA()
    optimizer = torch.optim.Adam(
        list(model.context_encoder.parameters())
        + list(model.condition_encoder.parameters())
        + list(model.predictor.parameters()),
        lr=1e-3,
    )
    losses = []
    for _ in range(20):
        x_t, a_t, x_next = CircularTrack.generate_batch(128)
        predicted, target = model(x_t, a_t, x_next)
        loss = F.mse_loss(predicted, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.ema_update()
        losses.append(loss.item())
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5


def test_train_returns_jepa():
    model = train(num_epochs=5, batch_size=32, lr=1e-3)
    assert isinstance(model, JEPA)
