"""Tests for the JEPA architecture (jepa.py)."""

import torch

from src.jepa import ConditionEncoder, ContextEncoder, JEPA, Predictor, STATE_SIZE, CONDITION_SIZE, LATENT_SIZE


class TestContextEncoder:
    def test_output_shape(self):
        encoder = ContextEncoder()
        x = torch.randn(8, STATE_SIZE)
        assert encoder(x).shape == (8, LATENT_SIZE)


class TestConditionEncoder:
    def test_output_shape(self):
        encoder = ConditionEncoder()
        a = torch.randn(8, CONDITION_SIZE)
        assert encoder(a).shape == (8, LATENT_SIZE)


class TestPredictor:
    def test_output_shape(self):
        predictor = Predictor()
        s = torch.randn(8, LATENT_SIZE)
        c = torch.randn(8, LATENT_SIZE)
        assert predictor(s, c).shape == (8, LATENT_SIZE)


class TestJEPA:
    def test_forward_output_shapes(self):
        model = JEPA()
        x_t = torch.randn(8, STATE_SIZE)
        a_t = torch.randn(8, CONDITION_SIZE)
        x_next = torch.randn(8, STATE_SIZE)
        predicted, target = model(x_t, a_t, x_next)
        assert predicted.shape == (8, LATENT_SIZE)
        assert target.shape == (8, LATENT_SIZE)

    def test_target_encoder_no_grad(self):
        model = JEPA()
        for param in model.target_encoder.parameters():
            assert param.requires_grad is False

    def test_ema_update_correctness(self):
        model = JEPA()
        old_target = [p.data.clone() for p in model.target_encoder.parameters()]
        context_params = list(model.context_encoder.parameters())
        model.ema_update()
        for old_t, p_target, p_context in zip(
            old_target, model.target_encoder.parameters(), context_params
        ):
            expected = 0.99 * old_t + 0.01 * p_context.data
            assert torch.allclose(p_target.data, expected, atol=1e-6)

    def test_no_decoder(self):
        model = JEPA()
        child_names = [name for name, _ in model.named_children()]
        assert "decoder" not in child_names
