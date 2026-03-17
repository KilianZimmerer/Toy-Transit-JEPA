"""Tests for the evaluation module (eval.py)."""

import os
import tempfile

from src.env import CircularTrack
from src.eval import is_anomaly, plot_results, run_normal_scenario, run_perturbed_scenario
from src.jepa import JEPA


def _make_model():
    model = JEPA()
    model.eval()
    return model


def test_normal_scenario_returns_correct_count():
    distances = run_normal_scenario(_make_model(), CircularTrack(num_trains=3), num_steps=20)
    assert len(distances) == 20
    assert all(isinstance(d, float) and d >= 0 for d in distances)


def test_perturbed_scenario_returns_correct_count():
    distances = run_perturbed_scenario(_make_model(), CircularTrack(num_trains=3), num_steps=20)
    assert len(distances) == 20


def test_is_anomaly_flags_spike():
    normal = [1.0] * 20
    perturbed = [1.0] * 20
    perturbed[10] = 100.0
    assert is_anomaly(normal, perturbed, step=10) is True


def test_is_anomaly_no_false_positive():
    normal = [1.0] * 20
    perturbed = [1.0] * 20
    assert is_anomaly(normal, perturbed, step=5) is False


def test_plot_creates_file():
    normal = [0.5 + i * 0.01 for i in range(20)]
    perturbed = list(normal)
    perturbed[10] = 5.0
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.png")
        plot_results(normal, perturbed, path)
        assert os.path.isfile(path)
