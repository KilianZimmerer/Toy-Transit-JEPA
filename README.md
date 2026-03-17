# Toy-Transit-JEPA

A minimal PyTorch proof-of-concept: a **Joint-Embedding Predictive Architecture (JEPA)** that learns transit dynamics in latent space and detects anomalies via latent distance.

<p align="center">
  <img src="viz/circular_track.gif" width="600" alt="Circular track simulation"/>
</p>

## Overview

Trains move clockwise on a circular track. A context encoder $E_\theta$ maps each state to a latent vector, a predictor $P_\psi$ forecasts the next latent state, and a target encoder $E_\phi$ (EMA copy of $E_\theta$) provides the learning signal.

Encode the current state and predict the next latent state:

$$s_t = E_\theta(x_t), \quad \hat{s}\_{t+1} = P_\psi(s_t)$$

Encode the actual next state as the target:

$$s\_{t+1}^{\text{target}} = E_\phi(x\_{t+1})$$

Minimize the squared prediction error in latent space:

$$\mathcal{L} = \| \hat{s}\_{t+1} - s\_{t+1}^{\text{target}} \|_2^2$$

Update the target encoder via exponential moving average:

$$\phi \leftarrow \tau \phi + (1 - \tau)\theta \quad (\tau = 0.99)$$

When a train stalls, the prediction error spikes — longer stalls cause more trains to pile up, producing sustained elevated error.

## Anomaly Detection

The JEPA is trained on normal dynamics and then evaluated on perturbed scenarios where a single train stalls for varying durations (3, 5, 15 steps). Longer stalls cause trailing trains to pile up, producing a sustained spike in latent prediction error.

<p align="center">
  <img src="viz/anomaly_dashboard.png" width="700" alt="Anomaly detection dashboard"/>
</p>

## Usage

```bash
uv sync
uv run python -m src.eval    # train + evaluate → anomaly_dashboard.png
uv run pytest tests/          # run tests
```

## Structure

```
src/
  env.py             # Circular track environment (24 cells, 5 trains)
  jepa.py            # JEPA model (encoders + predictor)
  train.py           # Self-supervised training loop
  eval.py            # Evaluation scenarios + dashboard
tests/               # Property-based & unit tests
viz/
  generate_gif.py    # Animated GIF generation
```
