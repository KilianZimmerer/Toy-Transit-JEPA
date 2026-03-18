# Toy-Transit-JEPA

A minimal PyTorch proof-of-concept: a **Joint-Embedding Predictive Architecture (JEPA)** that learns transit dynamics in latent space and detects anomalies via latent distance.

<p align="center">
  <img src="viz/circular_track.gif" width="600" alt="Circular track simulation"/>
</p>

Trains move clockwise on a circular track. A context encoder $E_\theta$ maps each state to a latent vector, a predictor $P_\psi$ forecasts the next latent state, and a target encoder $E_\phi$ (EMA copy of $E_\theta$) provides the learning signal. The latent distance is then calculated as the squared prediction error in latent space.

## Contents

- [Usage](#usage)
- [Architecture](#architecture)
- [Anomaly Detection](#anomaly-detection)
- [Applications](#applications)
- [Structure](#structure)

## Usage

```bash
uv sync
uv run python -m src.eval    # train + evaluate → anomaly_dashboard.png
uv run pytest tests/          # run tests
```

## Architecture

Encode the current state and predict the next latent state:

$$s_t = E_\theta(x_t), \quad \hat{s}_{t+1} = P_\psi(s_t)$$

Encode the actual next state as the target:

$$s_{t+1}^{\text{target}} = E_\phi(x_{t+1})$$

Minimize the squared prediction error in latent space:

$$\mathcal{L} = \| \hat{s}_{t+1} - s_{t+1}^{\text{target}} \|_2^2$$

Update the target encoder via exponential moving average:

$$\phi \leftarrow \tau \phi + (1 - \tau)\theta \quad (\tau = 0.99)$$

When a train stalls, the prediction error spikes — longer stalls cause more trains to pile up, producing sustained elevated error.

## Anomaly Detection

The JEPA is trained on unperturbed dynamics and then evaluated on perturbed scenarios where a single train stalls for varying durations (3, 5, 15 steps). Longer stalls cause trailing trains to pile up, producing a sustained spike in latent prediction error.

<p align="center">
  <img src="viz/anomaly_dashboard.png" width="700" alt="Anomaly detection dashboard"/>
</p>

## Applications

If scaled, a Transit-JEPA can be adapted for:

* **Real-Time Simulations:** Fast "what-if" scenarios via latent-space arithmetic instead of Monte Carlo rollouts.
* **Anomaly Detection:** Latent distance spikes flag unexpected behavior in real time.
* **Congestion Forecasting:** Unroll the predictor $k$ steps ahead to anticipate bottlenecks before they form.
* **Safety Auditing:** Cluster historical states to map the safe operating space and warn when live states drift toward high-risk regions.
* **Disruption Recovery:** Search latent space for rescheduling options that minimize global delay.

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
