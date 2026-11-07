<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/branding/png/lockup-inverse.png">
    <img src="assets/branding/png/lockup.png" alt="Spectral Grid lockup for Bioacoustics DL Toolbox" width="560">
  </picture>
</p>

# Bioacoustics DL Toolbox

[![License: GPL-3.0-only](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)

**From environmental recordings to reproducible bioacoustic predictions.** A
headless Python/PyTorch library for researchers and developers who need audio
features, classifiers, training, evaluation, and recording-wide inference
without a desktop app, database, or workflow server.

The core workflow is **audio → model-input spectral features → train and
evaluate → sliding-window class predictions → optional Raven selection table**.
It supports binary, multiclass, and explicitly configured multilabel tasks;
window predictions have timestamps and probabilities, but are not automatically
merged into biological events.

## Install

Requires **Python 3.11–3.13**. Clone this repository and install from its root:

```bash
python -m pip install -e .
```

Find published wheels and source distributions on
[GitHub Releases](https://github.com/darizae/bioacoustics-dl-toolbox/releases);
this project does **not** publish to PyPI. `main` may contain changes not yet
released. The NumPy inspection API demonstrated below first ships with
`v0.3.0`; verify that the tagged GitHub Release and its wheel actually
exist before depending on it, rather than using a sibling or editable checkout.

Optional features can be installed independently:

```bash
python -m pip install -e '.[rf]'             # librosa-based random-forest features
python -m pip install -e '.[visualization]'  # legacy plotting/image helpers
python -m pip install -e '.[summaries]'      # TensorBoard summaries
```

The new NumPy inspection renderer needs only base dependencies, **not** the
`[visualization]` extra.

## Inspect a sound: NumPy spectrogram

A synthetic 1 kHz tone makes a small, reproducible inspection example. Input
samples are **(channels, frames)**, even for mono. The renderer returns raw RGB
pixels and source coordinates; it does not open a window or create a PNG.

```python
import numpy as np

from bioacoustics_dl_toolbox.visualization.spectrogram import (
    calculate_stft_power,
    rasterize_spectrogram,
)

sample_rate = 8_000
time_s = np.arange(sample_rate, dtype=np.float64) / sample_rate
mono = np.sin(2 * np.pi * 1_000 * time_s)[None, :]  # (1, 8000)

power = calculate_stft_power(
    mono, sample_rate_hz=sample_rate, window_samples=256, hop_samples=64
)
image = rasterize_spectrogram(
    power,
    width=320,
    height=120,
    frequency_min_hz=200,
    frequency_max_hz=3_000,
    contrast_db=50,
    palette="magma",
)

assert image.rgb.shape == (120, 320, 3)
assert image.rgb.dtype == np.uint8
assert image.pixel_edge_time_seconds(image.width) == 1.0
print(image.frequency_centers_hz[:3])  # STFT bin coordinates in Hz
```

The colors show **per-image, band-peak-relative** decibels, not calibrated
sound pressure. Frequency increases toward the top of the raster. The returned
`relative_db` array, frequency coordinates, and `pixel_edge_*` methods help
align displayed sound with a source recording. For exact framing, short clips,
stereo averaging, color stops, and axis semantics, see the
[headless spectrogram API guide](docs/HEADLESS_SPECTROGRAM.md).

## Train with a different spectrogram contract

The **model-input** pipeline is deliberately separate from inspection
rasterization:

| Use | Public entry point | Output and purpose |
| --- | --- | --- |
| **Training and inference** | `audio.build_feature_preprocessor`, used by `SpectrogramDataset` and the predictor | PyTorch features with configured pre-emphasis, frequency compression (linear, mel, or MFCC), and normalization; layout `(channels, time, frequency)` |
| **Researcher inspection** | `visualization.spectrogram.calculate_stft_power` and `rasterize_spectrogram` | NumPy spectral power, relative-dB raster, RGB `uint8` image, and explicit source/time/frequency coordinates |

**Do not feed inspection RGB pixels or relative-dB values into a trained
classifier.** These numerical representations have different windows,
scaling, frequency processing, and intended consumers. There is no claimed
equivalence between them.

For example, create normalized model features from a synthetic mono waveform
without touching the inspection renderer:

```python
import torch

from bioacoustics_dl_toolbox import NormalizationConfig, SpectrogramConfig, TaskConfig
from bioacoustics_dl_toolbox.audio import build_feature_preprocessor

spec = SpectrogramConfig(
    sample_rate=8_000,
    n_fft=256,
    hop_length=64,
    n_freq_bins=64,
    f_min=200,
    f_max=3_000,
    freq_compression="linear",
)
mono = torch.sin(2 * torch.pi * 1_000 * torch.arange(8_000) / 8_000).unsqueeze(0)
features = build_feature_preprocessor(spec, NormalizationConfig())(mono)

assert features.shape[0] == 1 and features.shape[-1] == 64
task = TaskConfig(task_type="multilabel")  # for recordings with overlapping labels
```

A model must be trained before it can produce meaningful detections. Use a
`SpectrogramDataset` with a validated `class_to_index` mapping, build an
encoder and classifier, train with `Trainer`, and call `predict_windows` on
recordings using the **same** feature configuration and audio window duration.
Multilabel decisions use independent sigmoid probabilities and thresholds;
single-label binary/multiclass decisions use softmax. `write_raven_tsv` can
export timestamped predictions as Raven-compatible selections.

## Public API map

| Module | What it provides |
| --- | --- |
| [`audio`](src/bioacoustics_dl_toolbox/audio/) | Audio loading, model features, datasets, and PCM WAV extraction |
| [`models`](src/bioacoustics_dl_toolbox/models/) | Encoders and classifier components |
| [`training`](src/bioacoustics_dl_toolbox/training/) | `Trainer`, checkpointing, early stopping |
| [`metrics`](src/bioacoustics_dl_toolbox/metrics/) | Classification and multilabel metrics |
| [`prediction`](src/bioacoustics_dl_toolbox/prediction/) | `predict_windows`, `WindowPrediction`, Raven export |
| [`visualization`](src/bioacoustics_dl_toolbox/visualization/) | Headless inspection raster plus optional plotting utilities |

## Reproducibility and checkpoints

Use a consistent `SpectrogramConfig`, `NormalizationConfig`, recording window
length, and class ordering for training and inference. `save_model` preserves
the established six-key `trained_model.pt` payload by default; optional
versioned metadata records task and preprocessing choices. Trainer resume
checkpoints are distinct from exported models. See
[release notes](RELEASE_NOTES.md) for what was actually published, and the
[release procedure](docs/RELEASING.md) for maintainer steps.

## Contributing and attribution

For local checks, install `python -m pip install -e '.[test]'`, then run:

```bash
python -m pytest
python -m ruff check src tests
python -m ruff format --check src tests
```

The repository keeps contributor/agent guidance in [`AGENTS.md`](AGENTS.md)
and original Spectral Grid sources, light/dark artwork, and asset provenance
in [`assets/branding/`](assets/branding/README.md). Branding previews are
not GUI or spectrogram screenshots: this is a headless library.

**License:** [GNU GPL-3.0-only](LICENSE). This project contains adapted
[ANIMAL-SPOT](https://github.com/ChristianBergler/ANIMAL-SPOT) material and
originated from [`alpaca-pipelines`](https://github.com/darizae/alpaca-pipelines).
See [`NOTICE.md`](NOTICE.md) for upstream attribution and the scientific citation.
