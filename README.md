# Bioacoustics DL Toolbox

Reusable PyTorch audio transforms, ResidualEncoder classifiers, metrics, training,
and checkpoint serialization for bioacoustic research. This package contains no
Alpaca Desktop or `alpaca-pipelines` workflow, persistence, UI, or run-management
code.

The repository was extracted with history from
[`alpaca-pipelines`](https://github.com/darizae/alpaca-pipelines) at source commit
`64f650645c373a4d1b9b47ea3ad48ec2d8bf3f16`. Much of the implementation is
adapted from [ANIMAL-SPOT](https://github.com/ChristianBergler/ANIMAL-SPOT), an
animal-independent ResNet-based bioacoustic classification framework. See
[`NOTICE.md`](NOTICE.md) for attribution and the research citation.

## Roadmap and release notes

- [**v0.2.0 implementation plan and agent onboarding**](0.2_VERSION.md) — implementation in progress; only completed workstreams are listed under Unreleased in the release notes.
- [**v0.2.0 workstream specifications**](specs/0.2/) — one spec per change, with unit-test acceptance criteria.
- [**Release notes**](RELEASE_NOTES.md) — ongoing change log, including explicitly marked planned work.

## Install locally

```bash
python -m pip install -e /path/to/bioacoustics-dl-toolbox
```

RF feature extraction is optional:

```bash
python -m pip install -e '/path/to/bioacoustics-dl-toolbox[rf]'
```

TensorBoard summaries and plotting are optional:

```bash
python -m pip install -e '/path/to/bioacoustics-dl-toolbox[summaries,visualization]'
```

## Classification tasks

`TaskConfig` makes the classification task explicit while preserving the legacy
single-label behavior. `auto` resolves two classes to **binary with two raw
logits** and more than two classes to **multiclass**; one-class `auto` is invalid.
An explicitly requested `multilabel` task accepts any positive class count and
is never inferred from output dimensions.

```python
from bioacoustics_dl_toolbox import TaskConfig, resolve_task_type

legacy = TaskConfig()  # auto: two classes -> binary, still two logits
assert resolve_task_type(legacy, num_classes=2) == "binary"
multilabel = TaskConfig(task_type="multilabel")
assert resolve_task_type(multilabel, num_classes=2) == "multilabel"
```

Pass the same `task_config` to `SpectrogramDataset` and `Trainer`. All classifiers
emit raw `[B, C]` logits, without a built-in activation. Binary and multiclass
use integer `[B]` targets and softmax/argmax decisions. For multilabel training,
annotate **each file** with an iterable of class names, in the same order as
`file_names`; an empty iterable denotes no active class. A bare string is not
an item annotation. Repeated names within an item count only once.

```python
import logging
import torch
from bioacoustics_dl_toolbox import TaskConfig
from bioacoustics_dl_toolbox.audio.datasets import SpectrogramDataset
from bioacoustics_dl_toolbox.training.trainer import Trainer

task = TaskConfig(task_type="multilabel")
train_dataset = SpectrogramDataset(
    file_names=["first.wav", "second.wav", "third.wav"],
    classes=["bird", "frog", "rain"],
    labels=[["bird", "rain"], [], ["frog", "frog"]],
    task_config=task,
)
# Targets: [1, 0, 1], [0, 0, 0], [0, 1, 0], all float32.
# Use a classifier model producing three raw logits per example.
trainer = Trainer(model, logging.getLogger(__name__), task_config=task)
loss_fn = torch.nn.BCEWithLogitsLoss()  # pass to trainer.fit(...)
```

Use one validated `class_to_index` mapping (indices `0..C-1`) for training,
validation, and testing. Multilabel targets are float32 `[B, C]` multi-hot
vectors; **never apply sigmoid before `BCEWithLogitsLoss`**. For multi-label
inference, call `multilabel_probabilities(logits)` to obtain independent sigmoid
probabilities, then `threshold_multilabel(probabilities, thresholds=0.5)` for
boolean decisions (inclusive `>=`, with a scalar or per-class threshold vector
in `class_to_index` order). Probabilities and decisions remain separate.
`MultiLabelMetrics` streams per-class TP/FP/TN/FN and support across batches;
available scalar scores are `micro_precision`, `micro_recall`, `micro_f1`,
`macro_precision`, `macro_recall`, and `macro_f1`. Zero-denominator scores are 0,
and macro averages include unsupported classes.

```python
from bioacoustics_dl_toolbox.metrics import (
    MultiLabelMetrics, multilabel_probabilities, threshold_multilabel,
)

probabilities = multilabel_probabilities(logits)  # shape [B, C], no softmax
predictions = threshold_multilabel(probabilities, [0.5, 0.7, 0.6])
metric = MultiLabelMetrics(num_classes=3, score="macro_f1")
metric.update(targets, predictions)  # repeat for every batch
scores_and_counts = metric.compute()  # class-index-ordered per_class breakdown

# Use with Trainer.fit(metrics={"macro_f1": metric},
#                      val_metric="macro_f1", val_metric_mode="max").
# Trainer(..., multilabel_thresholds=[0.5, 0.7, 0.6]) applies the same
# thresholds when making decisions during train/validation/test epochs.
```

Single-label ROC/confusion and per-sample class-image summaries remain disabled
for multi-label tasks; metrics with `supports_multilabel = True` can opt in.
The standalone sliding-window predictor is described below (SPEC-06).
Checkpoint threshold metadata remains planned (SPEC-07); binary positive-class
mapping is addressed separately in SPEC-08.

## Prediction (preprocessing)

`SpectrogramDataset` (training/validation) and `StridedAudioDataset` (windowed
inference) use the same deterministic `build_feature_preprocessor(spec_config,
norm_config)` stages: mono/resampled audio → pre-emphasis → power STFT
(`center=False`) → linear/mel/MFCC frequency representation → normalization.
Linear and mel representations apply `Amp2Db` before normalization; MFCC uses
`F2M → M2MFCC(n_mfcc=32)` **without a second dB conversion**. The default
feature layout remains `[channel, time_frames, frequency_bins]` (32 output bins
for MFCC). Only training with `aug_config.enabled=True` adds stochastic
spectrogram augmentation; validation and prediction must leave it disabled.

Use identical `SpectrogramConfig`, `NormalizationConfig`, and sample-window
length for training/validation and prediction. Training's `sequence_length`
is **feature frames**, while inference's `sequence_len` is **audio samples**.
For a window of `N` samples, match the frame count with:

```python
from bioacoustics_dl_toolbox.audio import spectrogram_frames_for_samples

sequence_length = spectrogram_frames_for_samples(N, spec_config)
# Pass sequence_length to SpectrogramDataset; pass N as sequence_len to
# StridedAudioDataset (with the same spec_config and norm_config).
```

Audio shorter than `n_fft` is cyclically repeated and center-trimmed to one
FFT frame (empty audio becomes zero). Shorter-than-sequence-length features
are symmetrically zero-padded **after** normalization; non-augmented training
center-crops longer feature sequences. Inference applies the same frame-padding
convention to incomplete tail windows. The cached `.spec` files contain only
pre-compression power spectrograms, so the existing cache format is unchanged.

## Prediction

`predict_windows` is a headless, reusable sliding-window API. It uses the same
mono/resampled deterministic preprocessing as training; single-label tasks use
softmax and one `argmax` label, and multilabel tasks use independent sigmoid
scores and validated, inclusive thresholds. Results are `WindowPrediction`
records with `index`, half-open `start_sample`/`end_sample`, recording-relative
`start_seconds`/`end_seconds`, and class-index-ordered `probabilities` and
`predicted_labels`. The final overlapping window is included and its end is
clamped to the resampled recording duration. Events are **not** merged.

```python
from bioacoustics_dl_toolbox import NormalizationConfig, SpectrogramConfig, TaskConfig
from bioacoustics_dl_toolbox.prediction import predict_windows

results = predict_windows(
    model,  # any torch.nn.Module returning raw [B, C] logits
    "recording.wav",
    class_to_index={"bird": 0, "frog": 1},
    task_config=TaskConfig(task_type="multilabel"),
    spec_config=SpectrogramConfig(),
    norm_config=NormalizationConfig(),
    window_samples=44100,
    hop_samples=22050,
    thresholds={"bird": 0.55, "frog": 0.7},  # optional; default is 0.5
    batch_size=16,
    device="cpu",  # optional; otherwise keep the module's existing device
)
```

The first argument may instead be a saved `trained_model.pt` **path** or loaded
bundle. The six-key legacy format supplies spectrogram settings and class indices
but **not** normalization, task type, window length, or inference hop; supply
`norm_config`, `task_config`, `window_samples`, and `hop_samples` explicitly.
Versioned v2 checkpoint inference metadata, when present, can fill these fields.
The previous model training/evaluation state is restored after prediction;
passing `device` moves the model to that device and leaves it there. This is a
library API, not a CLI, and does not require visualization dependencies.

## Lossless PCM WAV segments

`extract_pcm_wav_segment` copies a half-open, integer-millisecond interval from
an uncompressed PCM WAV without decoding, resampling, channel mixing, or sample
conversion. Endpoints use nearest-source-frame rounding (half frames round up),
the copy is chunk-bounded, and an existing destination is never overwritten.
Cancellation and copy failures remove the partial destination.

```python
from bioacoustics_dl_toolbox.audio import extract_pcm_wav_segment

facts = extract_pcm_wav_segment(
    "recording.wav",
    "snippet.wav",
    start_ms=1_250,
    end_ms=2_750,
    cancelled=lambda: False,
)
print(facts.start_frame, facts.end_frame, facts.sample_rate_hz)
```

## Checkpoint contract

`save_model` writes the established six-key `trained_model.pt` payload:

- `encoderConfig`
- `classifierConfig`
- `spectrogramConfig`
- `encoderState`
- `classifierState`
- `classes`

`load_model` uses CPU placement and PyTorch's weights-only loader. Consumers own
workflow provenance and any additional metadata such as sequence length,
normalization, and training metrics.

## Development

```bash
python -m pip install -e '.[test]'
python -m pytest
python -m ruff check src tests
python -m ruff format --check src tests
```

`CsvSplit` now uses cumulative partition boundaries and local seeded randomness, including seed `0`, while preserving file paths when reusing split CSVs.

## License

GPL-3.0-only. See [`LICENSE`](LICENSE).

