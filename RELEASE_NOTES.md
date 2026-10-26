# Release notes

This is the ongoing change log for `bioacoustics-dl-toolbox`. See
[`docs/RELEASING.md`](docs/RELEASING.md) for the maintainer release process.

## Unreleased

_No changes yet._

## 0.2.0 — 2026-10-26

### Added

- Added bounded, no-clobber PCM WAV segment extraction with exact frame/format
  facts and cooperative cancellation.
- Added an explicit classification task contract for binary, multiclass, and
  multilabel learning while preserving legacy two-logit binary models.
- Added multi-hot audio labels and BCE-with-logits training/validation support
  for simultaneous bioacoustic classes.
- Added independent sigmoid/threshold multilabel predictions with per-class
  counts and micro/macro precision, recall, and F1 metrics.
- Added a reusable, headless sliding-window predictor with class-named
  probabilities, task-aware predictions, and recording-relative timestamps.
- Added optional versioned exported-model metadata for reproducible task,
  normalization, window, and threshold settings, while preserving legacy
  six-key checkpoints.
- Added optional Raven-compatible selection-table TSV output from timestamped
  bioacoustic predictions, including independent multilabel annotations.

### Changed

- Expanded deterministic CPU-based unit coverage for datasets, task dispatch,
  feature processing, prediction, metrics, and checkpoint compatibility.

### Fixed

- Fixed `CsvSplit` boundaries so train/validation/test partitions cover every
  input exactly once, including deterministic seed `0` handling and reliable
  CSV reuse.
- Unified deterministic training/prediction feature processing and corrected
  MFCC inference to avoid applying decibel conversion twice.
- Fixed silent binary-label collapse for datasets without a `target` class;
  explicit or deterministic bijective class mappings now validate early.

### Deprecated / Removed

_None yet._

### Migration notes

- Previously saved MFCC models were trained on the `M2MFCC` output
  without a second dB conversion, but the old inference path applied `Amp2Db`
  again. Re-run inference with the original `SpectrogramConfig`,
  `NormalizationConfig`, and matching sample-window length to obtain predictions
  consistent with training. Output scores/decisions can change relative to
  old predictions; re-evaluate any downstream thresholds against validation
  data. This correction does **not** require retraining or change the model's
  32-coefficient MFCC input shape. Linear/mel features and cached power
  spectrogram format are unchanged.
- Two-class datasets without `target`, `positive_class`, or an
  explicit `class_to_index` now map `classes[0]` to 0 and `classes[1]` to 1,
  instead of mapping both labels to 0. Check saved label assumptions and use
  an explicit mapping where binary polarity matters. Datasets with `target`
  retain the legacy convention (`target` = 1).

## 0.1.1 — baseline

Version 0.1.1 provided ResNet encoder/classifier building blocks, audio
transforms, single-label training, metrics, and the six-key `trained_model.pt`
save/load contract preserved by version 0.2.0.
