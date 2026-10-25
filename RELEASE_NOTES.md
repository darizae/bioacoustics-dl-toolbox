# Release notes

This is the **ongoing** change log for `bioacoustics-dl-toolbox`. The repository's implementation plan lives in [`0.2_VERSION.md`](0.2_VERSION.md). Most 0.2 workstreams are **specifications**, not evidence that functionality has shipped; completed work is tracked under Unreleased and has not yet been released.

## Unreleased

### Added

- **Alpaca Desktop Spec 04.5:** Added bounded, no-clobber PCM WAV segment
  extraction with exact frame/format facts and cooperative cancellation.
- **SPEC-03:** Added an explicit classification task contract for binary, multiclass, and multilabel learning while preserving legacy two-logit binary models.
- **SPEC-04:** Added multi-hot audio labels and BCE-with-logits training/validation support for simultaneous bioacoustic classes.
- **SPEC-05:** Added independent sigmoid/threshold multi-label predictions with per-class counts and micro/macro precision, recall, and F1 metrics.
- **SPEC-06:** Added a reusable, headless sliding-window predictor with class-named probabilities, task-aware predictions, and recording-relative timestamps.
- **SPEC-07:** Added optional versioned exported-model metadata for reproducible task, normalization, window, and threshold settings, while preserving legacy six-key checkpoints.

### Changed

_None yet._

### Fixed

- **SPEC-01:** Fixed `CsvSplit` boundaries so train/validation/test partitions cover every input exactly once, including deterministic handling of seed `0` and reliable CSV reuse.
- **SPEC-02:** Unified deterministic training/prediction feature processing and corrected MFCC inference to avoid applying decibel conversion twice.

### Deprecated / Removed

_None yet._

### Migration notes

- **SPEC-02:** Previously saved MFCC models were trained on the `M2MFCC` output
  without a second dB conversion, but the old inference path applied `Amp2Db`
  again. Re-run inference with the original `SpectrogramConfig`,
  `NormalizationConfig`, and matching sample-window length to obtain predictions
  consistent with training. Output scores/decisions can change relative to
  old predictions; re-evaluate any downstream thresholds against validation
  data. This correction does **not** require retraining or change the model's
  32-coefficient MFCC input shape. Linear/mel features and cached power
  spectrogram format are unchanged.

### Planned for 0.2.0 — not implemented

The following are proposals, **not released changes**. Each workstream provides a ready-to-use release-note entry; move it into the appropriate Unreleased category only once that feature is implemented and passes its unit tests.

| Spec | Planned work |
| --- | --- |
| [08](specs/0.2/08-binary-class-mapping.md) | Strict, unambiguous binary label mapping |
| [09](specs/0.2/09-unit-test-policy.md) | Broader focused **unit-test-only** coverage |
| [10](specs/0.2/10-raven-export.md) | Optional Raven-compatible selection-table export |

## 0.1.1 — baseline in supplied source archive

The snapshot used to author these specs declares version `0.1.1` in `pyproject.toml`. This entry records the **observed baseline only**, not a reconstructed historical release history. Existing capabilities include ResNet encoder/classifier building blocks, audio transforms, single-label training, metrics, and a six-key `trained_model.pt` save/load contract. See the [README](README.md) for the implemented API.

---

## Maintainer process

1. On completing a spec, put its prepared sentence into **Unreleased → Added / Changed / Fixed**, cite its ID, and add migration detail if existing behavior changes. Remove that item from the Planned table or mark it completed without calling it a release.
2. On release, transfer verified entries into `## 0.2.0 — YYYY-MM-DD`, leave a fresh `## Unreleased`, and update the version in `pyproject.toml`.
3. Keep entries concise, externally meaningful, and factual. Do not claim tests passed unless they ran; do not label a proposal as implemented.
4. Preserve this file and its README link as the ongoing release-notes location for subsequent versions.
