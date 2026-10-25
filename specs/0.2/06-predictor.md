# SPEC-06 — Reusable sliding-window prediction API

**Status:** Implemented (unit tests: `tests/test_predictor.py`; delivered as a Git patch for review and commit) · **Priority:** P2 · **Dependencies:** SPEC-02, SPEC-03, SPEC-05, SPEC-07 · **Target:** v0.2.0

## Problem / evidence

ANIMAL-SPOT provided a prediction entrypoint; the toolbox only exposes `StridedAudioDataset` (`audio/datasets.py`). Consumers currently must reconstruct a model, iterate windows, apply task-dependent logits interpretation, recover timestamps, and reattach class names themselves. This encourages preprocessing drift and incorrect multilabel `argmax` use.

## Required implementation

1. Add a **library-level, importable**, headless predictor module (suggested `prediction/predictor.py` plus `prediction/__init__.py`) operating on any compatible `nn.Module` that returns `[B,C]` raw logits. No application service, job orchestration, CLI requirement, database, or UI.
2. Define a public `predict_windows(...)` API or a small `Predictor` object. It must accept either a compatible saved-model bundle **with resolved inference metadata**, or an explicit model plus `SpectrogramConfig`, `NormalizationConfig`, task type, class-to-index mapping, window length, hop, thresholds, and device. For old six-key checkpoints, require caller-supplied missing normalization/window/task settings; don't invent reproducibility-critical values.
3. Reuse `StridedAudioDataset` plus the shared deterministic pipeline from SPEC-02. Support mono audio, sampling/resampling at the configured rate, fixed shape per window (including short final windows), configurable batch size, CPU by default and an optional torch device. Define the sliding-window count explicitly: with `0 < hop_samples <= window_samples`, use starts `0, hop, 2*hop, ...` **through the first start whose window reaches the final sample** (e.g., 11 samples, window 4, hop 3 → starts 0, 3, 6, 9). The present `StridedAudioDataset.__len__` formula can omit the final tail; correct it while preserving its public API. Reject `hop <= 0`, `hop > window_samples` (gapped scans), `window_samples <= 0`, empty/invalid mappings, or mismatched output dimensions **before** emitting results.
4. Run `model.eval()` and `torch.inference_mode()`; do not train, mutate weights, or calculate gradients. Preserve caller device expectations where possible; document whether previous training/eval mode is restored. Avoid mandatory visualization or `tensorboardX` imports.
5. Return structured results with at least `start_seconds`, `end_seconds`, `probabilities: dict[str,float]`, `predicted_labels: list[str]` (or equivalent explicit records with stable class order). Use `start_sample = index * hop`; `end_sample = min(start_sample + window_samples, n_frames)` and convert using the **resampled waveform's sample rate**. Clamp the last window to recording length. Include raw window index and optionally start/end sample indices for stable downstream export.
6. Binary/multiclass: `softmax` probabilities and exactly one selected label from `argmax`. Multilabel: independent sigmoid and validated scalar/per-class thresholds (SPEC-05), including zero or multiple selected labels. Probabilities must always be associated with the exact class index they came from. No implicit cross-window event merging: that belongs in an optional annotation adapter (SPEC-10).
7. Make error handling explicit for unreadable/empty audio, missing required legacy checkpoint metadata, non-finite probabilities, and classifier/map size mismatch. Avoid blanket exception swallowing.

## Proposed API sketch (illustrative only)

```python
results = predict_windows(
    model,
    "recording.wav",
    class_to_index={"bird": 0, "frog": 1, "rain": 2},
    task_config=TaskConfig(task_type="multilabel"),
    spec_config=spec_config,
    norm_config=norm_config,
    window_samples=44100,
    hop_samples=22050,
    thresholds={"bird": 0.55, "frog": 0.5, "rain": 0.7},
)
```

## Unit tests only

File: `tests/test_predictor.py`; use a **fake dataset or monkeypatched audio loader** and a deterministic fake model with pre-programmed logits. Do not run a real model on an audio recording.

- Assert `model.eval()` and inference/no-grad behavior using a spy model; check no parameter changes.
- Verify softmax/argmax in binary/multiclass and independent sigmoid/threshold in multilabel; zero/multiple predicted labels and stable name/index association.
- Verify timestamp math for exact, partial, and shorter-than-window recordings via fake `n_frames` and sample rate; for 11/4/3, assert starts `[0,3,6,9]`, and last end time equals—not exceeds—recording duration.
- Verify final fixed tensor shape, deterministic window ordering, correct batch output slicing and device handling via mocks.
- Validate invalid hops/window sizes/mappings/thresholds and mismatch between `C` logits and class mapping.
- Verify loading a legacy metadata-free bundle requires missing settings explicitly; do not assert that defaults are magically known.

## Acceptance criteria

- [x] Headless library predictor with documented public import path.
- [x] Deterministic feature path, window geometry, ordering, and named probabilities.
- [x] Correct exclusive versus multilabel decisions with no weight updates.
- [x] Legacy checkpoint users get an actionable missing-metadata error.
- [x] Only **unit tests** added; no real-file end-to-end predictions.

## Documentation and release notes

**README surgical update after implementation:** Add a brief "Prediction" subsection with the final public signature and one minimal example, plus the requirement to supply missing metadata for legacy checkpoints. Do not add a full tutorial or imply a CLI exists.

**Release-notes entry (copy only after completion)** → [`RELEASE_NOTES.md`](../../RELEASE_NOTES.md), **Unreleased → Added**:

> **SPEC-06:** Added a reusable, headless sliding-window predictor with class-named probabilities, task-aware predictions, and recording-relative timestamps.

Mark Implemented with unit tests and commit after verification.
