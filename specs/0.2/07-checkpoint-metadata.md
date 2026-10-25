# SPEC-07 — Versioned exported-model metadata, preserving legacy checkpoints

**Status:** Implemented (unit tests: `tests/test_model_metadata.py`, `tests/test_checkpoints.py`; delivered as a Git patch for review and commit) · **Priority:** P2 · **Dependencies:** SPEC-03; coordinate with SPEC-02 · **Target:** v0.2.0

## Problem / evidence

`training/checkpoints.py::save_model()` writes exactly six keys (`encoderConfig`, `classifierConfig`, `spectrogramConfig`, `encoderState`, `classifierState`, `classes`). There is **no declared task type, normalization mode/config, audio window length, spectral sequence length, or per-class decision threshold**. `load_model(path)` returns the raw dictionary. Reusing a model for prediction cannot be reliably specified by its saved payload alone. The `CheckpointHandler` epoch-resume file is a different mechanism and must not be conflated with exported-model persistence.

## Required implementation

1. **Retain the exact existing six keys** and their meaning. Existing `save_model(...)` calls without new metadata should still write those six keys, and `load_model(path)` should still return a raw dictionary. Leave existing callers and `tests/test_checkpoints.py` working unchanged.
2. Define an optional, versioned, **plain-data** metadata dataclass or validator. Recommended optional seventh key: `"metadata"` with `"schema_version": 2` and `task_type` (explicit, not `auto`, once resolved), `normalization_config`, `window_length_samples`, `sequence_length_frames` if used, `hop_samples` when meaningful, optional threshold(s), and optional ordered class names. All fields must be representable with primitive dictionaries/lists/ints/floats/strings and remain compatible with `torch.load(..., weights_only=True)`.
3. Add an optional keyword-only `metadata=None` argument to `save_model`; when given, validate it against `spectrogramConfig` and `classes`, write the versioned key, and preserve raw state-dict key structure. All C class indices must be unique/contiguous; threshold vectors/maps must match class mapping exactly. `TaskConfig` is serialized by value, not as a Python object.
4. Add an `extract_model_metadata(payload)` / equivalent helper that reads both six-key (v1) and seven-key (v2) artifacts: v1 returns an explicit **incomplete legacy** descriptor, not guessed normalization/sequence length/task/thresholds. A consumer may supply required missing values; fail clearly if prediction cannot be reproduced. Reject unknown future `schema_version` with a migration-oriented message.
5. A versioned v2 save can be recommended for all **newly trained 0.2** models. For existing assets, never silently infer `multilabel` from output count or rewrite existing checkpoints. Model weights and classifier dimensions remain unchanged.
6. For resume checkpoints written by `CheckpointHandler`, add only any optional task consistency information needed to prevent resuming a binary run as multilabel, with backward-compatible reading of old epoch files. Do not reuse the `trained_model.pt` schema for trainer resume state.

## Unit tests only

File: `tests/test_model_metadata.py`, plus retain `tests/test_checkpoints.py`. Use `nn.Linear`, synthetic state dictionaries, and `tmp_path`; no full training/inference.

- Existing six-key round-trip still has **exactly six keys** when metadata omitted, with same class indices and tensor weights.
- V2 seven-key save/load round-trip preserves task type, normalization, window/sample lengths, and per-class thresholds as primitives using weights-only `torch.load`.
- V1 parser returns missing fields as `None`/explicit incomplete flags; does not guess normalization or multilabel status.
- V2 invalid schema versions, missing required keys, invalid mapping/threshold lengths raise targeted errors.
- Assert `save_model` still accepts its preexisting positional argument list unchanged.
- Resume-checkpoint task consistency check can be unit-tested with a plain dict (no background worker or actual training).

## Acceptance criteria

- [x] No breaking change to six-key calls or raw `load_model` behavior.
- [x] Optional versioned metadata round-trips under weights-only loading.
- [x] Legacy missing values are explicit; unsupported future versions rejected.
- [x] Both exported-model and resume-checkpoint contracts are separately documented.
- [x] Unit tests pass; no integration tests.

## Documentation and release notes

**README surgical update after implementation:** Preserve its existing six-key bullet list; add a short note about optional versioned `metadata` and the legacy six-key behavior. State that metadata-free older models still need caller-provided inference settings.

**Release-notes entry (copy only after completion)** → [`RELEASE_NOTES.md`](../../RELEASE_NOTES.md), **Unreleased → Added**:

> **SPEC-07:** Added optional versioned exported-model metadata for reproducible task, normalization, window, and threshold settings, while preserving legacy six-key checkpoints.

If any payload becomes incompatible with an existing reader, add a specific migration note before marking Implemented.
