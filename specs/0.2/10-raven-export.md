# SPEC-10 — Optional Raven-compatible annotation export

**Status:** Implemented (unit tests: `tests/test_raven_export.py`; delivered as a Git patch for review and commit) · **Priority:** P2 · **Dependencies:** SPEC-06 · **Target:** v0.2.0

## Problem / evidence

The original ANIMAL-SPOT project offers an evaluation-to-Raven workflow; the toolbox has no structured annotation export. The new predictor (SPEC-06) will return window-relative timestamps and named class probabilities; a small optional **adapter** can convert these to a Raven selection table without importing application code.

## Required implementation

1. Add a narrow `prediction/raven.py` or `io/raven.py` adapter that consumes the **already computed in-memory** `WindowPrediction` records (SPEC-06). It must not load audio, build a model, or call the prediction pipeline.
2. Provide a pure `predictions_to_raven_rows(predictions, sample_rate, ...)` function and a TSV writer that accepts a file-like text stream or a path. Use Python's `csv` module with `delimiter="\t"`; no pandas/plotting dependency.
3. Use a documented Raven-compatible minimal selection table with these exact columns in order: `Selection`, `View`, `Channel`, `Begin Time (s)`, `End Time (s)`, `Low Freq (Hz)`, `High Freq (Hz)`, `Annotation`. `Annotation` contains the predicted class name, one row per **active** label and selected window (unless optional merge is enabled). Output header even when there are no detections.
4. Use 1-based selection numbering, stable sort by start time and mapping order, `View="Spectrogram 1"`, `Channel=1` by default, and source seconds clamped/validated to recording bounds. Default frequency bounds are the full sampled band `0..sample_rate/2` (not model feature cutoffs); allow optional explicit positive frequency bounds with `0 <= low < high <= Nyquist`.
5. Optionally merge overlapping/touching **same-label** windows into one event using pure interval logic; never merge distinct class labels or silently collapse concurrent detections. Merging is off by default; explain that window labels are not automatically behavioral events. No automatic threshold learning.
6. Reject malformed `start/end`, NaN/negative times, invalid channel/frequency bounds, or unordered/duplicate classes as appropriate. Preserve UTF-8 class names and deterministic numeric formatting. If a consumer wants ANIMAL-SPOT's exact historical CSV layout, that is a separate compatibility request, not an unverified guarantee from this spec.

## Unit tests only

File: `tests/test_raven_export.py`. Construct plain prediction records with synthetic timestamps/labels; use `io.StringIO` as writer. Do not run inference or read a recording.

- Empty input writes exactly the header and no selections.
- One window with two active labels produces two separate stable 1-based rows with correct times and names.
- Two overlapping windows of the **same** label merge only when explicitly requested; different labels never merge.
- Invalid start/end, NaN, out-of-range frequency bounds, and bad sampling rates raise useful errors.
- Tabs, UTF-8 labels, numeric precision, and column ordering are deterministic in text output.
- File writer delegates to pure row generator; test with in-memory stream, not a full application export.

## Acceptance criteria

- [x] Pure, headless Raven TSV generation from completed predictor result records.
- [x] Documented selection-table columns, class labels, bounds, and ordering.
- [x] Optional same-label merging does not hide simultaneous labels.
- [x] All added coverage is **unit-level** and in-memory.

## Documentation and release notes

**README surgical update after implementation:** Under the new "Prediction" subsection, add one link or short sentence explaining that prediction records can be exported as Raven selection-table TSV. No proprietary file formats or GUI workflow promises.

**Release-notes entry (copy only after completion)** → [`RELEASE_NOTES.md`](../../RELEASE_NOTES.md), **Unreleased → Added**:

> **SPEC-10:** Added optional Raven-compatible selection-table TSV output from timestamped bioacoustic predictions, including independent multi-label annotations.

Implementation verified with in-memory unit tests; commit hash pending application of this patch.
