# SPEC-02 — Guarantee feature parity between training and prediction

**Status:** Implemented in supplied patch (commit pending after application; tests: `tests/test_preprocessing.py`) · **Priority:** P0 · **Dependencies:** None · **Target:** v0.2.0

## Problem / evidence

`SpectrogramDataset.__getitem__()` (`audio/datasets.py`) applies `M2MFCC(n_mfcc=32)` **without** an additional `Amp2Db`. `StridedAudioDataset.__init__()` does `F2M → M2MFCC → Amp2Db`. Log scaling MFCC coefficients a second time is inconsistent and can alter signs/magnitudes. The two classes independently assemble closely related pipelines. Short inference windows may also differ from training's `PaddedSubsequenceSampler` behavior. A different feature tensor at inference invalidates trained classifier assumptions.

## Required implementation

1. Introduce a **single shared deterministic feature-preparation builder** in `audio/` (suggested `audio/preprocessing.py`), taking `SpectrogramConfig` and `NormalizationConfig`. Both training and inference datasets must use it for common transformations. Keep the current public constructors functional.
2. Canonical order: audio load/resample/mono → pre-emphasis → power spectrogram (`center=False`) → frequency compression (`linear`, `mel`, or `mfcc`) → `Amp2Db` **only for linear/mel** → normalization. MFCC path: `F2M → M2MFCC(n_mfcc=32)` → normalization, **no second `Amp2Db`**. Preserve existing training MFCC semantics and default representation dimensions; do not silently change a pre-trained model's input shape.
3. Training-only augmentation retains its existing documented sequencing, but feature extraction after augmentation must use exactly the inference representation/normalization functions. Disable stochastic transforms in validation and prediction.
4. Use a consistent, explicit window-padding convention. For equal-length audio windows and disabled augmentation, both datasets must yield numerically equivalent features with equal shapes (subject to deterministic crop alignment). Decide and document how shorter-than-FFT and shorter-than-sequence-length input is padded. Do not force an arbitrary MFCC dB conversion to eliminate NaNs; handle numerical stability in the MFCC transform itself if necessary.
5. Validate `freq_compression`, nonzero FFT/hop/sample-rate and frequency ranges at the shared pipeline boundary. Reuse existing dataclasses; do not require `torchaudio` or any new mandatory dependency.
6. Preserve caching behavior: if feature cache format/meaning changes, bump cache version or metadata fingerprint to prevent silent mixing of old and new features. Avoid unnecessary cache invalidation for unchanged linear/mel paths.

## Unit tests only

File: `tests/test_preprocessing.py`. Use synthetic fixed waveforms; monkeypatch file reading in datasets or invoke the pure transform pipeline directly. No real recordings or model training.

- Parameterize `linear`, `mel`, `mfcc`; compare **same waveform and config** through shared train/inference deterministic transform pipelines (`torch.testing.assert_close`).
- For MFCC, instrument a fake `Amp2Db` to assert it is **not** called after `M2MFCC`; verify signed MFCC values remain finite.
- Test shapes/dtype and padding on a short waveform and a full-size waveform; inspect expected `[channel, time, frequency]` feature convention in source before fixing assertions.
- Assert two sequential deterministic runs are numerically identical and no random augmentation runs during inference.
- Verify invalid FFT/hop/compression configurations raise descriptive errors.
- If cache metadata changes, test only pure version/fingerprint logic; do not build a cache+dataset multi-component test.

## Acceptance criteria

- [x] A shared deterministic transform is used by both dataset paths.
- [x] No MFCC double-dB step; linear/mel paths retain their dB operation.
- [x] Short-window feature geometry is explicitly defined and tested.
- [x] No unexpected dependency or model architecture change.
- [x] Listed **unit tests** pass; no integration tests introduced.

## Documentation and release notes

**README surgical update after implementation:** Add a brief statement under a future "Prediction" subsection that training/inference preprocessing shares the same configuration, including normalization and sample-window length. Do not represent MFCC parity as fixed before code lands.

**Release-notes entry (copy only after completion)** → [`RELEASE_NOTES.md`](../../RELEASE_NOTES.md), **Unreleased → Fixed**:

> **SPEC-02:** Unified deterministic training/prediction feature processing and corrected MFCC inference to avoid applying decibel conversion twice.

If the correction changes results for previously saved models, document that inference behavior change and mitigation under Migration notes, then mark this spec Implemented with tests/commit.

**Verification (patch working copy, 2026-10-25):** `tests/test_preprocessing.py`
passes, as does the full pytest suite (121 tests). This sandbox could not
install `resampy`, so the test run used an external import-only shim for the
missing existing dependency; no shim is included in the patch. `ruff` was not
available for verification. Record the commit hash here after the patch is
committed in the target repository.
