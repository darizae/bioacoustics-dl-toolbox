# 01 — Reusable headless, deterministic spectrogram calculation and rasterization

## Current ownership and actual overlap

BioCall Studio's `src/biocall_studio/review/workflows/spectrograms.py` contains a NumPy STFT, per-channel power mean, exact frequency/time map, peak-relative dB normalization, interpolation and Magma/Viridis RGB rasterization. Its `review/domain/spectrograms.py` provides **application-specific** candidate context, settings and provenance.

The toolbox already has two **different** types of spectrogram utility:
- `audio/transforms.py::Spectrogram` uses **PyTorch** for *model-input* power features and `audio/preprocessing.py` preserves the trained model's feature contract. These must not change or be implicitly reused as the Review renderer.
- `visualization/{image,colormaps,plotting}.py` provides optional Torch/Pillow/Matplotlib/TorchVision visualization, but `visualization/__init__.py` currently imports them eagerly, and its colormap fallback can be nondeterministic without Matplotlib. Do not simply expose the new NumPy path through this eager-import chain or make optional extras mandatory.

## Target public contract

Provide the *smallest* reusable, tested **headless NumPy-only** module within the established toolbox layout (for example `bioacoustics_dl_toolbox.visualization.spectrogram`). If package import requires refactoring `visualization/__init__.py`, preserve old public exports and their optional-extra behavior with explicit lazy import or another proven backwards-compatible approach; imported headless rendering must work without `[visualization]` extras, Qt, Pillow, Matplotlib or TorchVision. Base NumPy is already required.

The API accepts explicit waveform arrays, sample rate and validated rendering parameters, and returns typed/data-only spectral power or raster bytes/array **plus time/frequency coordinate metadata**, not `QImage`, encoded PNG, a database cache, a filesystem output or a Studio-specific request. Preserve the approved Review semantics as explicit options/defaults where scientifically justified:
- symmetric Hann, non-centered framing, `n_fft` pad semantics, short/tail clip behavior and DC handling;
- independent-channel power averaging **before** band peak/reference, including opposite-phase stereo;
- correct frequency crop, per-image peak-relative dB with finite silence floor, contrast and deterministic Magma/Viridis interpolation;
- data-only thumbnail/detail pixel size selection passed by caller, exact mapping to source-frame/time and frequency axes, no decorative annotations or invented loudness units.

Avoid “whole module” transplant, preset/settings/session serialization, candidate IDs, `spectrogram_content_key`, Review cache paths, its generation scheduler, GUI painting/audio, `alp-ui` code or Train/Detect domain types. No new default scientific import side effects. If a reusable boundary cannot reduce real downstream code, document the no-extraction decision rather than growing a framework.

## Ordered implementation

1. Inspect exact Review renderer and current test cases `tests/unit/test_review_spectrograms.py` and `tests/integration/test_review_audio_spectrogram.py` (read-only here), including current parameter defaults, interpolation and cache/version invariants.
2. Define the small public input/output signature and separate numerics from current toolbox ML transforms and optional plot APIs; write deterministic unit tests first, including import with optional visualization extras absent.
3. Implement and verify exact representative array/parity checks: tone, chirp, silence, short boundaries, stereo inversion, frequency-axis edges, per-band reference, multiple sample rates and color determinism. Keep public defaults documented; no model or checkpoint schema change. Use tiny generated fixtures, no 50k/long-running data.
4. Update toolbox public API docs/examples only for **shipped** behavior. If moving code changes Review raster semantics, explicitly specify a version/cache migration for the downstream adapter rather than pretending pixel identity.
5. Execute `python -m pytest`, `python -m ruff check src tests`, `python -m ruff format --check src tests`; report actual commands/results. No app-specific Qt tests in this repo.

**Acceptance:** a public import in a clean wheel without optional visualization extras; deterministic typed pixel + axis outputs; parity (or explicit versioned mismatch explanation) against Review's current reference; unchanged `FeaturePreprocessor`, `Spectrogram`, `trained_model.pt`, and no Studio or Qt dependency. One focused implementation PR, no auto-tag.
