# Release notes

This is the **ongoing** change log for `bioacoustics-dl-toolbox`. The repository's implementation plan lives in [`0.2_VERSION.md`](0.2_VERSION.md). The current 0.2 documents are **specifications**, not evidence that functionality has shipped.

## Unreleased

### Added

_None yet._

### Changed

_None yet._

### Fixed

_None yet._

### Deprecated / Removed

_None yet._

### Migration notes

_None yet._

### Planned for 0.2.0 — not implemented

The following are proposals, **not released changes**. Each workstream provides a ready-to-use release-note entry; move it into the appropriate Unreleased category only once that feature is implemented and passes its unit tests.

| Spec | Planned work |
| --- | --- |
| [01](specs/0.2/01-csv-splitting.md) | Correct partition boundaries and deterministic CSV splitting |
| [02](specs/0.2/02-preprocessing-parity.md) | Consistent spectrogram/MFCC preprocessing in training and inference |
| [03](specs/0.2/03-task-configuration.md) | Binary, multiclass, and multilabel task contract |
| [04](specs/0.2/04-multilabel-training.md) | Multi-hot dataset labels and BCE-with-logits training |
| [05](specs/0.2/05-multilabel-metrics.md) | Per-label probabilities, thresholds, and micro/macro metrics |
| [06](specs/0.2/06-predictor.md) | Reusable sliding-window prediction API |
| [07](specs/0.2/07-checkpoint-metadata.md) | Versioned optional task and inference metadata with legacy loading |
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
