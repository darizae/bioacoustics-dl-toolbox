# SPEC-09 — Unit-test-only regression coverage for v0.2.0

**Status:** Planned · **Priority:** P2 (continuous) · **Dependencies:** Applies to all other specs · **Target:** v0.2.0

## Problem / evidence

The supplied repository has one test module (`tests/test_checkpoints.py`) containing four tests: six-key round-trip, optional import boundaries, trainer import, and sampler picklability. The planned fixes affect splitting, features, label representation, trainer task dispatch, metrics, prediction, and serialization; they need focused reproducible checks. **The testing scope is intentionally unit tests only.**

## Required implementation

1. Organize tests by module under `tests/`: `test_csv_split.py`, `test_preprocessing.py`, `test_task_config.py`, `test_multilabel_dataset.py`, `test_multilabel_trainer.py`, `test_multilabel_metrics.py`, `test_predictor.py`, `test_model_metadata.py`, `test_class_mapping.py`, `test_raven_export.py`. A workstream may consolidate closely related tests without changing its coverage requirements.
2. Each test must focus on **one function or class behavior** with fake/minimal dependencies. Use synthetic NumPy/Torch tensors, fake models, monkeypatched audio loaders, spies for writer hooks, `tmp_path` for isolated serialization, and controlled random seeds. No real data download, no external web/network, and no GPU requirement.
3. Keep assertions precise: output shape, dtype, values, exact class-index order, exception type/message, timestamp bounds, stable serialization keys, and absence of inappropriate `argmax`/visualization calls in multilabel paths. Prefer small hand-computed truth tables to opaque model-quality scores.
4. Separate pure unit test concerns from workflow tests: calling a **single trainer step** with an in-memory fake loader is allowed when testing target type or branch behavior; training multiple epochs or running a complete dataset→trainer→checkpoint→predictor workflow is **not**. A temporary model checkpoint round-trip unit test is allowed for the serialization function itself, not a model training pipeline.
5. Preserve the current four tests, including import-boundary assertions that RF and visualization extras remain optional. Ensure tests can run on CPU and do not require `torchaudio`, plotting libraries, application modules, or a real dataset.
6. New public behavior should have at least one successful unit case, a boundary case, and a failure case. Regression tests for fixed bugs must fail against old behavior and pass once corrected. All test names should explain the contract and avoid sleeps/timeouts.
7. CI configuration is **not** required by this spec; unit-test workflows can be run locally. Do not add integration or end-to-end test stages, dependencies, or harnesses.

## Unit tests only

Use this checklist as the unit-level coverage contract. No cross-module or end-to-end harness should be created.

## Coverage checklist (each belongs to a unit test)

| Workstream | Required focused evidence |
| --- | --- |
| 01 | Every item accounted for, seed 0, CSV relative paths |
| 02 | Linear/mel/MFCC parity, finite features, short padding |
| 03 | Task resolution, error cases, two-logit binary compatibility |
| 04 | Multi-hot labels, BCE float targets, no multilabel `argmax` |
| 05 | Independent sigmoid, thresholds, micro/macro counting |
| 06 | Mocked window timestamps, probabilities, device, metadata errors |
| 07 | Legacy six-key and versioned metadata round-trip |
| 08 | Binary `target` legacy case and non-target bijection |
| 10 | Pure Raven row/schema serialization and optional event merging |

## Commands / verification

```bash
python -m pip install -e '.[test]'
python -m pytest
python -m ruff check src tests
python -m ruff format --check src tests
```

These are the intended verification commands **after code implementation**. Do not report that v0.2 changes are passing when only specifications have been added.

## Acceptance criteria

- [ ] Every functional spec has a relevant, isolated unit-test module and its pass/failure/boundary cases.
- [ ] No real recording, GPU, network, integration, or full pipeline tests were introduced.
- [ ] Existing four baseline tests still pass.
- [ ] Test execution and lint results are recorded at code-review time.

## Documentation and release notes

**README surgical update after implementation:** The existing Development commands remain authoritative; add only one sentence stating the unit-test-only policy for v0.2.0. No test matrix in README.

**Release-notes entry (copy only after completion)** → [`RELEASE_NOTES.md`](../../RELEASE_NOTES.md), **Unreleased → Changed**:

> **SPEC-09:** Expanded deterministic CPU-based unit coverage for datasets, task dispatch, feature processing, prediction, metrics, and checkpoint compatibility.

This note is justified only when the suite actually exists and has been executed. Mark Implemented with test references and commit.
