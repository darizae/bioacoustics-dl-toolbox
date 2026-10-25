# SPEC-04 — Multi-hot labels and multi-label model training

**Status:** Planned · **Priority:** P1 · **Dependencies:** SPEC-03 · **Target:** v0.2.0

## Problem / evidence

`SpectrogramDataset` currently accepts `labels: Iterable[str]`, where one class name is assigned per file; `_load_label()` returns one integer. In `training/trainer.py`, both `_train_epoch()` and `_test_epoch()` cast targets to `torch.int64`, compute loss on scalar labels, and use `argmax`. These paths cannot train multi-label outputs with `BCEWithLogitsLoss`, despite the existing classifier's ability to output C logits.

## Required implementation

1. Extend `SpectrogramDataset` with **task-aware** labels without breaking single-label callers. For explicit `TaskConfig(task_type="multilabel")`, accept one iterable of class names **per audio item** (e.g., `[["bird", "rain"], [], ["frog"]]`), aligned with `file_names`; optionally support an equivalent documented sparse label representation. A string must not be mistaken for a sequence of characters. No-label (`[]`) is valid. Existing filename-prefix extraction remains **single-label only** unless caller explicitly supplies multilabel annotations.
2. Validate unknown labels, duplicate class declarations, invalid `class_to_index`, and mismatched item/label counts. Choose one documented policy for repeated labels **within a sample**: deduplicate to a single positive value. Label order comes solely from the mapping; support two or more simultaneous positives.
3. Return `label["call"]` as `torch.float32` tensor `[C]` in multilabel mode, values exactly 0.0/1.0. Keep the old integer class-index shape and `label["file_name"]` for binary/multiclass. Existing model logits `[B,C]` and `ClassifierConfig` are unchanged.
4. Branch trainer train/val/test label casting by resolved task: `float32 [B,C]` for multilabel; existing `int64 [B]` for single-label. Validate model logits and targets are identically shaped `[B,C]` before `BCEWithLogitsLoss`; require `nn.BCEWithLogitsLoss` in multilabel mode or an explicitly documented compatible logits-based criterion. Do not apply sigmoid **before** BCE-with-logits.
5. All trainer metrics, AUC, confusion, and sample image summaries must branch appropriately. For multilabel, **never** use `argmax`, legacy scalar-class names, binary AUC, or multiclass confusion plots. Use multi-label metric support from SPEC-05; if it is absent, disable incompatible summaries with an explanatory warning rather than reporting false numbers. Track validation loss and early stopping normally.
6. Ensure labels and task mode are consistent across train and validation datasets; don't reinterpret class indices when mapping order changes. Keep current `Trainer.fit` parameters/callbacks working for existing callers.

## Unit tests only

Files: `tests/test_multilabel_dataset.py`, `tests/test_multilabel_trainer.py`. Mock spectrogram feature generation or construct minimal fake datasets; use a tiny `nn.Linear`, **one trainer step method call** on a controlled fake loader, not a full model training run.

- Dataset: `[["bird", "rain"], [], ["frog", "frog"]]` with 3 classes yields respectively `[1,0,1]`, `[0,0,0]`, `[0,1,0]` as `float32`.
- Dataset: reject unknown classes, wrong label counts, invalid mapping, string used as multi-label item, duplicate class declarations.
- Trainer: use synthetic logits/targets to verify `BCEWithLogitsLoss` receives floats of shape `[B,C]`, gradients can be computed, and `argmax` is not called in multilabel branch.
- Trainer: validation path handles all-zero targets and reports loss without single-label summary logic; metrics stubs must receive `[B,C]` booleans/tensors as specified in SPEC-05.
- Regression: single-label training calls still consume `int64 [B]` labels with the existing loss contract.

## Acceptance criteria

- [ ] Multi-label data loads in a deterministic class order and supports concurrent positives/empty sets.
- [ ] Train and validation steps use float multi-hot targets and logits-based BCE correctly.
- [ ] Single-label data and trainer remain backward-compatible.
- [ ] No accidental `argmax` or scalar-label image summary path for multilabel.
- [ ] Unit tests only; no full training-run tests.

## Documentation and release notes

**README surgical update after implementation:** Extend "Classification tasks" with a short multilabel annotation example and the required `BCEWithLogitsLoss` / `TaskConfig(task_type="multilabel")` configuration. Do not advertise before the trainer is working.

**Release-notes entry (copy only after completion)** → [`RELEASE_NOTES.md`](../../RELEASE_NOTES.md), **Unreleased → Added**:

> **SPEC-04:** Added multi-hot audio labels and BCE-with-logits training/validation support for simultaneous bioacoustic classes.

Mark Implemented with verified unit tests and commit.
