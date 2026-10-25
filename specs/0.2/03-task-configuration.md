# SPEC-03 — Explicit binary, multiclass, and multilabel task contract

**Status:** Planned · **Priority:** P1 · **Dependencies:** None · **Target:** v0.2.0

## Problem / evidence

Currently `ClassifierConfig.num_classes` only controls logits count; `SpectrogramDataset` and `Trainer` infer behavior from `num_classes == 2`, which cannot distinguish **two-label multilabel** from **two-class single-label**. A standardized task contract is needed before extending targets, metrics, checkpoints, and prediction.

## Required implementation

1. Add a frozen, serializable `TaskConfig` in `config.py` with `task_type: Literal["auto", "binary", "multiclass", "multilabel"] = "auto"`. Add a central resolver/validator (suggested `resolve_task_type(task_config, num_classes)`), without modifying the ResNet encoder or `Classifier`'s raw-logit output.
2. `auto` preserves legacy behavior: `num_classes == 2` resolves to **binary with 2 logits**; `num_classes > 2` resolves to multiclass. `num_classes == 1` in `auto` should raise an informative error (old single-output semantics were not part of the established public contract). Explicit `multilabel` supports `num_classes >= 1`. Explicit binary requires exactly two classes; explicit multiclass requires at least two.
3. Define shape contracts in one place: single-label targets `[B]` (integer indices), multi-label targets `[B,C]` (float), model logits always `[B,C]`. Binary and multiclass predict with `softmax/argmax`; multilabel predicts with `sigmoid` and independent thresholds. `TaskConfig` does **not** put a sigmoid layer into the model.
4. Add optional `task_config` arguments (or an equivalent shared task descriptor) to dataset and trainer configuration paths, preserving existing calls. The task must remain consistent across train/validation loaders, trainer, predictor, metrics, and checkpoint metadata.
5. Raise early, useful `ValueError` messages for configuration mismatches rather than branching silently on output shape or selecting a default for multilabel. Document that a separate **positive class mapping** for binary is governed by SPEC-08.

## Proposed API sketch (not yet implemented)

```python
TaskConfig(task_type="auto")          # legacy: 2-class => binary, >2 => multiclass
TaskConfig(task_type="multilabel")    # independent outputs, any C >= 1
```

## Unit tests only

File: `tests/test_task_config.py`.

- Parameterize `auto` with 2 and 3 classes; explicit `binary`, `multiclass`, `multilabel` with valid counts.
- Assert `auto` with 1 class and invalid explicit binary counts fail with stable explanatory errors.
- Verify `dataclasses.asdict(TaskConfig(...))` is plain-data serializable.
- Verify raw classifier forward output still has shape `[B,C]` and no internal probability activation (use a simple synthetic tensor and a small classifier head, not a full training pipeline).
- Check resolving task type never changes an existing 2-class classifier into a single-logit binary architecture.

## Acceptance criteria

- [ ] One canonical resolver is reused by task-aware modules.
- [ ] Legacy `auto` output/target shape remains unchanged.
- [ ] Multi-label must be requested explicitly and cannot be inferred from class count.
- [ ] Unit tests cover both valid and invalid task contracts.

## Documentation and release notes

**README surgical update after implementation:** Add a short "Classification tasks" subsection explaining legacy `auto` versus explicit `multilabel`, with a few lines of configuration only after APIs exist.

**Release-notes entry (copy only after completion)** → [`RELEASE_NOTES.md`](../../RELEASE_NOTES.md), **Unreleased → Added**:

> **SPEC-03:** Added an explicit classification task contract for binary, multiclass, and multilabel learning while preserving legacy two-logit binary models.

Mark Implemented only after unit tests and backward-compatibility checks.
