# SPEC-05 — Independent probabilities, thresholds, and multi-label metrics

**Status:** Planned · **Priority:** P1 · **Dependencies:** SPEC-03; coordinate with SPEC-04 · **Target:** v0.2.0

## Problem / evidence

Current trainer prediction uses `argmax` with exclusive class labels. Its binary AUC path applies `softmax` and its multiclass confusion table assumes one class prediction per example. In multi-label bioacoustics, each clip may have multiple active classes; probability and metric definitions must be independent per class.

## Required implementation

1. Add a pure helper (suggested `metrics/multilabel.py`) `multilabel_probabilities(logits)` returning `torch.sigmoid(logits)` with the same shape `[B,C]` and no premature CPU conversion/detachment; input must be floating logits. Keep raw `Classifier` output unchanged.
2. Add `threshold_multilabel(probabilities, thresholds=0.5)` supporting a scalar threshold or length-C ordered vector, validating each threshold is in `(0,1)` and finite. Return boolean predictions `[B,C]`; **boundary comparison is `>=`**. Do not normalize probabilities to sum to 1. Thresholds use the authoritative class order.
3. Provide streaming multi-label metric accumulators (suggested `MultiLabelMetrics` with `reset`, `update`, and `compute` or named `MetricBase` subclasses): per-class TP/FP/TN/FN, micro precision/recall/F1 and macro precision/recall/F1, with **zero-division result 0**. Define macro as arithmetic mean of per-class scores including unsupported classes; optional support and per-class breakdown should be returned alongside scalar values. Existing single-label metrics stay intact.
4. Update task-aware trainer logging to compute these on **all samples across batches**, never average batch-level F1 values. Support selecting `val_metric` for a scalar multilabel metric through the existing named-metrics mechanism. Do not pass multilabel targets to single-label AUC or confusion plots. Plotting is optional and not a blocker: scalar metrics must work headlessly with no `matplotlib`/`torchvision` requirement.
5. Prediction output must retain probabilities and threshold decisions separately. Expose thresholds via checkpoint metadata when specified (SPEC-07) and via the reusable predictor (SPEC-06); defaults are explicit, not silently inferred from validation examples.

## Unit tests only

File: `tests/test_multilabel_metrics.py`; use tiny hand-computed tensors.

- `sigmoid([0, ln(3), -ln(3)])` returns approximately `[0.5, 0.75, 0.25]` independently.
- At threshold `0.5`, probability **equal** to 0.5 is positive; per-class threshold vector changes expected decisions.
- Shape/count validation rejects NaN/inf thresholds, zero/out-of-range thresholds, and threshold vector length mismatch.
- Hand-calculate TP, FP, FN, TN from 2–3 examples and assert micro and macro precision/recall/F1 exactly or approximately.
- Test all-negative targets/predictions, labels with no support, empty metric reset, and accumulation split across two mini-batches to prevent mean-of-batch-F1 errors.
- Unit-test task dispatcher with mocked metric/plot callbacks to prove multilabel skips single-label AUC/confusion; no actual TensorBoard or file outputs.

## Acceptance criteria

- [ ] Independent class probabilities and inclusive thresholds work for arbitrary C.
- [ ] Micro/macro metrics are computed from accumulated counts, with defined zero-support behavior.
- [ ] Multi-label task mode never reports mutually exclusive confusion/AUC as though it were valid.
- [ ] Unit tests run without optional visualization dependencies.

## Documentation and release notes

**README surgical update after implementation:** State that multilabel inference uses sigmoid and threshold(s), and list available micro/macro scores in one short paragraph.

**Release-notes entry (copy only after completion)** → [`RELEASE_NOTES.md`](../../RELEASE_NOTES.md), **Unreleased → Added**:

> **SPEC-05:** Added independent sigmoid/threshold multi-label predictions with per-class counts and micro/macro precision, recall, and F1 metrics.

Mark Implemented only after the unit tests pass.
