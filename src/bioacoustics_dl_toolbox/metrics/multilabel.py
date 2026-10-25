"""Independent multi-label probabilities and streaming classification metrics.

Following ANIMAL-SPOT's ``MetricBase`` counter/update/get convention, counts
are accumulated across batches before scores are calculated. Unlike its
single-label metrics, every class is a separate binary decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch

from bioacoustics_dl_toolbox.metrics.core import MetricBase

MultiLabelScore = Literal[
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
]
_SCORE_NAMES = (
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "macro_f1",
)


def multilabel_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Return independent sigmoid probabilities for floating ``[B, C]`` logits.

    The result remains on the input device and retains autograd information.
    No softmax, detachment, or CPU transfer is performed.
    """
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
        or not logits.is_floating_point()
    ):
        raise ValueError("multilabel logits must be a floating tensor of shape [B, C]")
    if logits.shape[1] < 1:
        raise ValueError("multilabel logits require at least one class")
    return torch.sigmoid(logits)


def threshold_multilabel(
    probabilities: torch.Tensor,
    thresholds: float | Sequence[float] | torch.Tensor = 0.5,
) -> torch.Tensor:
    """Return boolean ``[B, C]`` decisions using inclusive ``>=`` thresholds.

    ``thresholds`` is either one value or a length-C vector in class-index
    order. Each value must be finite and strictly between zero and one.
    Probabilities are *not* normalized across classes.
    """
    if (
        not isinstance(probabilities, torch.Tensor)
        or probabilities.ndim != 2
        or not probabilities.is_floating_point()
        or probabilities.shape[1] < 1
    ):
        raise ValueError("multilabel probabilities must be a floating tensor of shape [B, C]")
    if not bool(torch.isfinite(probabilities).all()) or not bool(
        ((probabilities >= 0) & (probabilities <= 1)).all()
    ):
        raise ValueError("multilabel probabilities must be finite and in [0, 1]")

    try:
        threshold_tensor = torch.as_tensor(
            thresholds, dtype=torch.float64, device=probabilities.device
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("thresholds must be a numeric scalar or length-C vector") from exc
    if threshold_tensor.ndim > 1 or (
        threshold_tensor.ndim == 1 and threshold_tensor.numel() != probabilities.shape[1]
    ):
        raise ValueError(
            f"thresholds must be a scalar or a length-{probabilities.shape[1]} vector "
            "in class-index order"
        )
    if not bool(torch.isfinite(threshold_tensor).all()) or not bool(
        ((threshold_tensor > 0) & (threshold_tensor < 1)).all()
    ):
        raise ValueError("each threshold must be finite and strictly between 0 and 1")
    # Compare without rounding a valid threshold to 0 or 1 in low-precision logits.
    return probabilities.to(dtype=torch.float64) >= threshold_tensor


def _divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Zero-denominator-safe division for accumulated nonnegative counts."""
    return numerator / denominator.clamp(min=1)


class MultiLabelMetrics(MetricBase):
    """Per-class confusion counts and corpus-wide micro/macro scores.

    ``update(targets, predictions)`` accepts multi-hot floating/bool targets
    and boolean decisions, both ``[B, C]``. The vectors in ``compute()['per_class']``
    follow the same class-index order as the targets and model output. Unsupported
    classes remain in macro averages with a score of zero. Calling ``get()``
    returns the scalar selected by ``score`` for use with the Trainer's existing
    named-metrics / ``val_metric`` interface.
    """

    supports_multilabel = True

    def __init__(
        self,
        num_classes: int,
        score: MultiLabelScore = "micro_f1",
        device: torch.device | None = None,
    ) -> None:
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, int)
            or num_classes < 1
        ):
            raise ValueError("num_classes must be a positive integer")
        if score not in _SCORE_NAMES:
            raise ValueError(f"score must be one of {_SCORE_NAMES}; got {score!r}")
        self.num_classes = num_classes
        self.score = score
        self.device = device
        self.reset(device)

    def reset(self, device: torch.device | None = None) -> None:
        """Clear the running counts, optionally switching device."""
        if device is not None:
            self.device = device
        for name in ("tp", "fp", "tn", "fn"):
            setattr(
                self, name, torch.zeros(self.num_classes, dtype=torch.int64, device=self.device)
            )

    def update(self, targets: torch.Tensor, predictions: torch.Tensor) -> None:
        """Accumulate counts for a batch without computing per-batch F1."""
        if not isinstance(targets, torch.Tensor) or not isinstance(predictions, torch.Tensor):
            raise ValueError("targets and predictions must be tensors")
        if targets.ndim != 2 or targets.shape[1] != self.num_classes:
            raise ValueError(f"targets must have shape [B, {self.num_classes}]")
        if predictions.shape != targets.shape or predictions.dtype != torch.bool:
            raise ValueError("predictions must be boolean [B, C] matching targets")
        if targets.dtype != torch.bool and not targets.is_floating_point():
            raise ValueError("targets must be boolean or floating multi-hot labels")
        if not bool(((targets == 0) | (targets == 1)).all()):
            raise ValueError("targets must contain only 0 or 1 (multi-hot labels)")

        with torch.no_grad():
            actual = targets.to(device=self.tp.device, dtype=torch.bool)
            predicted = predictions.to(device=self.tp.device)
            self.tp += (actual & predicted).sum(dim=0)
            self.fp += (~actual & predicted).sum(dim=0)
            self.tn += (~actual & ~predicted).sum(dim=0)
            self.fn += (actual & ~predicted).sum(dim=0)

    def _scores(self) -> dict[str, torch.Tensor]:
        tp, fp, fn = (
            counts.to(dtype=torch.float64) for counts in (self.tp, self.fp, self.fn)
        )
        precision = _divide(tp, tp + fp)
        recall = _divide(tp, tp + fn)
        f1 = _divide(2 * tp, 2 * tp + fp + fn)
        total_tp, total_fp, total_fn = tp.sum(), fp.sum(), fn.sum()
        return {
            "micro_precision": _divide(total_tp, total_tp + total_fp),
            "micro_recall": _divide(total_tp, total_tp + total_fn),
            "micro_f1": _divide(2 * total_tp, 2 * total_tp + total_fp + total_fn),
            "macro_precision": precision.mean(),
            "macro_recall": recall.mean(),
            "macro_f1": f1.mean(),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def _get_tensor(self) -> torch.Tensor:
        return self._scores()[self.score]

    def compute(self) -> dict[str, float | dict[str, list[int] | list[float]]]:
        """Report aggregate scores and class-ordered counts, support and scores."""
        scores = self._scores()
        return {
            **{name: scores[name].item() for name in _SCORE_NAMES},
            "per_class": {
                "tp": self.tp.tolist(),
                "fp": self.fp.tolist(),
                "tn": self.tn.tolist(),
                "fn": self.fn.tolist(),
                "support": (self.tp + self.fn).tolist(),
                "precision": scores["precision"].tolist(),
                "recall": scores["recall"].tolist(),
                "f1": scores["f1"].tolist(),
            },
        }
