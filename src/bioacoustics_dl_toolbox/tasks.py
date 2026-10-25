"""Canonical tensor shape and dtype checks for classification tasks.

Probability conversion follows the task contract: single-label tasks use
softmax/argmax, while multilabel uses sigmoid/independent thresholds. Full
multilabel postprocessing and metrics are introduced in SPEC-05.
"""

from __future__ import annotations

import torch

from bioacoustics_dl_toolbox.config import TaskConfig, resolve_task_type


def validate_task_batch(
    logits: torch.Tensor,
    targets: torch.Tensor,
    task_config: TaskConfig | None,
    num_classes: int,
) -> None:
    """Require raw ``[B,C]`` logits and targets appropriate for the task.

    Single-label binary/multiclass targets have shape ``[B]`` and an integer
    dtype. Multilabel targets have shape ``[B,C]`` and a floating dtype. This
    validation does not apply any activation to logits.
    """
    task_type = resolve_task_type(task_config, num_classes)
    if logits.ndim != 2 or logits.shape[1] != num_classes:
        raise ValueError(
            f"{task_type} logits must have shape [B, {num_classes}]; "
            f"got {tuple(logits.shape)}"
        )
    if not logits.is_floating_point():
        raise ValueError(f"{task_type} logits must have a floating dtype (raw scores)")
    if task_type == "multilabel":
        if targets.shape != logits.shape:
            raise ValueError(
                f"multilabel targets must have shape [B, {num_classes}] matching "
                f"logits; got {tuple(targets.shape)}"
            )
        if not targets.is_floating_point():
            raise ValueError("multilabel targets must have a floating dtype (multi-hot labels)")
    else:
        if targets.shape != (logits.shape[0],):
            raise ValueError(
                f"{task_type} targets must have shape [B] (integer class indices); "
                f"got {tuple(targets.shape)}"
            )
        if targets.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"{task_type} targets must have an integer dtype")
