"""
Checkpoint handling and model persistence.

Ported from ANIMAL-SPOT ``utils/checkpoints.py`` (Bergler & Schroeter, GPL-3.0).
Adds typed ``save_model`` / ``load_model`` helpers that serialize config
dataclasses alongside state dicts.
"""

from __future__ import annotations

import glob
import logging
import os
import queue
import threading
from collections.abc import Mapping
from dataclasses import asdict, fields
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from bioacoustics_dl_toolbox.config import (
    ClassifierConfig,
    EncoderConfig,
    NormalizationConfig,
    ResolvedTaskType,
    SpectrogramConfig,
    TaskConfig,
    resolve_task_type,
)


class CheckpointHandler:
    """Async checkpoint writer with automatic rotation.

    Parameters
    ----------
    checkpoint_dir:
        Directory to write checkpoints into.
    prefix:
        Filename prefix for checkpoint files.
    max_checkpoints:
        Maximum number of checkpoints to keep on disk.
    logger:
        Optional logger instance.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        prefix: str = "",
        max_checkpoints: int = 5,
        logger: logging.Logger | None = None,
    ) -> None:
        self.max_checkpoints = max_checkpoints
        self.checkpoint_dir = checkpoint_dir
        if self.checkpoint_dir is None:
            return
        if not prefix.endswith("_"):
            prefix += "_"
        self.prefix = prefix
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self._logger = logger
        if self._logger is not None:
            self._logger.debug("Starting checkpoint writer thread")
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._should_stop = threading.Event()
        self._worker = threading.Thread(target=self._write_worker, daemon=True)
        self._worker.start()

    def _write_worker(self) -> None:
        while not self._should_stop.is_set():
            try:
                checkpoint_dict = self._queue.get()
                if checkpoint_dict is None:
                    break
                checkpoint_name = os.path.join(
                    self.checkpoint_dir,
                    "{}epoch_{:05d}.checkpoint".format(
                        self.prefix, checkpoint_dict["trainState"]["epoch"]
                    ),
                )
                if self._logger is not None:
                    self._logger.debug("Writing checkpoint to {}".format(checkpoint_name))
                torch.save(checkpoint_dict, checkpoint_name)
                checkpoints = glob.glob(
                    os.path.join(self.checkpoint_dir, self.prefix + "*.checkpoint")
                )
                if len(checkpoints) > self.max_checkpoints:
                    checkpoints.sort()
                    for i in range(len(checkpoints) - self.max_checkpoints):
                        os.remove(checkpoints[i])
            except Exception as e:
                if self._logger is not None:
                    self._logger.error("Failed to write checkpoint: {}".format(str(e)))
            finally:
                self._queue.task_done()
        if self._logger is not None:
            self._logger.info("Shutting down checkpoint writer thread")

    def write(self, checkpoint_dict: dict[str, Any]) -> None:
        """Enqueue a checkpoint for async writing."""
        if self.checkpoint_dir is None:
            return
        self._queue.put(checkpoint_dict)

    def read_latest(self) -> dict[str, Any] | None:
        """Read the most recent checkpoint, or ``None`` if none exists."""
        if self.checkpoint_dir is None:
            return None
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, self.prefix + "*.checkpoint"))
        if len(checkpoints) == 0:
            if self._logger is not None:
                self._logger.info("No checkpoints found in {}".format(self.checkpoint_dir))
            return None
        checkpoints.sort()
        if self._logger is not None:
            self._logger.info("Restoring checkpoint {}".format(checkpoints[-1]))
        return torch.load(  # type: ignore[no-any-return]
            checkpoints[-1], map_location="cpu", weights_only=True
        )

    def _shutdown_worker(self) -> None:
        self._should_stop.set()
        if self._worker.is_alive():
            self._queue.put(None)
            self._worker.join()

    def __del__(self) -> None:
        if self._logger is not None:
            self._logger.info("Shutting down CheckpointHandler")
        self._shutdown_worker()


# ANIMAL-SPOT's rotating epoch checkpoints and exported trained-model bundles
# have different contracts. Only exported bundles use this metadata schema.
_MODEL_METADATA_VERSION = 2
_REQUIRED_METADATA_FIELDS = frozenset(
    {"schema_version", "task_type", "normalization_config", "window_length_samples"}
)
_OPTIONAL_METADATA_FIELDS = frozenset(
    {"sequence_length_frames", "hop_samples", "thresholds", "class_names"}
)


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"metadata {name} must be a positive integer")
    return value


def _ordered_class_names(classes: object) -> list[str]:
    if not isinstance(classes, Mapping) or not classes:
        raise ValueError("classes must be a nonempty class-name to index mapping")
    if any(not isinstance(name, str) or not name for name in classes):
        raise ValueError("classes must have nonempty string names")
    indices = list(classes.values())
    if any(not isinstance(idx, int) or isinstance(idx, bool) for idx in indices) or set(
        indices
    ) != set(range(len(indices))):
        raise ValueError("classes indices must be unique, contiguous integers 0..C-1")
    return sorted(classes, key=classes.__getitem__)


def _normalization_data(value: object) -> dict[str, str | float]:
    if isinstance(value, NormalizationConfig):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise ValueError("metadata normalization_config must be a mapping or NormalizationConfig")
    required = {field.name for field in fields(NormalizationConfig)}
    missing = required - value.keys()
    if missing:
        raise ValueError(f"metadata normalization_config is missing: {', '.join(sorted(missing))}")
    unknown = value.keys() - required
    if unknown:
        raise ValueError(
            f"metadata normalization_config has unknown fields: {', '.join(sorted(unknown))}"
        )
    if value["mode"] not in ("db", "min_max"):
        raise ValueError("metadata normalization_config mode must be 'db' or 'min_max'")
    for name in ("min_level_db", "ref_level_db"):
        number = value[name]
        if not isinstance(number, Real) or isinstance(number, bool) or not isfinite(number):
            raise ValueError(f"metadata normalization_config {name} must be finite and numeric")
    if value["mode"] == "db" and value["min_level_db"] >= 0:
        raise ValueError("metadata normalization_config min_level_db must be negative in db mode")
    return {
        "mode": value["mode"],
        "min_level_db": float(value["min_level_db"]),
        "ref_level_db": float(value["ref_level_db"]),
    }


def _probability(value: object) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value):
        raise ValueError("metadata thresholds must be finite numeric probabilities")
    if not 0 < value < 1:
        raise ValueError("metadata thresholds must be strictly between 0 and 1")
    return float(value)


def _threshold_data(
    value: object, names: list[str], task_type: str
) -> float | list[float] | dict[str, float]:
    if task_type != "multilabel":
        raise ValueError("metadata thresholds are only supported for multilabel tasks")
    if isinstance(value, Mapping):
        if set(value) != set(names):
            raise ValueError("metadata thresholds mapping must match classes exactly")
        return {name: _probability(value[name]) for name in names}
    if isinstance(value, (list, tuple)):
        if len(value) != len(names):
            raise ValueError(f"metadata thresholds must contain exactly {len(names)} class values")
        return [_probability(item) for item in value]
    return _probability(value)


def _spectrogram_from_payload(payload: Mapping[str, Any]) -> SpectrogramConfig:
    data = payload.get("spectrogramConfig")
    if not isinstance(data, Mapping):
        raise ValueError("checkpoint spectrogramConfig must be a configuration mapping")
    required = {field.name for field in fields(SpectrogramConfig)}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"checkpoint spectrogramConfig is missing: {', '.join(sorted(missing))}")
    try:
        return SpectrogramConfig(**data)
    except TypeError as exc:
        raise ValueError(f"Invalid checkpoint spectrogramConfig: {exc}") from exc


def _check_schema_version(metadata: object) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint metadata must be a mapping")
    version = metadata.get("schema_version")
    if type(version) is not int or version != _MODEL_METADATA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint metadata schema_version {version!r}; "
            "upgrade the library or migrate the checkpoint to supported schema_version 2"
        )


def _validated_metadata(
    metadata: Mapping[str, Any],
    spec_config: SpectrogramConfig,
    classes: Mapping[str, int],
    classifier_classes: int | None = None,
) -> dict[str, Any]:
    """Validate v2 fields and return only primitives (safe for weights_only=True)."""
    _check_schema_version(metadata)
    missing = _REQUIRED_METADATA_FIELDS - metadata.keys()
    if missing:
        raise ValueError(f"checkpoint metadata is missing: {', '.join(sorted(missing))}")
    unknown = metadata.keys() - _REQUIRED_METADATA_FIELDS - _OPTIONAL_METADATA_FIELDS
    if unknown:
        raise ValueError(f"checkpoint metadata has unknown fields: {', '.join(sorted(unknown))}")

    names = _ordered_class_names(classes)
    if classifier_classes is not None and classifier_classes != len(names):
        raise ValueError("classifierConfig num_classes must match the classes mapping")
    task_value = metadata["task_type"]
    if isinstance(task_value, TaskConfig):
        task_value = task_value.task_type
    if not isinstance(task_value, str) or task_value == "auto":
        raise ValueError("checkpoint metadata task_type must be resolved, not 'auto'")
    # Validate that binary remains a two-logit task and multilabel is explicit.
    task_type = resolve_task_type(TaskConfig(task_type=task_value), len(names))
    norm = _normalization_data(metadata["normalization_config"])
    window = _positive_int(metadata["window_length_samples"], "window_length_samples")
    _positive_int(spec_config.n_fft, "spectrogramConfig.n_fft")
    _positive_int(spec_config.hop_length, "spectrogramConfig.hop_length")
    _positive_int(spec_config.sample_rate, "spectrogramConfig.sample_rate")
    validated: dict[str, Any] = {
        "schema_version": _MODEL_METADATA_VERSION,
        "task_type": task_type,
        "normalization_config": norm,
        "window_length_samples": window,
    }
    if "sequence_length_frames" in metadata:
        frames = _positive_int(metadata["sequence_length_frames"], "sequence_length_frames")
        # center=False, with the same minimum one-frame padding as the shared
        # spectrogram_frames_for_samples preprocessing helper.
        expected = (
            1 + (max(window, spec_config.n_fft) - spec_config.n_fft) // spec_config.hop_length
        )
        if frames != expected:
            raise ValueError(
                f"metadata sequence_length_frames={frames} does not match "
                f"spectrogramConfig/window_length_samples (expected {expected})"
            )
        validated["sequence_length_frames"] = frames
    if "hop_samples" in metadata:
        hop = _positive_int(metadata["hop_samples"], "hop_samples")
        if hop > window:
            raise ValueError("metadata hop_samples cannot exceed window_length_samples")
        validated["hop_samples"] = hop
    if "thresholds" in metadata:
        validated["thresholds"] = _threshold_data(metadata["thresholds"], names, task_type)
    if "class_names" in metadata:
        if (
            not isinstance(metadata["class_names"], (list, tuple))
            or list(metadata["class_names"]) != names
        ):
            raise ValueError("metadata class_names must match classes in ascending index order")
        validated["class_names"] = list(names)
    return validated


def extract_model_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Describe legacy or v2 exported model metadata without guessing settings.

    A v1 six-key bundle returns an incomplete descriptor: unknown task,
    normalization, window length, hop, frames, and thresholds remain ``None``.
    A v2 bundle is validated before values are returned. ``is_complete`` means
    the bundle specifies all settings required for windowed prediction (the
    optional sequence length and thresholds do not affect that assessment).
    The original ``load_model`` continues returning the *raw* saved payload.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    if "metadata" not in payload:
        return {
            "schema_version": 1,
            "legacy": True,
            "is_complete": False,
            "task_type": None,
            "normalization_config": None,
            "window_length_samples": None,
            "sequence_length_frames": None,
            "hop_samples": None,
            "thresholds": None,
            "class_names": None,
        }
    _check_schema_version(payload["metadata"])
    spec_config = _spectrogram_from_payload(payload)
    classifier_config = payload.get("classifierConfig")
    classifier_classes = (
        classifier_config.get("num_classes") if isinstance(classifier_config, Mapping) else None
    )
    validated = _validated_metadata(
        payload["metadata"], spec_config, payload.get("classes"), classifier_classes
    )
    return {
        "schema_version": 2,
        "legacy": False,
        "is_complete": "hop_samples" in validated,
        "task_type": validated["task_type"],
        "normalization_config": validated["normalization_config"],
        "window_length_samples": validated["window_length_samples"],
        "sequence_length_frames": validated.get("sequence_length_frames"),
        "hop_samples": validated.get("hop_samples"),
        "thresholds": validated.get("thresholds"),
        "class_names": validated.get("class_names"),
    }


def validate_resume_checkpoint_task(
    payload: Mapping[str, Any], task_type: ResolvedTaskType
) -> None:
    """Reject cross-task resumes; accept ANIMAL-SPOT-style untagged epochs.

    Epoch-resume files contain ``modelState``/``trainState``/``classes`` and
    optionally ``taskType``. They do *not* carry exported-model metadata.
    """
    if task_type not in ("binary", "multiclass", "multilabel"):
        raise ValueError("Resume task_type must be binary, multiclass, or multilabel")
    saved = payload.get("taskType")
    if saved is not None and saved != task_type:
        raise ValueError(
            f"Resume checkpoint task '{saved}' does not match trainer task '{task_type}'"
        )


def save_model(
    model: nn.Module,
    encoder: nn.Module,
    encoder_config: EncoderConfig,
    classifier: nn.Module,
    classifier_config: ClassifierConfig,
    spec_config: SpectrogramConfig,
    path: str | Path,
    class_dist_dict: dict[str, int],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save a trained model with all configs needed to reload it.

    The saved dict contains:
    - ``encoderConfig``: serialized ``EncoderConfig``
    - ``classifierConfig``: serialized ``ClassifierConfig``
    - ``spectrogramConfig``: serialized ``SpectrogramConfig``
    - ``encoderState``: encoder state dict
    - ``classifierState``: classifier state dict
    - ``classes``: class name → index mapping

    If ``metadata`` is supplied, a validated seventh key contains primitive
    v2 task, normalization, geometry, and optional decision settings. Calls
    without metadata retain the exact ANIMAL-SPOT-derived six-key contract.
    """
    validated = (
        _validated_metadata(metadata, spec_config, class_dist_dict, classifier_config.num_classes)
        if metadata is not None
        else None
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model = model.cpu()
    encoder = encoder.cpu()
    classifier = classifier.cpu()

    save_dict = {
        "encoderConfig": asdict(encoder_config),
        "classifierConfig": asdict(classifier_config),
        "spectrogramConfig": asdict(spec_config),
        "encoderState": encoder.state_dict(),
        "classifierState": classifier.state_dict(),
        "classes": class_dist_dict,
    }
    if validated is not None:
        save_dict["metadata"] = validated
    torch.save(save_dict, str(path))


def load_model(
    path: str | Path,
) -> dict[str, Any]:
    """Load a model dict saved by ``save_model``.

    Returns the raw dict. The consumer is responsible for instantiating
    the encoder and classifier from the stored configs and loading
    state dicts.
    """
    return torch.load(str(path), map_location="cpu", weights_only=True)  # type: ignore[no-any-return]
