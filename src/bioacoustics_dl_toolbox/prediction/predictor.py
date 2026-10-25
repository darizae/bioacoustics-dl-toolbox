"""ANIMAL-SPOT-inspired sliding-window prediction as a headless library API.

The window dataset and spectrogram/normalization stages are shared with
training. Unlike ANIMAL-SPOT's CLI, this module has no visualization, file
output, or event-merging side effects. Checkpoint construction follows the
existing encoder/classifier ``trained_model.pt`` contract.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from bioacoustics_dl_toolbox.audio.datasets import StridedAudioDataset
from bioacoustics_dl_toolbox.config import (
    ClassifierConfig,
    EncoderConfig,
    NormalizationConfig,
    SpectrogramConfig,
    TaskConfig,
    resolve_task_type,
)
from bioacoustics_dl_toolbox.metrics.multilabel import (
    multilabel_probabilities,
    threshold_multilabel,
)
from bioacoustics_dl_toolbox.models.classifier import Classifier
from bioacoustics_dl_toolbox.models.encoder import ResidualEncoder
from bioacoustics_dl_toolbox.training.checkpoints import extract_model_metadata, load_model


@dataclass(frozen=True)
class WindowPrediction:
    """Prediction for the half-open recording interval ``[start_sample, end_sample)``.

    Probabilities are inserted in ascending class-index order. Windows may
    overlap; no cross-window event merging is performed.
    """

    index: int
    start_sample: int
    end_sample: int
    start_seconds: float
    end_seconds: float
    probabilities: dict[str, float]
    predicted_labels: list[str]


def _ordered_classes(mapping: Mapping[str, int]) -> list[str]:
    if not isinstance(mapping, Mapping) or not mapping:
        raise ValueError("class_to_index must be a nonempty class-name to index mapping")
    if any(not isinstance(name, str) or not name for name in mapping):
        raise ValueError("class_to_index names must be nonempty strings")
    indices = list(mapping.values())
    if any(not isinstance(idx, int) or isinstance(idx, bool) for idx in indices) or set(
        indices
    ) != set(range(len(indices))):
        raise ValueError("class_to_index indices must be unique, contiguous integers 0..C-1")
    return sorted(mapping, key=mapping.__getitem__)


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer (audio samples or batch count)")
    return value


def _validated_thresholds(
    thresholds: float | Sequence[float] | Mapping[str, float] | torch.Tensor | None,
    names: list[str],
    multilabel: bool,
) -> float | list[float] | None:
    if not multilabel:
        if thresholds is not None:
            raise ValueError("thresholds are only supported for multilabel tasks")
        return None
    if thresholds is None:
        return 0.5  # SPEC-05's documented independent-label default.
    if isinstance(thresholds, Mapping):
        if set(thresholds) != set(names):
            raise ValueError("thresholds must name every class exactly once")
        thresholds = [thresholds[name] for name in names]
    if isinstance(thresholds, (str, bytes)):
        raise ValueError("thresholds must be a numeric scalar or one per class")
    try:
        values = torch.as_tensor(thresholds, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError("thresholds must be a numeric scalar or one per class") from exc
    if values.ndim > 1 or (values.ndim == 1 and values.numel() != len(names)):
        raise ValueError(f"thresholds must be a scalar or length-{len(names)} per-class values")
    if values.ndim == 0 and isinstance(thresholds, bool):
        raise ValueError("thresholds must be numeric probabilities, not bool")
    if not bool(torch.isfinite(values).all()) or not bool(((values > 0) & (values < 1)).all()):
        raise ValueError("thresholds must be finite and strictly between 0 and 1")
    if values.ndim == 0:
        return float(values.item())
    return values.tolist()


def _config_from_bundle(config_type: type, data: object, name: str) -> Any:
    """Do not fill missing saved feature settings from current dataclass defaults."""
    if not isinstance(data, Mapping):
        raise ValueError(f"checkpoint {name} must be a configuration mapping")
    required = {field.name for field in fields(config_type)}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"checkpoint {name} is missing: {', '.join(sorted(missing))}")
    return config_type(**data)


def _bundle_inputs(
    source: Mapping[str, Any],
    *,
    class_to_index: Mapping[str, int] | None,
    task_config: TaskConfig | None,
    spec_config: SpectrogramConfig | None,
    norm_config: NormalizationConfig | None,
    window_samples: int | None,
    hop_samples: int | None,
    thresholds: float | Sequence[float] | Mapping[str, float] | torch.Tensor | None,
) -> tuple[
    Mapping[str, int],
    TaskConfig,
    SpectrogramConfig,
    NormalizationConfig,
    int,
    int,
    float | Sequence[float] | Mapping[str, float] | torch.Tensor | None,
]:
    """Resolve v2 metadata, but never make up missing v1 inference settings."""
    metadata = extract_model_metadata(source)

    if class_to_index is None:
        class_to_index = source.get("classes")
    elif "classes" in source and dict(class_to_index) != source["classes"]:
        raise ValueError("class_to_index conflicts with the checkpoint's classes mapping")
    if spec_config is None and source.get("spectrogramConfig") is not None:
        spec_config = _config_from_bundle(
            SpectrogramConfig, source["spectrogramConfig"], "spectrogramConfig"
        )
    if task_config is None and metadata.get("task_type") is not None:
        task_config = TaskConfig(task_type=metadata["task_type"])
    if norm_config is None and metadata.get("normalization_config") is not None:
        norm_config = _config_from_bundle(
            NormalizationConfig, metadata["normalization_config"], "normalization_config"
        )
    if window_samples is None:
        window_samples = metadata.get("window_length_samples")
    if hop_samples is None:
        hop_samples = metadata.get("hop_samples")
    if thresholds is None:
        thresholds = metadata.get("thresholds")

    missing = [
        name
        for name, value in (
            ("class_to_index", class_to_index),
            ("spec_config", spec_config),
            ("task_config", task_config),
            ("norm_config", norm_config),
            ("window_samples", window_samples),
            ("hop_samples", hop_samples),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Checkpoint lacks inference metadata: "
            + ", ".join(missing)
            + ". Supply these explicitly for a legacy six-key checkpoint "
            "or export a bundle with resolved v2 metadata."
        )
    return (
        class_to_index,
        task_config,
        spec_config,
        norm_config,
        window_samples,
        hop_samples,
        thresholds,
    )


def _restore_model(source: Mapping[str, Any]) -> nn.Module:
    """Reconstruct exactly the encoder/classifier used by ``save_model``."""
    try:
        encoder = ResidualEncoder(EncoderConfig(**source["encoderConfig"]))
        classifier = Classifier(ClassifierConfig(**source["classifierConfig"]))
        encoder.load_state_dict(source["encoderState"], strict=True)
        classifier.load_state_dict(source["classifierState"], strict=True)
    except KeyError as exc:
        raise ValueError(f"checkpoint is missing required model field {exc.args[0]!r}") from exc
    return nn.Sequential(OrderedDict((("encoder", encoder), ("classifier", classifier))))


def predict_windows(
    model: nn.Module | Mapping[str, Any] | str | Path,
    audio_file: str | Path,
    *,
    class_to_index: Mapping[str, int] | None = None,
    task_config: TaskConfig | None = None,
    spec_config: SpectrogramConfig | None = None,
    norm_config: NormalizationConfig | None = None,
    window_samples: int | None = None,
    hop_samples: int | None = None,
    thresholds: float | Sequence[float] | Mapping[str, float] | torch.Tensor | None = None,
    batch_size: int = 16,
    device: str | torch.device | None = None,
) -> list[WindowPrediction]:
    """Predict each overlapping window in recording order, without aggregation.

    ``model`` may be any ``nn.Module`` returning floating ``[B, C]`` *raw*
    logits, an exported-model dict, or its path (loaded with ``weights_only``).
    An explicit module requires all inference settings except thresholds;
    a v2 bundle can supply them from metadata. Legacy six-key bundles have
    no normalization, task, window length or hop metadata: callers must give
    these explicitly. ``spec_config`` and ``classes`` can use the checkpoint's
    existing fields. Multilabel thresholds default to 0.5 (SPEC-05).

    Windows start at 0, hop, ... through the first window reaching the last
    sample. The final interval is clamped to the resampled recording length,
    and short-window features are padded to a fixed shape by the shared
    deterministic feature pipeline. All intervals use half-open sample bounds.

    Explicit modules stay on their current device if ``device`` is omitted
    (usually CPU). Specifying a device moves the module there and leaves it
    there; a reconstructed bundle starts on CPU. The module's previous
    training/eval flags are restored even when prediction fails. No weights
    or gradients are changed.
    """
    bundle = None
    if not isinstance(model, nn.Module):
        if isinstance(model, (str, Path)):
            model = load_model(model)
        if not isinstance(model, Mapping):
            raise TypeError("model must be an nn.Module, checkpoint mapping, or checkpoint path")
        bundle = model
        (
            class_to_index,
            task_config,
            spec_config,
            norm_config,
            window_samples,
            hop_samples,
            thresholds,
        ) = _bundle_inputs(
            bundle,
            class_to_index=class_to_index,
            task_config=task_config,
            spec_config=spec_config,
            norm_config=norm_config,
            window_samples=window_samples,
            hop_samples=hop_samples,
            thresholds=thresholds,
        )
    else:
        missing = [
            name
            for name, value in (
                ("class_to_index", class_to_index),
                ("task_config", task_config),
                ("spec_config", spec_config),
                ("norm_config", norm_config),
                ("window_samples", window_samples),
                ("hop_samples", hop_samples),
            )
            if value is None
        ]
        if missing:
            raise ValueError("Explicit models require " + ", ".join(missing))

    names = _ordered_classes(class_to_index)
    resolved = resolve_task_type(task_config, len(names))
    window_samples = _positive_int(window_samples, "window_samples")
    hop_samples = _positive_int(hop_samples, "hop_samples")
    if hop_samples > window_samples:
        raise ValueError("hop_samples cannot exceed window_samples (gapped scan)")
    _positive_int(batch_size, "batch_size")
    validated_thresholds = _validated_thresholds(thresholds, names, resolved == "multilabel")
    if not isinstance(spec_config, SpectrogramConfig):
        raise ValueError("spec_config must be a SpectrogramConfig instance")
    if not isinstance(norm_config, NormalizationConfig):
        raise ValueError("norm_config must be a NormalizationConfig instance")
    if bundle is not None:
        # Validate reproducibility-critical settings before building the encoder.
        model = _restore_model(bundle)

    # ``StridedAudioDataset`` uses the shared deterministic training feature path,
    # including mono resampling and fixed feature-frame geometry.
    dataset = StridedAudioDataset(
        str(audio_file),
        sequence_len=window_samples,
        hop=hop_samples,
        spec_config=spec_config,
        norm_config=norm_config,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    if device is None:
        first_tensor = next(iter(model.parameters()), None)
        if first_tensor is None:
            first_tensor = next(iter(model.buffers()), None)
        target_device = first_tensor.device if first_tensor is not None else torch.device("cpu")
    else:
        target_device = torch.device(device)
        model.to(target_device)

    modes = [(module, module.training) for module in model.modules()]
    results: list[WindowPrediction] = []
    try:
        model.eval()
        with torch.inference_mode():
            for offset, features in enumerate(loader):
                logits = model(features.to(target_device))
                if (
                    not isinstance(logits, torch.Tensor)
                    or logits.ndim != 2
                    or logits.shape != (features.shape[0], len(names))
                    or not logits.is_floating_point()
                ):
                    raise ValueError(
                        f"model must return floating [B, {len(names)}] raw logits; "
                        f"received shape {getattr(logits, 'shape', None)}"
                    )
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("model returned non-finite logits/probabilities")
                probabilities = (
                    multilabel_probabilities(logits)
                    if resolved == "multilabel"
                    else torch.softmax(logits, dim=1)
                )
                if not bool(torch.isfinite(probabilities).all()):
                    raise ValueError("model returned non-finite probabilities")
                if resolved == "multilabel":
                    selected = threshold_multilabel(probabilities, validated_thresholds)
                else:
                    selected = torch.nn.functional.one_hot(
                        probabilities.argmax(dim=1), num_classes=len(names)
                    ).bool()
                for within, (probs, decisions) in enumerate(zip(probabilities, selected)):
                    index = offset * batch_size + within
                    start = index * hop_samples
                    end = min(start + window_samples, dataset.n_frames)
                    results.append(
                        WindowPrediction(
                            index=index,
                            start_sample=start,
                            end_sample=end,
                            start_seconds=start / spec_config.sample_rate,
                            end_seconds=end / spec_config.sample_rate,
                            probabilities={
                                name: float(probs[j].item()) for j, name in enumerate(names)
                            },
                            predicted_labels=[name for j, name in enumerate(names) if decisions[j]],
                        )
                    )
    finally:
        for module, original_mode in modes:
            module.training = original_mode
    return results
