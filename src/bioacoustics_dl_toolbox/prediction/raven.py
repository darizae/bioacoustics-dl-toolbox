"""Pure Raven selection-table export for completed :class:`WindowPrediction` records.

This adapter intentionally does not read prediction logs, apply a second
detection threshold, or assume a noise class.
Window classifications are *not* automatically behavioral events.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import TextIO

from bioacoustics_dl_toolbox.prediction.predictor import WindowPrediction

RAVEN_COLUMNS = (
    "Selection",
    "View",
    "Channel",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Annotation",
)


def _finite_number(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _format_number(value: float) -> str:
    """Consistent decimal point and up to nine fractional places, without exponent notation."""
    return f"{value:.9f}".rstrip("0").rstrip(".") or "0"


def _validated_options(
    sample_rate: int,
    channel: int,
    low_freq_hz: float,
    high_freq_hz: float | None,
    recording_duration_seconds: float | None,
    merge_overlaps: bool,
) -> tuple[float, float, float | None]:
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer in Hz")
    if not isinstance(channel, int) or isinstance(channel, bool) or channel < 1:
        raise ValueError("channel must be a positive 1-based integer")
    if not isinstance(merge_overlaps, bool):
        raise ValueError("merge_overlaps must be a boolean")
    nyquist = sample_rate / 2
    low = _finite_number(low_freq_hz, "low_freq_hz")
    high = nyquist if high_freq_hz is None else _finite_number(high_freq_hz, "high_freq_hz")
    if not 0 <= low < high <= nyquist:
        raise ValueError("frequency bounds must satisfy 0 <= low < high <= sample_rate / 2")
    duration = None
    if recording_duration_seconds is not None:
        duration = _finite_number(recording_duration_seconds, "recording_duration_seconds")
        if duration < 0:
            raise ValueError("recording_duration_seconds must be nonnegative")
    return low, high, duration


def predictions_to_raven_rows(
    predictions: Iterable[WindowPrediction],
    sample_rate: int,
    *,
    recording_duration_seconds: float | None = None,
    channel: int = 1,
    low_freq_hz: float = 0.0,
    high_freq_hz: float | None = None,
    merge_overlaps: bool = False,
) -> list[dict[str, str]]:
    """Convert already-computed windows to ordered Raven selection-table rows.

    Uses the predictor's ``predicted_labels`` decisions, **not** new score
    thresholds. The keys of ``probabilities`` encode ascending class-index
    order (SPEC-06); every window must use the same keys in the same order.
    Each active label receives its own selection, even when labels coincide.

    Timestamps come from ``start_seconds`` and ``end_seconds``, validated
    against the half-open sample interval and ``sample_rate``. Optionally pass
    ``recording_duration_seconds`` to clamp a tail ending beyond the recording;
    starts outside the recording are rejected. The default frequency interval
    is the full sampled band, not the model's feature-frequency cutoffs.
    Merging is *off* by default; enabling it joins overlapping/touching
    intervals independently for each label, without inferring behavioral events.

    Row values are TSV-ready strings, with deterministic up-to-9-decimal-place
    formatting. Selection numbers start at one; rows sort by start time and
    class-index order. The caller's records are never modified.
    """
    low, high, duration = _validated_options(
        sample_rate,
        channel,
        low_freq_hz,
        high_freq_hz,
        recording_duration_seconds,
        merge_overlaps,
    )
    class_names: tuple[str, ...] | None = None
    events: list[tuple[float, float, str, int]] = []
    for prediction in predictions:
        if not isinstance(prediction, WindowPrediction):
            raise TypeError("predictions must contain WindowPrediction records")
        if (
            not isinstance(prediction.start_sample, int)
            or isinstance(prediction.start_sample, bool)
            or not isinstance(prediction.end_sample, int)
            or isinstance(prediction.end_sample, bool)
            or not 0 <= prediction.start_sample < prediction.end_sample
        ):
            raise ValueError("window start_sample/end_sample must be increasing sample indices")
        start = _finite_number(prediction.start_seconds, "start_seconds")
        end = _finite_number(prediction.end_seconds, "end_seconds")
        if start < 0 or end <= start:
            raise ValueError("window start_seconds/end_seconds must satisfy 0 <= start < end")
        # Both representations are produced by SPEC-06. Do not silently
        # export a different timeline if the supplied sample rate is wrong.
        if not (
            math.isclose(start, prediction.start_sample / sample_rate, abs_tol=1e-6, rel_tol=0)
            and math.isclose(end, prediction.end_sample / sample_rate, abs_tol=1e-6, rel_tol=0)
        ):
            raise ValueError("window seconds must match sample indices and sample_rate")
        if duration is not None:
            if start >= duration:
                raise ValueError("window start_seconds is outside recording_duration_seconds")
            end = min(end, duration)
        if not isinstance(prediction.probabilities, Mapping) or not prediction.probabilities:
            raise ValueError("probabilities must be a nonempty class-name mapping")
        names = tuple(prediction.probabilities)
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("probabilities must have nonempty string class names")
        if class_names is None:
            class_names = names
        elif names != class_names:
            raise ValueError("probabilities must use the same class names in class-index order")
        for name, score in prediction.probabilities.items():
            probability = _finite_number(score, f"probability for {name!r}")
            if not 0 <= probability <= 1:
                raise ValueError(f"probability for {name!r} must be between 0 and 1")
        labels = prediction.predicted_labels
        if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
            raise ValueError("predicted_labels must be a sequence of class names")
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("predicted_labels must contain nonempty string class names")
        if len(labels) != len(set(labels)):
            raise ValueError("predicted_labels contains duplicate classes")
        if any(label not in prediction.probabilities for label in labels):
            raise ValueError("predicted_labels contains a class absent from probabilities")
        selected = set(labels)
        for class_order, name in enumerate(names):
            if name in selected:
                events.append((start, end, name, class_order))

    if merge_overlaps:
        # Sort within each class before taking interval unions; the original
        # record order need not be chronological. Other labels are unaffected.
        by_label: dict[str, list[tuple[float, float, str, int]]] = {}
        for event in events:
            by_label.setdefault(event[2], []).append(event)
        events = []
        for label_events in by_label.values():
            for start, end, name, order in sorted(label_events, key=lambda e: e[0]):
                if events and events[-1][2] == name and start <= events[-1][1]:
                    previous = events[-1]
                    events[-1] = (previous[0], max(previous[1], end), name, order)
                else:
                    events.append((start, end, name, order))

    events.sort(key=lambda event: (event[0], event[3]))
    rows: list[dict[str, str]] = []
    for selection, (start, end, label, _) in enumerate(events, start=1):
        rows.append(
            dict(
                zip(
                    RAVEN_COLUMNS,
                    (
                        str(selection),
                        "Spectrogram 1",
                        str(channel),
                        _format_number(start),
                        _format_number(end),
                        _format_number(low),
                        _format_number(high),
                        label,
                    ),
                )
            )
        )
    return rows


def write_raven_tsv(
    predictions: Iterable[WindowPrediction],
    sample_rate: int,
    destination: str | Path | TextIO,
    *,
    recording_duration_seconds: float | None = None,
    channel: int = 1,
    low_freq_hz: float = 0.0,
    high_freq_hz: float | None = None,
    merge_overlaps: bool = False,
) -> None:
    """Write a UTF-8 Raven TSV to a path or an existing writable text stream.

    Always writes the exact eight-column header, even for no detections.
    Opens paths with ``newline=''``; leaves caller-owned streams open.
    Uses :mod:`csv` to quote tabs, quotes, and newlines in class names.
    """
    rows = predictions_to_raven_rows(
        predictions,
        sample_rate,
        recording_duration_seconds=recording_duration_seconds,
        channel=channel,
        low_freq_hz=low_freq_hz,
        high_freq_hz=high_freq_hz,
        merge_overlaps=merge_overlaps,
    )

    def write(stream: TextIO) -> None:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(RAVEN_COLUMNS)
        writer.writerows([row[column] for column in RAVEN_COLUMNS] for row in rows)

    if isinstance(destination, (str, Path)):
        with open(destination, "w", encoding="utf-8", newline="") as stream:
            write(stream)
    else:
        write(destination)
