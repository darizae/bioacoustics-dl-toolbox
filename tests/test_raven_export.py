"""SPEC-10: pure in-memory Raven TSV conversion from synthetic prediction records."""

import csv
import io
from contextlib import nullcontext
import math
from dataclasses import replace

import pytest

from bioacoustics_dl_toolbox.prediction import (
    RAVEN_COLUMNS,
    WindowPrediction,
    predictions_to_raven_rows,
    write_raven_tsv,
)
from bioacoustics_dl_toolbox.prediction import raven

SAMPLE_RATE = 8000
HEADER = (
    "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\t"
    "Low Freq (Hz)\tHigh Freq (Hz)\tAnnotation\n"
)


def window(start, end, labels, *, probabilities=None, index=0):
    return WindowPrediction(
        index=index,
        start_sample=start,
        end_sample=end,
        start_seconds=start / SAMPLE_RATE,
        end_seconds=end / SAMPLE_RATE,
        probabilities=probabilities or {"bird": 0.8, "frog": 0.9, "rain": 0.1},
        predicted_labels=list(labels),
    )


def render(predictions, **kwargs):
    stream = io.StringIO(newline="")
    write_raven_tsv(predictions, SAMPLE_RATE, stream, **kwargs)
    return stream.getvalue()


def test_empty_input_has_header_only():
    assert predictions_to_raven_rows([], SAMPLE_RATE) == []
    assert render([]) == HEADER
    assert tuple(next(csv.reader(io.StringIO(HEADER), delimiter="\t"))) == RAVEN_COLUMNS


def test_two_active_labels_keep_distinct_rows_and_class_mapping_order():
    prediction = window(4000, 12000, ["frog", "bird"])
    rows = predictions_to_raven_rows([prediction], SAMPLE_RATE)
    assert [r["Selection"] for r in rows] == ["1", "2"]
    assert [r["Annotation"] for r in rows] == ["bird", "frog"]
    assert list(rows[0]) == list(RAVEN_COLUMNS)
    assert rows[0] == {
        "Selection": "1",
        "View": "Spectrogram 1",
        "Channel": "1",
        "Begin Time (s)": "0.5",
        "End Time (s)": "1.5",
        "Low Freq (Hz)": "0",
        "High Freq (Hz)": "4000",
        "Annotation": "bird",
    }
    assert render([prediction]) == (
        HEADER
        + "1\tSpectrogram 1\t1\t0.5\t1.5\t0\t4000\tbird\n"
        + "2\tSpectrogram 1\t1\t0.5\t1.5\t0\t4000\tfrog\n"
    )


def test_sort_by_start_then_class_index_not_window_index_or_input_order():
    rows = predictions_to_raven_rows(
        [window(8000, 16000, ["bird"], index=0), window(0, 12000, ["frog", "bird"], index=4)],
        SAMPLE_RATE,
    )
    assert [(r["Begin Time (s)"], r["Annotation"]) for r in rows] == [
        ("0", "bird"),
        ("0", "frog"),
        ("1", "bird"),
    ]


def test_merge_only_same_label_overlapping_or_touching_and_never_concurrent():
    predictions = [
        window(0, 8000, ["bird", "frog"]),
        window(4000, 12000, ["bird", "frog"]),
        window(12000, 16000, ["bird"]),
    ]
    unmerged = predictions_to_raven_rows(predictions, SAMPLE_RATE)
    assert len(unmerged) == 5
    merged = predictions_to_raven_rows(predictions, SAMPLE_RATE, merge_overlaps=True)
    assert [(r["Annotation"], r["Begin Time (s)"], r["End Time (s)"]) for r in merged] == [
        ("bird", "0", "2"),
        ("frog", "0", "1.5"),
    ]
    assert [r["Selection"] for r in merged] == ["1", "2"]
    assert len(predictions[0].predicted_labels) == 2  # pure; did not modify source


def test_merge_stays_separate_across_gaps_and_works_out_of_order():
    predictions = [
        window(24000, 32000, ["frog"]),
        window(4000, 16000, ["bird"]),
        window(0, 8000, ["bird"]),
        window(20000, 24000, ["frog"]),
        window(16001, 20000, ["bird"]),
    ]
    rows = predictions_to_raven_rows(predictions, SAMPLE_RATE, merge_overlaps=True)
    assert [(r["Annotation"], r["Begin Time (s)"], r["End Time (s)"]) for r in rows] == [
        ("bird", "0", "2"),
        ("bird", "2.000125", "2.5"),
        ("frog", "2.5", "4"),
    ]


def test_clamp_explicit_recording_end_and_validate_begin_bound():
    prediction = window(7500, 9000, ["bird"])
    rows = predictions_to_raven_rows(
        [prediction], SAMPLE_RATE, recording_duration_seconds=1.0
    )
    assert rows[0]["Begin Time (s)"] == "0.9375"
    assert rows[0]["End Time (s)"] == "1"
    with pytest.raises(ValueError, match="outside recording_duration_seconds"):
        predictions_to_raven_rows(
            [window(8000, 9000, ["bird"])],
            SAMPLE_RATE,
            recording_duration_seconds=1.0,
        )


@pytest.mark.parametrize("sample_rate", [0, -1, 0.5, True, float("nan")])
def test_invalid_sampling_rate(sample_rate):
    with pytest.raises(ValueError, match="sample_rate"):
        predictions_to_raven_rows([], sample_rate)


@pytest.mark.parametrize("channel", [0, -1, 1.5, True])
def test_invalid_channel(channel):
    with pytest.raises(ValueError, match="channel"):
        predictions_to_raven_rows([], SAMPLE_RATE, channel=channel)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"low_freq_hz": -1}, "frequency bounds"),
        ({"high_freq_hz": 4001}, "frequency bounds"),
        ({"low_freq_hz": 1000, "high_freq_hz": 1000}, "frequency bounds"),
        ({"low_freq_hz": 1001, "high_freq_hz": 1000}, "frequency bounds"),
        ({"low_freq_hz": math.nan}, "low_freq_hz"),
        ({"high_freq_hz": math.inf}, "high_freq_hz"),
        ({"recording_duration_seconds": -1}, "recording_duration_seconds"),
        ({"recording_duration_seconds": math.nan}, "recording_duration_seconds"),
        ({"merge_overlaps": 1}, "merge_overlaps"),
    ],
)
def test_invalid_options_raise_even_without_predictions(kwargs, message):
    with pytest.raises(ValueError, match=message):
        predictions_to_raven_rows([], SAMPLE_RATE, **kwargs)


def test_explicit_frequency_bounds_and_channel_do_not_use_model_cutoffs():
    row = predictions_to_raven_rows(
        [window(0, 8000, ["bird"])],
        SAMPLE_RATE,
        channel=2,
        low_freq_hz=200.125,
        high_freq_hz=3200.5,
    )[0]
    assert (row["Channel"], row["Low Freq (Hz)"], row["High Freq (Hz)"]) == (
        "2", "200.125", "3200.5"
    )


@pytest.mark.parametrize(
    ("changes", "error"),
    [
        ({"start_seconds": math.nan}, "start_seconds"),
        ({"end_seconds": math.inf}, "end_seconds"),
        ({"start_seconds": -0.1}, "start_seconds/end_seconds"),
        ({"end_seconds": 0.0}, "start_seconds/end_seconds"),
        ({"start_seconds": 1.5}, "start_seconds/end_seconds"),
        ({"start_sample": -1}, "start_sample/end_sample"),
        ({"end_sample": 0}, "start_sample/end_sample"),
        ({"end_sample": 9000}, "seconds must match"),
        ({"predicted_labels": ["bird", "bird"]}, "duplicate"),
        ({"predicted_labels": ["unknown"]}, "absent"),
        ({"predicted_labels": [["bird"]]}, "nonempty string"),
        ({"probabilities": {"bird": 1.1}}, "probability"),
    ],
)
def test_invalid_prediction_fields(changes, error):
    prediction = replace(window(0, 8000, ["bird"]), **changes)
    with pytest.raises(ValueError, match=error):
        predictions_to_raven_rows([prediction], SAMPLE_RATE)


def test_reject_inconsistent_class_mapping_order_across_windows():
    predictions = [
        window(0, 8000, ["bird"]),
        window(8000, 16000, ["frog"], probabilities={"frog": 0.5, "bird": 0.5, "rain": 0.5}),
    ]
    with pytest.raises(ValueError, match="class-index order"):
        predictions_to_raven_rows(predictions, SAMPLE_RATE)


def test_no_active_label_emits_header_and_does_not_infer_threshold():
    assert render([window(0, 8000, [], probabilities={"bird": 0.99})]) == HEADER


def test_tabs_quotes_unicode_and_precision_are_quoted_without_loss():
    bird = 'pájaro\t"cantor"'
    frogs = "雨声\nchorus"
    probability_map = {bird: 0.9, frogs: 0.9}
    prediction = window(1, 3, [frogs, bird], probabilities=probability_map)
    content = render([prediction])
    assert '"pájaro\t""cantor"""' in content
    assert '"雨声\nchorus"' in content
    cells = list(csv.DictReader(io.StringIO(content), delimiter="\t"))
    assert [row["Annotation"] for row in cells] == [bird, frogs]
    assert [row["Begin Time (s)"] for row in cells] == ["0.000125", "0.000125"]
    assert [row["End Time (s)"] for row in cells] == ["0.000375", "0.000375"]
    assert content == render([prediction])


def test_writer_calls_pure_generator_and_leaves_stream_open(monkeypatch):
    record = window(0, 8000, ["bird"])
    original = raven.predictions_to_raven_rows
    calls = []

    def spy(predictions, sample_rate, **kwargs):
        calls.append((predictions, sample_rate, kwargs))
        return original(predictions, sample_rate, **kwargs)

    monkeypatch.setattr(raven, "predictions_to_raven_rows", spy)
    stream = io.StringIO()
    write_raven_tsv([record], SAMPLE_RATE, stream, merge_overlaps=True)
    assert len(calls) == 1
    assert calls[0][1] == SAMPLE_RATE
    assert calls[0][2]["merge_overlaps"] is True
    assert not stream.closed
    assert stream.getvalue().startswith(HEADER)


def test_path_writer_uses_utf8_and_newlines_with_in_memory_open(monkeypatch):
    # Exercise the path branch without writing a real file.
    label = "á\t鳥"
    prediction = window(0, 8000, [label], probabilities={label: 0.9})
    stream = io.StringIO(newline="")
    opened = []

    def fake_open(path, mode, *, encoding, newline):
        opened.append((path, mode, encoding, newline))
        return nullcontext(stream)

    monkeypatch.setattr("builtins.open", fake_open)
    write_raven_tsv([prediction], SAMPLE_RATE, "raven.txt")
    assert opened == [("raven.txt", "w", "utf-8", "")]
    assert stream.getvalue() == render([prediction])


def test_invalid_data_does_not_open_output(monkeypatch):
    opened = []
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: opened.append(args))
    with pytest.raises(ValueError, match="start_seconds/end_seconds"):
        write_raven_tsv(
            [replace(window(0, 8000, ["bird"]), end_seconds=-5)], SAMPLE_RATE, "raven.txt"
        )
    assert opened == []
