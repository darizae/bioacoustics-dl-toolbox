"""SPEC-07 exported-model metadata and separate epoch-resume task checks."""

from dataclasses import asdict

import pytest
import torch
from torch import nn

from bioacoustics_dl_toolbox.config import (
    ClassifierConfig,
    EncoderConfig,
    NormalizationConfig,
    SpectrogramConfig,
    TaskConfig,
)
from bioacoustics_dl_toolbox.training.checkpoints import (
    extract_model_metadata,
    load_model,
    save_model,
    validate_resume_checkpoint_task,
)

CLASSES = {"frog": 1, "bird": 0, "rain": 2}
SPEC = SpectrogramConfig(sample_rate=48000, n_fft=1024, hop_length=256)
WINDOW = 2048
FRAMES = 5  # 1 + (2048 - 1024) // 256, center=False


def _model_parts(class_count=3):
    encoder = nn.Linear(3, 4)
    classifier = nn.Linear(4, class_count)
    return (
        nn.Sequential(encoder, classifier),
        encoder,
        EncoderConfig(),
        classifier,
        ClassifierConfig(num_classes=class_count),
    )


def _metadata(**overrides):
    result = {
        "schema_version": 2,
        "task_type": "multilabel",
        "normalization_config": asdict(NormalizationConfig(mode="min_max")),
        "window_length_samples": WINDOW,
        "sequence_length_frames": FRAMES,
        "hop_samples": 1024,
        "thresholds": {"rain": 0.7, "frog": 0.6, "bird": 0.4},
        "class_names": ["bird", "frog", "rain"],
    }
    result.update(overrides)
    return result


def _save(tmp_path, metadata=None, classes=None, class_count=3):
    args = _model_parts(class_count)
    path = tmp_path / "trained_model.pt"
    # Preexisting positional arguments remain valid.
    save_model(*args[:5], SPEC, path, CLASSES if classes is None else classes, metadata=metadata)
    return path, args


def test_six_key_export_is_unchanged_with_old_positional_call(tmp_path):
    args = _model_parts()
    path = tmp_path / "legacy.pt"
    save_model(*args[:5], SPEC, path, CLASSES)
    raw = load_model(path)
    assert set(raw) == {
        "encoderConfig",
        "classifierConfig",
        "spectrogramConfig",
        "encoderState",
        "classifierState",
        "classes",
    }
    assert raw["classes"] == CLASSES
    assert raw["spectrogramConfig"] == asdict(SPEC)
    for key, original in (("encoderState", args[1]), ("classifierState", args[3])):
        for name, weight in original.state_dict().items():
            torch.testing.assert_close(raw[key][name], weight)


def test_v2_weights_only_round_trip_preserves_primitives_and_weights(tmp_path):
    input_metadata = _metadata(
        task_type=TaskConfig(task_type="multilabel"),
        normalization_config=NormalizationConfig(mode="min_max"),
    )
    path, parts = _save(tmp_path, metadata=input_metadata)
    payload = torch.load(path, weights_only=True)
    assert load_model(path).keys() == payload.keys()
    assert set(payload) == {
        "encoderConfig",
        "classifierConfig",
        "spectrogramConfig",
        "encoderState",
        "classifierState",
        "classes",
        "metadata",
    }
    expected = _metadata(thresholds={"bird": 0.4, "frog": 0.6, "rain": 0.7})
    assert payload["metadata"] == expected
    assert payload["classes"] == CLASSES
    for key, original in (("encoderState", parts[1]), ("classifierState", parts[3])):
        for name, weight in original.state_dict().items():
            torch.testing.assert_close(payload[key][name], weight)
    descriptor = extract_model_metadata(payload)
    assert descriptor["legacy"] is False
    assert descriptor["is_complete"] is True
    assert descriptor["task_type"] == "multilabel"
    assert descriptor["normalization_config"]["mode"] == "min_max"
    assert descriptor["window_length_samples"] == WINDOW
    assert descriptor["sequence_length_frames"] == FRAMES
    assert descriptor["hop_samples"] == 1024
    assert descriptor["thresholds"] == expected["thresholds"]
    assert descriptor["class_names"] == ["bird", "frog", "rain"]


def test_legacy_parser_never_infers_task_normalization_frames_or_thresholds(tmp_path):
    path, _ = _save(tmp_path)
    descriptor = extract_model_metadata(load_model(path))
    assert descriptor["schema_version"] == 1
    assert descriptor["legacy"] is True
    assert descriptor["is_complete"] is False
    for name in (
        "task_type",
        "normalization_config",
        "window_length_samples",
        "sequence_length_frames",
        "hop_samples",
        "thresholds",
        "class_names",
    ):
        assert descriptor[name] is None


def test_v2_optional_hop_is_explicitly_incomplete_for_prediction(tmp_path):
    minimal = {
        "schema_version": 2,
        "task_type": "multilabel",
        "normalization_config": asdict(NormalizationConfig()),
        "window_length_samples": WINDOW,
    }
    path, _ = _save(tmp_path, metadata=minimal)
    descriptor = extract_model_metadata(load_model(path))
    assert descriptor["is_complete"] is False
    assert descriptor["hop_samples"] is None
    assert descriptor["thresholds"] is None
    assert descriptor["sequence_length_frames"] is None


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (3, "Unsupported checkpoint metadata schema_version"),
        (1, "Unsupported checkpoint metadata schema_version"),
        (True, "Unsupported checkpoint metadata schema_version"),
        ("2", "Unsupported checkpoint metadata schema_version"),
        (None, "Unsupported checkpoint metadata schema_version"),
    ],
)
def test_invalid_metadata_version_is_rejected(tmp_path, value, message):
    with pytest.raises(ValueError, match=message):
        _save(tmp_path, metadata=_metadata(schema_version=value))


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ({"task_type": "auto"}, "must be resolved"),
        ({"task_type": "binary"}, "exactly 2 classes"),
        ({"task_type": "mystery"}, "Unknown task_type"),
        ({"normalization_config": {"mode": "min_max"}}, "normalization_config is missing"),
        (
            {"normalization_config": {"mode": "bogus", "min_level_db": -100, "ref_level_db": 20}},
            "mode",
        ),
        ({"window_length_samples": 0}, "window_length_samples"),
        ({"window_length_samples": True}, "window_length_samples"),
        ({"hop_samples": WINDOW + 1}, "hop_samples"),
        ({"sequence_length_frames": FRAMES + 1}, "sequence_length_frames"),
        ({"class_names": ["frog", "bird", "rain"]}, "class_names"),
        ({"thresholds": [0.3, 0.7]}, "exactly 3"),
        ({"thresholds": {"bird": 0.3, "frog": 0.7}}, "mapping must match"),
        ({"thresholds": {"bird": 0.3, "frog": 0.7, "rain": 1.0}}, "strictly between"),
        ({"thresholds": 0.0}, "strictly between"),
        ({"thresholds": float("nan")}, "finite"),
        ({"thresholds": True}, "numeric"),
    ],
)
def test_invalid_v2_metadata_is_rejected_on_save(tmp_path, invalid, message):
    with pytest.raises(ValueError, match=message):
        _save(tmp_path, metadata=_metadata(**invalid))


def test_metadata_required_fields_and_unknown_fields_rejected(tmp_path):
    for missing in ("task_type", "normalization_config", "window_length_samples"):
        metadata = _metadata()
        del metadata[missing]
        with pytest.raises(ValueError, match=f"metadata is missing: {missing}"):
            _save(tmp_path, metadata=metadata)
    with pytest.raises(ValueError, match="unknown fields"):
        _save(tmp_path, metadata=_metadata(undocumented=True))


@pytest.mark.parametrize(
    "classes",
    [
        {"bird": 0, "frog": 0, "rain": 2},
        {"bird": 0, "frog": 2, "rain": 3},
        {"bird": 0, "frog": 1, "rain": True},
    ],
)
def test_class_indices_must_be_unique_and_contiguous_for_v2(tmp_path, classes):
    with pytest.raises(ValueError, match="unique, contiguous"):
        _save(tmp_path, metadata=_metadata(), classes=classes)


def test_v2_classifier_count_must_match_mapping(tmp_path):
    with pytest.raises(ValueError, match="classifierConfig num_classes"):
        _save(tmp_path, metadata=_metadata(), class_count=2)


def test_threshold_scalar_and_vector_stored_as_plain_data(tmp_path):
    for thresholds, expected in ((0.5, 0.5), ([0.3, 0.4, 0.6], [0.3, 0.4, 0.6])):
        path, _ = _save(tmp_path, metadata=_metadata(thresholds=thresholds))
        assert load_model(path)["metadata"]["thresholds"] == expected


def test_v2_extraction_validates_malformed_payload_not_only_save(tmp_path):
    path, _ = _save(tmp_path, metadata=_metadata())
    payload = load_model(path)
    payload["metadata"]["schema_version"] = 99
    with pytest.raises(ValueError, match="migrate the checkpoint"):
        extract_model_metadata(payload)
    payload["metadata"] = _metadata()
    del payload["metadata"]["task_type"]
    with pytest.raises(ValueError, match="metadata is missing: task_type"):
        extract_model_metadata(payload)
    payload["metadata"] = _metadata()
    payload["spectrogramConfig"] = {"sample_rate": SPEC.sample_rate}
    with pytest.raises(ValueError, match="spectrogramConfig is missing"):
        extract_model_metadata(payload)


def test_resume_task_consistency_is_separate_and_accepts_legacy_plain_dict():
    legacy = {"modelState": {}, "trainState": {"epoch": 2}, "classes": CLASSES}
    validate_resume_checkpoint_task(legacy, "multilabel")
    validate_resume_checkpoint_task({**legacy, "taskType": "multilabel"}, "multilabel")
    with pytest.raises(ValueError, match="does not match"):
        validate_resume_checkpoint_task({**legacy, "taskType": "binary"}, "multilabel")
    with pytest.raises(ValueError, match="does not match"):
        validate_resume_checkpoint_task({**legacy, "taskType": "multilabel"}, "binary")
