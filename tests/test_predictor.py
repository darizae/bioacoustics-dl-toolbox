"""Unit tests for SPEC-06 (synthetic audio and fake models only)."""

import builtins
import importlib
import math
from dataclasses import replace

import pytest
import torch
from torch import nn

from bioacoustics_dl_toolbox.audio import datasets
from bioacoustics_dl_toolbox.config import NormalizationConfig, SpectrogramConfig, TaskConfig
from bioacoustics_dl_toolbox.prediction import WindowPrediction, predict_windows
from bioacoustics_dl_toolbox.prediction import predictor as module


SPEC = SpectrogramConfig(
    sample_rate=8000, n_fft=4, hop_length=2, n_freq_bins=8, f_min=50, f_max=3000
)
NORM = NormalizationConfig(mode="min_max")
CLASSES = {"frog": 1, "bird": 0, "rain": 2}  # Intentionally not in index order.


class FakeModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.offset = 0
        self.features = []
        self.flags = []

    def forward(self, features):
        assert features.ndim == 4
        self.features.append(features.detach().clone())
        self.flags.append(
            (self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled())
        )
        offset = self.offset
        self.offset += len(features)
        return self.logits[offset:self.offset].to(device=features.device)


@pytest.fixture
def audio(monkeypatch):
    files = {}

    def load(path, sample_rate, mono):
        assert sample_rate == 8000 and mono is True
        value = files[path]
        if isinstance(value, Exception):
            raise value
        return value.clone()

    monkeypatch.setattr(datasets.audio_io, "load_audio_file", load)
    return files


def _predict(model, *, audio_file="sample.wav", **kwargs):
    return predict_windows(
        model, audio_file, class_to_index=kwargs.pop("class_to_index", CLASSES),
        task_config=kwargs.pop("task_config", TaskConfig(task_type="multiclass")),
        spec_config=kwargs.pop("spec_config", SPEC),
        norm_config=kwargs.pop("norm_config", NORM),
        window_samples=kwargs.pop("window_samples", 4),
        hop_samples=kwargs.pop("hop_samples", 4),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("task", "mapping", "logits", "selected"),
    [
        ("binary", {"positive": 1, "negative": 0}, [[-1, 2], [4, -1]], ["positive", "negative"]),
        ("multiclass", CLASSES, [[0, 3, 1], [4, -1, 1]], ["frog", "bird"]),
    ],
)
def test_exclusive_softmax_argmax_and_eval_mode_restored(audio, task, mapping, logits, selected):
    audio["sample.wav"] = torch.arange(8, dtype=torch.float32)[None, :]
    model = FakeModel(logits)
    model.train()
    original_weight = model.weight.detach().clone()
    predictions = _predict(
        model, class_to_index=mapping, task_config=TaskConfig(task), batch_size=1
    )
    assert all(isinstance(result, WindowPrediction) for result in predictions)
    assert model.training is True
    torch.testing.assert_close(model.weight.detach(), original_weight)
    assert model.weight.grad is None
    assert model.flags == [(False, False, True), (False, False, True)]
    assert [p.predicted_labels for p in predictions] == [[x] for x in selected]
    assert all(list(p.probabilities) == sorted(mapping, key=mapping.get) for p in predictions)
    for i, row in enumerate(logits):
        expected = torch.softmax(torch.tensor(row, dtype=torch.float32), dim=0)
        for name, idx in mapping.items():
            assert predictions[i].probabilities[name] == pytest.approx(expected[idx].item())
        assert sum(predictions[i].probabilities.values()) == pytest.approx(1.0)


def test_multilabel_sigmoid_independent_thresholds_and_final_tail(audio):
    audio["sample.wav"] = torch.arange(11, dtype=torch.float32)[None, :]
    model = FakeModel([[10, 10, 0], [-10, -10, -10], [0, 2, 2], [10, -10, 0]])
    results = _predict(
        model, task_config=TaskConfig("multilabel"), hop_samples=3,
        thresholds={"rain": 0.7, "frog": 0.8, "bird": 0.5}, batch_size=3,
        device="cpu",
    )
    assert [p.index for p in results] == [0, 1, 2, 3]
    assert [p.start_sample for p in results] == [0, 3, 6, 9]
    assert [p.end_sample for p in results] == [4, 7, 10, 11]
    assert [p.predicted_labels for p in results] == [
        ["bird", "frog"], [], ["bird", "frog", "rain"], ["bird"]
    ]
    assert [p.start_seconds for p in results] == [x / 8000 for x in [0, 3, 6, 9]]
    assert results[-1].end_seconds == 11 / 8000
    assert len(model.features) == 2
    assert [tuple(x.shape) for x in model.features] == [(3, 1, 1, 8), (1, 1, 1, 8)]
    assert all(torch.isfinite(x).all() for x in model.features)
    assert all(model.flags[i] == (False, False, True) for i in range(2))
    for p, raw in zip(results, model.logits):
        for name, idx in CLASSES.items():
            assert p.probabilities[name] == pytest.approx(torch.sigmoid(raw[idx]).item())


@pytest.mark.parametrize(
    ("n_frames", "window", "hop", "starts", "ends"),
    [
        (8, 4, 4, [0, 4], [4, 8]),  # exact coverage
        (8, 4, 3, [0, 3, 6], [4, 7, 8]),  # partial last window
        (2, 4, 3, [0], [2]),  # shorter than a window
        (11, 4, 3, [0, 3, 6, 9], [4, 7, 10, 11]),
    ],
)
def test_window_coverage_and_fixed_padding(audio, n_frames, window, hop, starts, ends):
    audio["sample.wav"] = torch.arange(n_frames, dtype=torch.float32)[None, :]
    dataset = datasets.StridedAudioDataset(
        "sample.wav", sequence_len=window, hop=hop, spec_config=SPEC, norm_config=NORM,
    )
    assert len(dataset) == len(starts)
    assert len(set(tuple(dataset[i].shape) for i in range(len(dataset)))) == 1
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    model = FakeModel([[1, 2, 3]] * len(starts))
    results = _predict(model, window_samples=window, hop_samples=hop, batch_size=2)
    assert [p.start_sample for p in results] == starts
    assert [p.end_sample for p in results] == ends
    assert results[-1].end_seconds == pytest.approx(n_frames / SPEC.sample_rate)
    assert sum(len(chunk) for chunk in model.features) == len(starts)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_samples": 0}, "window_samples"),
        ({"window_samples": -1}, "window_samples"),
        ({"window_samples": True}, "window_samples"),
        ({"hop_samples": 0}, "hop_samples"),
        ({"hop_samples": 5}, "gapped"),
        ({"batch_size": 0}, "batch_size"),
        ({"class_to_index": {}}, "nonempty"),
        ({"class_to_index": {"a": 0, "b": 2}}, "contiguous"),
        ({"class_to_index": {"a": 0, "b": 0}}, "contiguous"),
        ({"class_to_index": {"a": False, "b": 1}}, "contiguous"),
        ({"class_to_index": {"": 0, "b": 1}}, "names"),
        ({"task_config": TaskConfig("binary")}, "exactly 2 classes"),
        ({"thresholds": 0.3}, "only supported for multilabel"),
        ({"spec_config": replace(SPEC, sample_rate=0)}, "sample_rate"),
    ],
)
def test_invalid_inputs_raise_without_predicting(audio, kwargs, message):
    audio["sample.wav"] = torch.ones((1, 4))
    model = FakeModel([[1, 2, 3]])
    with pytest.raises(ValueError, match=message):
        _predict(model, **kwargs)
    assert model.offset == 0


@pytest.mark.parametrize("thresholds", [0, 1, -0.2, float("nan"), float("inf"), [0.2, 0.3],
                                            [0.2, 0.3, 1.0], {"bird": 0.5, "frog": 0.5},
                                            "bad", [[0.5, 0.5, 0.5]]])
def test_invalid_multilabel_thresholds(audio, thresholds):
    audio["sample.wav"] = torch.ones((1, 4))
    model = FakeModel([[0, 0, 0]])
    with pytest.raises(ValueError, match="threshold"):
        _predict(model, task_config=TaskConfig("multilabel"), thresholds=thresholds)
    assert model.offset == 0


def test_multilabel_default_half_and_index_ordered_vector(audio):
    audio["sample.wav"] = torch.ones((1, 4))
    result = _predict(FakeModel([[0, 0, 0]]), task_config=TaskConfig("multilabel"))[0]
    assert result.predicted_labels == ["bird", "frog", "rain"]  # inclusive >= 0.5
    result = _predict(
        FakeModel([[0, 0, 0]]), task_config=TaskConfig("multilabel"),
        thresholds=[0.6, 0.4, 0.6],
    )[0]
    assert result.predicted_labels == ["frog"]


@pytest.mark.parametrize("bad_logits", [[[1, 2]], [[1, 2, 3, 4]], [[math.nan, 0, 0]],
                                             [[math.inf, 0, 0]]])
def test_incorrect_output_or_non_finite_rejected_and_mode_restored(audio, bad_logits):
    audio["sample.wav"] = torch.ones((1, 4))
    model = FakeModel(bad_logits)
    model.train(False)
    with pytest.raises(ValueError, match="logits|non-finite"):
        _predict(model)
    assert model.training is False


@pytest.mark.parametrize("waveform, message", [
    (torch.empty((1, 0)), "nonempty mono"),
    (torch.ones((2, 4)), "nonempty mono"),
    (torch.tensor([[float("nan"), 1.0]]), "non-finite"),
    (OSError("unsupported audio format"), "Unable to read audio file"),
])
def test_empty_bad_or_unreadable_audio_is_explicit(audio, waveform, message):
    audio["sample.wav"] = waveform
    with pytest.raises(ValueError, match=message):
        _predict(FakeModel([[0, 0, 0]]))


def test_checkpoint_legacy_settings_missing_and_bundle_reconstruction_mock(monkeypatch, audio):
    audio["sample.wav"] = torch.ones((1, 4))
    bundle = {
        "spectrogramConfig": vars(SPEC), "classes": CLASSES,
        "encoderConfig": {}, "classifierConfig": {},
        "encoderState": {}, "classifierState": {},
    }
    with pytest.raises(ValueError, match="task_config, norm_config, window_samples, hop_samples"):
        predict_windows(bundle, "sample.wav")
    with pytest.raises(ValueError, match="norm_config"):
        predict_windows(bundle, "sample.wav", task_config=TaskConfig("multilabel"))

    model = FakeModel([[0, 0, 0]])
    calls = []

    def fake_restore(payload):
        calls.append(payload)
        return model

    monkeypatch.setattr(module, "_restore_model", fake_restore)
    result = predict_windows(
        bundle, "sample.wav", task_config=TaskConfig("multilabel"),
        norm_config=NORM, window_samples=4, hop_samples=4,
    )
    assert calls == [bundle]
    assert result[0].predicted_labels == ["bird", "frog", "rain"]


def test_v2_bundle_uses_resolved_inference_metadata_and_path(monkeypatch, audio):
    audio["sample.wav"] = torch.ones((1, 4))
    bundle = {
        "spectrogramConfig": vars(SPEC), "classes": CLASSES,
        "metadata": {
            "schema_version": 2, "task_type": "multilabel",
            "normalization_config": vars(NORM), "window_length_samples": 4,
            "hop_samples": 4, "thresholds": {"bird": 0.6, "frog": 0.4, "rain": 0.6},
        },
    }
    model = FakeModel([[0, 0, 0]])
    monkeypatch.setattr(module, "load_model", lambda path: bundle)
    monkeypatch.setattr(module, "_restore_model", lambda payload: model)
    results = predict_windows("bundle.pt", "sample.wav")
    assert results[0].predicted_labels == ["frog"]
    with pytest.raises(ValueError, match="Unsupported checkpoint metadata schema_version"):
        predict_windows({**bundle, "metadata": {"schema_version": 3}}, "sample.wav")
    with pytest.raises(ValueError, match="conflicts"):
        predict_windows(bundle, "sample.wav", class_to_index={"bird": 1, "frog": 0, "rain": 2})


def test_headless_import_does_not_require_optional_visualization_dependencies(monkeypatch):
    # The trainer can legitimately import optional summaries in other tests;
    # assert specifically that importing the predictor never requests them.
    original_import = builtins.__import__

    def reject_optional(name, *args, **kwargs):
        if name.startswith(("tensorboardX", "torchvision", "matplotlib")):
            raise AssertionError(f"predictor imported optional visualization: {name}")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", reject_optional)
        importlib.reload(module)


def test_checkpoint_reconstruction_uses_named_encoder_and_classifier(monkeypatch):
    encoder = nn.Linear(2, 2)
    classifier = nn.Linear(2, 3)
    payload = {
        "encoderConfig": {}, "classifierConfig": {},
        "encoderState": encoder.state_dict(),
        "classifierState": classifier.state_dict(),
    }
    monkeypatch.setattr(module, "ResidualEncoder", lambda config: nn.Linear(2, 2))
    monkeypatch.setattr(module, "Classifier", lambda config: nn.Linear(2, 3))
    reconstructed = module._restore_model(payload)
    assert list(reconstructed._modules) == ["encoder", "classifier"]
    for original, restored in zip((encoder, classifier), reconstructed.children()):
        for key, value in original.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)
    with pytest.raises(ValueError, match="missing required model field"):
        module._restore_model({"encoderConfig": {}, "classifierConfig": {}})


def test_checkpoint_partial_feature_metadata_is_not_filled_with_defaults(audio):
    audio["sample.wav"] = torch.ones((1, 4))
    payload = {
        "classes": CLASSES,
        "spectrogramConfig": {"sample_rate": SPEC.sample_rate},
        "metadata": {
            "schema_version": 2, "task_type": "multilabel",
            "normalization_config": {"mode": "min_max"},
            "window_length_samples": 4, "hop_samples": 4,
        },
    }
    with pytest.raises(ValueError, match="spectrogramConfig is missing"):
        predict_windows(payload, "sample.wav")
    payload["spectrogramConfig"] = vars(SPEC)
    with pytest.raises(ValueError, match="normalization_config is missing"):
        predict_windows(payload, "sample.wav")
    payload["metadata"]["normalization_config"] = vars(NORM)
    payload["metadata"]["task_type"] = "auto"
    with pytest.raises(ValueError, match="must be resolved"):
        predict_windows(payload, "sample.wav")


def test_explicit_device_and_mixed_training_modes_are_preserved(audio):
    audio["sample.wav"] = torch.ones((1, 4))

    class DeviceSpy(FakeModel):
        def __init__(self):
            super().__init__([[0, 1, 2]])
            self.batchnorm = nn.BatchNorm1d(1)
            self.move_calls = []

        def to(self, device):
            self.move_calls.append(torch.device(device))
            return super().to(device)

    model = DeviceSpy()
    model.train()
    model.batchnorm.eval()
    _predict(model)  # Keep the current module device when no device is passed.
    assert not model.move_calls
    assert model.training and not model.batchnorm.training
    model.offset = 0
    _predict(model, device=torch.device("cpu"))
    assert model.move_calls == [torch.device("cpu")]
    assert model.training and not model.batchnorm.training
