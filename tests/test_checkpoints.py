import pickle
import sys

from torch import nn

from bioacoustics_dl_toolbox.audio.transforms import PaddedSubsequenceSampler
from bioacoustics_dl_toolbox.config import (
    ClassifierConfig,
    EncoderConfig,
    SpectrogramConfig,
)
from bioacoustics_dl_toolbox.training.checkpoints import load_model, save_model


def test_checkpoint_round_trip_keeps_compatible_six_key_contract(tmp_path):
    encoder = nn.Linear(2, 2)
    classifier = nn.Linear(2, 2)
    model = nn.Sequential(encoder, classifier)
    path = tmp_path / "trained_model.pt"

    save_model(
        model,
        encoder,
        EncoderConfig(),
        classifier,
        ClassifierConfig(input_channels=2),
        SpectrogramConfig(sample_rate=48000),
        path,
        {"noise": 0, "target": 1},
    )

    loaded = load_model(path)
    assert set(loaded) == {
        "encoderConfig",
        "classifierConfig",
        "spectrogramConfig",
        "encoderState",
        "classifierState",
        "classes",
    }
    assert loaded["classes"] == {"noise": 0, "target": 1}
    assert loaded["spectrogramConfig"]["sample_rate"] == 48000


def test_top_level_import_does_not_require_optional_rf_stack():
    import bioacoustics_dl_toolbox

    assert bioacoustics_dl_toolbox.SpectrogramConfig is SpectrogramConfig
    assert "librosa" not in sys.modules


def test_headless_trainer_import_does_not_load_visualization_stack():
    import bioacoustics_dl_toolbox.training.trainer  # noqa: F401

    assert "torchvision" not in sys.modules


def test_sequence_sampler_is_spawn_picklable():
    sampler = pickle.loads(pickle.dumps(PaddedSubsequenceSampler(16, random=False)))
    assert sampler.sequence_length == 16
