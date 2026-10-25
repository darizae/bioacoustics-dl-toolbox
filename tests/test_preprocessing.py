"""Unit regression tests for SPEC-02 shared deterministic preprocessing."""

from dataclasses import replace

import pytest
import torch

from bioacoustics_dl_toolbox.audio import datasets, transforms as T
from bioacoustics_dl_toolbox.audio.preprocessing import (
    build_feature_preprocessor,
    spectrogram_frames_for_samples,
)
from bioacoustics_dl_toolbox.config import (
    AugmentationConfig,
    NormalizationConfig,
    SpectrogramConfig,
)


def _config(compression: str) -> SpectrogramConfig:
    return SpectrogramConfig(
        sample_rate=8000,
        preemphasis=0.97,
        n_fft=64,
        hop_length=16,
        n_freq_bins=48,
        f_min=100,
        f_max=3000,
        freq_compression=compression,
    )


def _waveform(samples: int) -> torch.Tensor:
    t = torch.arange(samples, dtype=torch.float32)
    return (0.5 * torch.sin(t * 0.23) + 0.2 * torch.cos(t * 0.83)).unsqueeze(0)


@pytest.fixture
def fake_audio(monkeypatch):
    # Never touch a real recording or spawn an asynchronous reader worker.
    monkeypatch.setattr(datasets, "AsyncFileReader", lambda: object())
    audio_by_path: dict[str, torch.Tensor] = {}

    def load(path, sample_rate, mono=True):
        assert sample_rate == 8000 and mono
        return audio_by_path[path].clone()

    monkeypatch.setattr(datasets.audio_io, "load_audio_file", load)
    return audio_by_path


@pytest.mark.parametrize("compression", ["linear", "mel", "mfcc"])
@pytest.mark.parametrize("mode", ["db", "min_max"])
@pytest.mark.parametrize("samples", [20, 192])
def test_train_predict_features_match_for_same_audio_and_config(
    fake_audio, compression, mode, samples
):
    spec = _config(compression)
    norm = NormalizationConfig(mode=mode)
    raw = _waveform(samples)
    fake_audio["target-test.wav"] = raw
    window_samples = 192
    frame_count = spectrogram_frames_for_samples(window_samples, spec)
    train = datasets.SpectrogramDataset(
        ["target-test.wav"],
        classes=["target", "noise"],
        spec_config=spec,
        norm_config=norm,
        sequence_length=frame_count,
        aug_config=AugmentationConfig(enabled=False),
    )
    predict = datasets.StridedAudioDataset(
        "target-test.wav",
        sequence_len=window_samples,
        hop=window_samples,
        spec_config=spec,
        norm_config=norm,
    )
    train_tensor, labels = train[0]
    inference_tensor = predict[0]
    n_features = 32 if compression == "mfcc" else spec.n_freq_bins
    assert train_tensor.shape == inference_tensor.shape == (1, frame_count, n_features)
    assert train_tensor.dtype == inference_tensor.dtype == torch.float32
    assert labels["call"] == 1
    assert torch.isfinite(train_tensor).all()
    torch.testing.assert_close(train_tensor, inference_tensor, rtol=0, atol=0)


@pytest.mark.parametrize("compression", ["linear", "mel", "mfcc"])
@pytest.mark.parametrize("samples", [20, 96])
def test_short_window_is_post_feature_center_zero_padded(fake_audio, compression, samples):
    spec = _config(compression)
    fake_audio["short.wav"] = _waveform(samples)
    preprocessor = build_feature_preprocessor(spec, NormalizationConfig())
    native = preprocessor(_waveform(samples))
    # The ANIMAL-SPOT-derived transform repeats/center-trims to n_fft for
    # less-than-FFT signals. Larger incomplete windows have several frames.
    assert native.shape[1] == spectrogram_frames_for_samples(samples, spec)
    inference = datasets.StridedAudioDataset(
        "short.wav", sequence_len=192, hop=192, spec_config=spec
    )[0]
    assert inference.shape[1] == 9
    left = (9 - native.shape[1]) // 2
    assert torch.count_nonzero(inference[:, :left, :]) == 0
    assert torch.count_nonzero(inference[:, left + native.shape[1]:, :]) == 0
    torch.testing.assert_close(
        inference[:, left:left + native.shape[1], :], native, rtol=0, atol=0
    )


def test_empty_audio_uses_zero_fft_frame_and_zero_feature_padding():
    prep = build_feature_preprocessor(_config("linear"), NormalizationConfig())
    native = prep(torch.empty((1, 0), dtype=torch.float32))
    assert native.shape == (1, 1, 48)
    assert torch.isfinite(native).all()


@pytest.mark.parametrize("compression", ["linear", "mel", "mfcc"])
def test_two_runs_are_identical_and_inference_never_uses_random_augmentation(
    monkeypatch, fake_audio, compression
):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("random augmentation ran during prediction")

    for transform in (
        T.RandomAmplitude, T.RandomTimeStretch, T.RandomPitchShift, T.RandomAddNoise
    ):
        monkeypatch.setattr(transform, "__call__", unexpected)
    fake_audio["test.wav"] = _waveform(192)
    predictor = datasets.StridedAudioDataset(
        "test.wav", sequence_len=192, hop=128, spec_config=_config(compression)
    )
    first, second = predictor[0], predictor[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    prep = build_feature_preprocessor(_config(compression), NormalizationConfig())
    torch.testing.assert_close(prep(_waveform(192)), prep(_waveform(192)), rtol=0, atol=0)


def test_mfcc_never_instantiates_or_calls_amp2db(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Amp2Db must not run on MFCC features")

    monkeypatch.setattr(T, "Amp2Db", forbidden)
    prep = build_feature_preprocessor(_config("mfcc"), NormalizationConfig())
    raw_mfcc = prep.representation(
        prep.frequency_compression(prep.waveform_to_power(_waveform(192)))
    )
    assert raw_mfcc.shape[-1] == 32
    assert torch.isfinite(raw_mfcc).all()
    assert (raw_mfcc < 0).any() and (raw_mfcc > 0).any()
    assert torch.isfinite(prep(_waveform(192))).all()
    # Zero mel band does not produce -inf or NaN, matching the legacy MFCC convention.
    assert torch.isfinite(prep.representation(torch.zeros((1, 2, 48)))).all()


@pytest.mark.parametrize("compression", ["linear", "mel"])
def test_linear_and_mel_still_apply_amp2db(monkeypatch, compression):
    calls = []
    original = T.Amp2Db

    class TracedAmp2Db(original):
        def __call__(self, spec):
            calls.append(True)
            return super().__call__(spec)

    monkeypatch.setattr(T, "Amp2Db", TracedAmp2Db)
    prep = build_feature_preprocessor(_config(compression), NormalizationConfig())
    assert torch.isfinite(prep(_waveform(192))).all()
    assert calls == [True]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_rate", 0, "sample_rate"),
        ("n_fft", 0, "n_fft"),
        ("n_fft", 1, "n_fft"),
        ("hop_length", 0, "hop_length"),
        ("n_freq_bins", 0, "n_freq_bins"),
        ("freq_compression", "octave", "freq_compression"),
        ("f_min", -1, "Frequency range"),
        ("f_max", 5000, "Frequency range"),
        ("f_min", 3000, "Frequency range"),
        ("min_level_db", 0, "min_level_db"),
    ],
)
def test_invalid_spectrogram_configs_rejected(field, value, message):
    spec = replace(_config("linear"), **{field: value})
    with pytest.raises(ValueError, match=message):
        build_feature_preprocessor(spec, NormalizationConfig())


def test_invalid_normalization_and_mfcc_width_rejected():
    with pytest.raises(ValueError, match="normalization mode"):
        build_feature_preprocessor(
            _config("linear"), NormalizationConfig(mode="unknown")
        )
    with pytest.raises(ValueError, match="normalization min_level_db"):
        build_feature_preprocessor(
            _config("linear"), NormalizationConfig(min_level_db=0)
        )
    with pytest.raises(ValueError, match="MFCC requires"):
        build_feature_preprocessor(
            replace(_config("mfcc"), n_freq_bins=32), NormalizationConfig()
        )


def test_invalid_window_sizes_are_rejected_before_loading_audio(fake_audio):
    spec = _config("linear")
    with pytest.raises(ValueError, match="sequence_len"):
        datasets.StridedAudioDataset("does-not-exist.wav", 0, 1, spec_config=spec)
    with pytest.raises(ValueError, match="hop"):
        datasets.StridedAudioDataset("does-not-exist.wav", 192, 0, spec_config=spec)
    with pytest.raises(ValueError, match="sequence_length"):
        datasets.SpectrogramDataset(
            ["target-test.wav"],
            classes=["target", "noise"],
            sequence_length=0,
            spec_config=spec
        )


def test_cached_power_spectrogram_version_is_unchanged():
    # CachedSpectrogram stores raw power before the representation stage:
    # there is no MFCC-vs-dB feature data in the cache to invalidate.
    assert T.CachedSpectrogram.version == 4
