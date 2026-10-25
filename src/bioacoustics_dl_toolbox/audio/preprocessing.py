"""Shared deterministic audio feature preparation.

The waveform is already loaded/resampled/mono before this pipeline is called.
Training may augment the *power spectrogram* before calling ``to_features``;
prediction never augments. All feature tensors are ``[channel, time, frequency]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import Callable

import torch

from bioacoustics_dl_toolbox.audio import transforms as T
from bioacoustics_dl_toolbox.config import NormalizationConfig, SpectrogramConfig


@dataclass(frozen=True)
class FeaturePreprocessor:
    """Reusable deterministic stages, separated to permit training augmentation.

    ``waveform_to_power`` applies pre-emphasis then a ``center=False`` power
    spectrogram. ``frequency_compression`` maps that power spectrogram to
    linear, mel, or mel-bank features. ``representation`` converts linear/mel
    power to dB, or converts mel power to 32 MFCCs *without* a second dB pass.
    ``normalization`` is always last. Both dataset types use these exact stages.
    """

    waveform_to_power: T.Compose
    frequency_compression: Callable[[torch.Tensor], torch.Tensor]
    representation: Callable[[torch.Tensor], torch.Tensor]
    normalization: Callable[[torch.Tensor], torch.Tensor]

    def to_features(self, compressed: torch.Tensor) -> torch.Tensor:
        """Convert compressed, optionally augmented power to model features."""
        return self.normalization(self.representation(compressed))

    def from_spectrogram(self, power: torch.Tensor) -> torch.Tensor:
        """Convert a power spectrogram into normalized model features."""
        return self.to_features(self.frequency_compression(power))

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert a mono waveform ``[1, samples]`` into model features."""
        return self.from_spectrogram(self.waveform_to_power(waveform))


def _positive_int(value: object, field: str, minimum: int = 1) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}; got {value!r}")


def build_feature_preprocessor(
    spec_config: SpectrogramConfig, norm_config: NormalizationConfig
) -> FeaturePreprocessor:
    """Validate configs and build the shared, augmentation-free feature stages.

    MFCC keeps 32 coefficients (DCT columns 1..32), independent of the mel
    filterbank width; the default feature dimensions are unchanged.
    ``torchaudio`` is not required.
    """
    _positive_int(spec_config.sample_rate, "sample_rate")
    _positive_int(spec_config.n_fft, "n_fft", minimum=2)
    _positive_int(spec_config.hop_length, "hop_length")
    _positive_int(spec_config.n_freq_bins, "n_freq_bins")
    if spec_config.freq_compression not in {"linear", "mel", "mfcc"}:
        raise ValueError(
            f"Invalid freq_compression {spec_config.freq_compression!r}; "
            "expected 'linear', 'mel', or 'mfcc'"
        )
    if (
        not isinstance(spec_config.f_min, (int, float))
        or not isinstance(spec_config.f_max, (int, float))
        or not isfinite(spec_config.f_min)
        or not isfinite(spec_config.f_max)
        or not 0 <= spec_config.f_min < spec_config.f_max <= spec_config.sample_rate / 2
    ):
        raise ValueError("Frequency range must satisfy 0 <= f_min < f_max <= sample_rate / 2")
    if spec_config.freq_compression == "mfcc" and spec_config.n_freq_bins < 33:
        raise ValueError("MFCC requires n_freq_bins >= 33 to produce 32 coefficients")
    if spec_config.freq_compression == "linear":
        # Nearest-neighbor interpolation requires at least one FFT bin in range.
        # Keep the exclusive upper bound to avoid changing previously trained
        # features.
        min_bin = max(0, floor(spec_config.n_fft * spec_config.f_min / spec_config.sample_rate))
        max_bin = min(
            spec_config.n_fft - 1,
            ceil(spec_config.n_fft * spec_config.f_max / spec_config.sample_rate),
        )
        if max_bin <= min_bin:
            raise ValueError("Frequency range selects no FFT bins for linear compression")
    if norm_config.mode not in {"db", "min_max"}:
        raise ValueError(
            f"Invalid normalization mode {norm_config.mode!r}; expected 'db' or 'min_max'"
        )
    if norm_config.mode == "db" and (
        not isfinite(norm_config.min_level_db) or norm_config.min_level_db >= 0
    ):
        raise ValueError("normalization min_level_db must be finite and negative")
    if spec_config.freq_compression != "mfcc" and (
        not isfinite(spec_config.min_level_db) or spec_config.min_level_db >= 0
    ):
        raise ValueError("spectrogram min_level_db must be finite and negative")

    waveform_to_power = T.Compose(
        T.PreEmphasize(spec_config.preemphasis),
        T.Spectrogram(spec_config.n_fft, spec_config.hop_length, center=False),
    )
    if spec_config.freq_compression == "linear":
        frequency_compression: Callable[[torch.Tensor], torch.Tensor] = T.Interpolate(
            spec_config.n_freq_bins,
            spec_config.sample_rate,
            spec_config.f_min,
            spec_config.f_max,
        )
    else:
        frequency_compression = T.F2M(
            sr=spec_config.sample_rate,
            n_mels=spec_config.n_freq_bins,
            f_min=spec_config.f_min,
            f_max=spec_config.f_max,
        )

    representation: Callable[[torch.Tensor], torch.Tensor]
    if spec_config.freq_compression == "mfcc":
        representation = T.M2MFCC(n_mfcc=32)
    else:
        representation = T.Amp2Db(min_level_db=spec_config.min_level_db)

    normalization: Callable[[torch.Tensor], torch.Tensor]
    if norm_config.mode == "min_max":
        normalization = T.MinMaxNormalize()
    else:
        normalization = T.Normalize(
            min_level_db=norm_config.min_level_db, ref_level_db=norm_config.ref_level_db
        )
    return FeaturePreprocessor(
        waveform_to_power, frequency_compression, representation, normalization
    )


def spectrogram_frames_for_samples(sample_count: int, spec_config: SpectrogramConfig) -> int:
    """Expected ``center=False`` STFT frames for a fixed-size audio window.

    Shorter-than-FFT signals are cyclically repeated and center-trimmed to
    ``n_fft`` samples by ``Spectrogram`` (empty signals become zero), yielding
    one frame. Shorter-than-window tails are *not* waveform-padded; their
    normalized feature frames are zero-padded symmetrically to this count.
    """
    _positive_int(sample_count, "sample_count")
    _positive_int(spec_config.n_fft, "n_fft", minimum=2)
    _positive_int(spec_config.hop_length, "hop_length")
    return 1 + (max(sample_count, spec_config.n_fft) - spec_config.n_fft) // spec_config.hop_length
