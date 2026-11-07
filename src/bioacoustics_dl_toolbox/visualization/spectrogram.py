"""NumPy-only spectrogram calculation and rasterization for audio inspection.

Independent of the PyTorch model-preprocessing feature contract. Waveforms use
(channels, frames) layout; raster rows run from high to low frequency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

PaletteName = Literal["magma", "viridis"]

# Fixed Review v1 RGB stops, without Matplotlib's version-dependent colormaps.
_PALETTE_STOPS = {
    "magma": np.asarray(
        (
            (0, 0, 4),
            (28, 16, 68),
            (79, 18, 123),
            (129, 37, 129),
            (181, 54, 122),
            (229, 80, 100),
            (251, 135, 97),
            (254, 194, 135),
            (252, 253, 191),
        ),
        dtype=np.float64,
    ),
    "viridis": np.asarray(
        (
            (68, 1, 84),
            (71, 44, 122),
            (59, 82, 139),
            (44, 113, 142),
            (33, 144, 141),
            (39, 173, 129),
            (92, 200, 99),
            (170, 220, 50),
            (253, 231, 37),
        ),
        dtype=np.float64,
    ),
}


def _positive_int(value: int, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


@dataclass(frozen=True, slots=True)
class SpectralPower:
    """One-sided, unscaled float64 power with exact absolute frame centers.

    Power has shape (frequency bins, windows); frame_centers_x2 is doubled
    to retain exact half-sample positions of even-length windows.
    """

    power: np.ndarray
    frequencies_hz: np.ndarray
    frame_centers_x2: tuple[int, ...]
    sample_rate_hz: int
    source_start_frame: int
    source_end_frame: int
    requested_window_samples: int
    window_samples: int
    hop_samples: int
    n_fft: int
    remove_dc: bool

    @property
    def short_clip_window(self) -> bool:
        return self.window_samples != self.requested_window_samples

    @property
    def window_function(self) -> str:
        return "symmetric-hann" if self.window_samples >= 3 else "short-clip-rectangular"


@dataclass(frozen=True, slots=True)
class SpectrogramRaster:
    """Data-only float64 relative-dB and uint8 RGB pixels with source axes."""

    rgb: np.ndarray
    relative_db: np.ndarray
    frame_centers_x2: tuple[int, ...]
    frequency_centers_hz: tuple[float, ...]
    frequency_edges_hz: tuple[float, ...]
    sample_rate_hz: int
    source_start_frame: int
    source_end_frame: int
    frequency_min_hz: float
    frequency_max_hz: float
    contrast_db: float
    palette: PaletteName

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def rgb_bytes(self) -> bytes:
        return self.rgb.tobytes(order="C")

    @property
    def relative_db_f32_bytes(self) -> bytes:
        """Review v1-compatible little-endian float32 raster bytes."""
        return self.relative_db.astype("<f4").tobytes(order="C")

    def pixel_edge_frame(self, x: int) -> float:
        """Absolute source frame at a vertical pixel edge, 0..width."""
        if type(x) is not int or not 0 <= x <= self.width:
            raise ValueError("x must be a pixel boundary within the raster")
        return (
            self.source_start_frame
            + x * (self.source_end_frame - self.source_start_frame) / self.width
        )

    def pixel_edge_time_seconds(self, x: int) -> float:
        return self.pixel_edge_frame(x) / self.sample_rate_hz

    def pixel_edge_frequency_hz(self, y: int) -> float:
        """Frequency of a horizontal pixel edge, 0 at the top."""
        if type(y) is not int or not 0 <= y <= self.height:
            raise ValueError("y must be a pixel boundary within the raster")
        return (
            self.frequency_max_hz
            - y * (self.frequency_max_hz - self.frequency_min_hz) / self.height
        )


def calculate_stft_power(
    samples: np.ndarray,
    *,
    sample_rate_hz: int,
    window_samples: int,
    hop_samples: int,
    n_fft: int | None = None,
    remove_dc: bool = True,
    source_start_frame: int = 0,
) -> SpectralPower:
    """Return channel-power-mean STFT with Review v1 framing semantics.

    Noncentered, full windows only; incomplete tail frames are omitted. A
    symmetric N-1 Hann is used except for windows shorter than three samples,
    where rectangular weights avoid zeroing the clip. The requested window
    is shortened to the available frames. FFT zero-padding is within windows,
    never at clip borders; omitted n_fft selects the next power of two.
    """

    _positive_int(sample_rate_hz, "sample_rate_hz")
    _positive_int(window_samples, "window_samples")
    _positive_int(hop_samples, "hop_samples")
    if type(source_start_frame) is not int or source_start_frame < 0:
        raise ValueError("source_start_frame must be a nonnegative integer")
    if not isinstance(remove_dc, bool):
        raise TypeError("remove_dc must be a bool")
    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[0] < 1 or audio.shape[1] < 1:
        raise ValueError("samples must have shape (channels, frames), both nonempty")
    if not np.isfinite(audio).all():
        raise ValueError("samples must be finite")
    effective_window = min(window_samples, audio.shape[1])
    if n_fft is None:
        n_fft = 1 << (effective_window - 1).bit_length()
    _positive_int(n_fft, "n_fft")
    if n_fft < effective_window or n_fft & (n_fft - 1):
        raise ValueError("n_fft must be a power of two covering the effective window")

    working = audio - audio.mean(axis=1, keepdims=True) if remove_dc else audio
    starts = np.arange(0, working.shape[1] - effective_window + 1, hop_samples, dtype=np.int64)
    if effective_window < 3:
        window = np.ones(effective_window, dtype=np.float64)
    else:
        positions = np.arange(effective_window, dtype=np.float64)
        window = 0.5 - 0.5 * np.cos(2 * np.pi * positions / (effective_window - 1))
    accumulated = np.zeros((n_fft // 2 + 1, starts.size), dtype=np.float64)
    for channel in working:
        frames = np.stack(
            tuple(channel[int(start) : int(start) + effective_window] for start in starts),
            axis=0,
        )
        transformed = np.fft.rfft(frames * window[np.newaxis, :], n=n_fft, axis=1)
        accumulated += (transformed.real**2 + transformed.imag**2).T
    accumulated /= working.shape[0]
    if not np.isfinite(accumulated).all():
        raise ValueError("STFT power overflowed; scale input samples")
    return SpectralPower(
        power=accumulated,
        frequencies_hz=np.arange(n_fft // 2 + 1, dtype=np.float64) * sample_rate_hz / n_fft,
        frame_centers_x2=tuple(
            int(2 * (source_start_frame + start) + effective_window - 1) for start in starts
        ),
        sample_rate_hz=sample_rate_hz,
        source_start_frame=source_start_frame,
        source_end_frame=source_start_frame + audio.shape[1],
        requested_window_samples=window_samples,
        window_samples=effective_window,
        hop_samples=hop_samples,
        n_fft=n_fft,
        remove_dc=remove_dc,
    )


def _frequency_edges(
    centers: np.ndarray, *, lower: float, upper: float, bin_width: float
) -> tuple[float, ...]:
    if centers.size <= 1:
        return (lower, upper)
    midpoints = (centers[:-1] + centers[1:]) / 2
    edges = np.concatenate(
        (
            np.asarray((max(lower, centers[0] - bin_width / 2),)),
            midpoints,
            np.asarray((min(upper, centers[-1] + bin_width / 2),)),
        )
    )
    edges[0] = lower
    edges[-1] = upper
    return tuple(float(value) for value in edges)


def rasterize_spectrogram(
    spectral: SpectralPower,
    *,
    width: int,
    height: int,
    frequency_min_hz: float = 0.0,
    frequency_max_hz: float | None = None,
    contrast_db: float = 50.0,
    palette: PaletteName = "magma",
) -> SpectrogramRaster:
    """Return deterministic Review v1 RGB and dB pixels.

    Interpolate in linear power to pixel centers; use the *cropped STFT bin*
    peak as the image's relative dB reference, prior to interpolation.
    Silence is finite at -contrast_db. Frequency high edge clamps to Nyquist.
    """

    if not isinstance(spectral, SpectralPower):
        raise TypeError("spectral must be SpectralPower")
    _positive_int(width, "width")
    _positive_int(height, "height")
    if not isinstance(frequency_min_hz, (int, float)) or not math.isfinite(frequency_min_hz):
        raise ValueError("frequency_min_hz must be finite")
    if frequency_max_hz is None:
        frequency_max_hz = spectral.sample_rate_hz / 2
    if not isinstance(frequency_max_hz, (int, float)) or not math.isfinite(frequency_max_hz):
        raise ValueError("frequency_max_hz must be finite")
    lower = float(frequency_min_hz)
    upper = min(float(frequency_max_hz), spectral.sample_rate_hz / 2)
    if lower < 0 or lower >= upper:
        raise ValueError("frequency limits must form a nonempty range within Nyquist")
    if (
        not isinstance(contrast_db, (int, float))
        or not math.isfinite(contrast_db)
        or contrast_db <= 0
    ):
        raise ValueError("contrast_db must be positive and finite")
    if not isinstance(palette, str) or palette not in _PALETTE_STOPS:
        raise ValueError("palette must be 'magma' or 'viridis'")

    frequencies = spectral.frequencies_hz
    power = spectral.power
    populated = (frequencies >= lower) & (frequencies <= upper)
    cropped_centers = frequencies[populated]
    edges = _frequency_edges(
        cropped_centers,
        lower=lower,
        upper=upper,
        bin_width=spectral.sample_rate_hz / spectral.n_fft,
    )
    target_frames = (
        (np.arange(width, dtype=np.float64) + 0.5)
        * (spectral.source_end_frame - spectral.source_start_frame)
        / width
    )
    frame_centers = np.asarray(spectral.frame_centers_x2, dtype=np.float64) / 2
    frame_centers -= spectral.source_start_frame
    time_resampled = np.empty((power.shape[0], width), dtype=np.float64)
    for frequency_index, row in enumerate(power):
        time_resampled[frequency_index] = np.interp(
            target_frames, frame_centers, row, left=row[0], right=row[-1]
        )
    target_frequencies = lower + (
        (np.arange(height, dtype=np.float64) + 0.5) * (upper - lower) / height
    )
    raster_power = np.empty((height, width), dtype=np.float64)
    for column in range(width):
        raster_power[:, column] = np.interp(
            target_frequencies, frequencies, time_resampled[:, column]
        )
    raster_power = np.flipud(raster_power)

    peak = float(np.max(power[populated])) if populated.any() else 0.0
    relative_db = np.full(raster_power.shape, -float(contrast_db), dtype=np.float64)
    if np.isfinite(peak) and peak > 0:
        positive = raster_power > 0
        relative_db[positive] = 10 * np.log10(raster_power[positive] / peak)
        relative_db = np.clip(relative_db, -float(contrast_db), 0.0)

    normalized = np.clip((relative_db + contrast_db) / contrast_db, 0, 1)
    stops = _PALETTE_STOPS[palette]
    stop_x = np.linspace(0, 1, stops.shape[0])
    rgb = np.empty((*relative_db.shape, 3), dtype=np.uint8)
    for channel in range(3):
        rgb[:, :, channel] = np.rint(
            np.interp(normalized.ravel(), stop_x, stops[:, channel]).reshape(normalized.shape)
        ).astype(np.uint8)
    return SpectrogramRaster(
        rgb=rgb,
        relative_db=relative_db,
        frame_centers_x2=spectral.frame_centers_x2,
        frequency_centers_hz=tuple(float(value) for value in cropped_centers),
        frequency_edges_hz=edges,
        sample_rate_hz=spectral.sample_rate_hz,
        source_start_frame=spectral.source_start_frame,
        source_end_frame=spectral.source_end_frame,
        frequency_min_hz=lower,
        frequency_max_hz=upper,
        contrast_db=float(contrast_db),
        palette=palette,
    )


__all__ = [
    "PaletteName",
    "SpectralPower",
    "SpectrogramRaster",
    "calculate_stft_power",
    "rasterize_spectrogram",
]
