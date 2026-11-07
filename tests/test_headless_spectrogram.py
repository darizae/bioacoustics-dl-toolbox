"""Small deterministic parity fixtures against BioCall Studio Review STFT/raster v1."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys

import numpy as np
import pytest

from bioacoustics_dl_toolbox.visualization.spectrogram import (
    calculate_stft_power,
    rasterize_spectrogram,
)


def test_stft_matches_independent_r_reference_and_opposite_phase_stereo():
    # R reference: w <- .5-.5*cos(2*pi*(0:7)/7); Mod(fft(x*w)[1:5])^2
    signal = np.array([0, 1, 0, -1, 0, 1, 0, -1], dtype=np.float64)
    expected = np.array(
        [
            0.022791599075738,
            1.08235420046213,
            3.0625,
            1.08235420046213,
            0.022791599075738,
        ]
    )
    mono = calculate_stft_power(
        signal[None, :], sample_rate_hz=8_000, window_samples=8, hop_samples=4, n_fft=8
    )
    inverted = calculate_stft_power(
        np.stack((signal, -signal)),
        sample_rate_hz=8_000,
        window_samples=8,
        hop_samples=4,
        n_fft=8,
    )
    np.testing.assert_allclose(mono.power[:, 0], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(inverted.power, mono.power)
    assert inverted.frame_centers_x2 == mono.frame_centers_x2 == (7,)
    assert mono.frequencies_hz.tolist() == [0, 1_000, 2_000, 3_000, 4_000]


def test_unpadded_tail_short_window_and_dc_policy():
    clip = np.arange(10, dtype=np.float64)[None, :]
    spectrum = calculate_stft_power(
        clip, sample_rate_hz=1_000, window_samples=4, hop_samples=4, n_fft=8, source_start_frame=100
    )
    assert spectrum.frame_centers_x2 == (203, 211)  # Drops two-sample tail.
    assert spectrum.power.shape == (5, 2)
    assert spectrum.window_function == "symmetric-hann"
    short = calculate_stft_power(
        np.array([[-1.0, 1.0]]),
        sample_rate_hz=1_000,
        window_samples=20,
        hop_samples=10,
        source_start_frame=100,
    )
    assert short.short_clip_window
    assert (short.window_samples, short.n_fft) == (2, 2)
    assert short.window_function == "short-clip-rectangular"
    assert short.frame_centers_x2 == (201,)
    assert np.any(short.power > 0)
    one = calculate_stft_power(
        np.array([[3.0]]),
        sample_rate_hz=8_000,
        window_samples=8,
        hop_samples=4,
        remove_dc=False,
    )
    assert one.window_function == "short-clip-rectangular"
    np.testing.assert_array_equal(one.power, np.array([[9.0]]))
    dc = calculate_stft_power(
        np.full((2, 8), 7.0), sample_rate_hz=8_000, window_samples=8, hop_samples=4
    )
    assert not np.any(dc.power)


# SHA-256 values generated from the BioCall Studio Review v1 functions,
# not from this toolbox implementation. RGB uses the v1 9-stop palette,
# pixel-center interpolation and nearest-even uint8 rounding. Relative-dB
# hashes use Studio's little-endian float32 raster export.
@pytest.mark.parametrize(
    ("rate", "kind", "palette", "band", "rgb_sha", "db_sha"),
    [
        (
            8000, "tone", "magma", (200, 3000),
            "c09108f2e44f497711d0a2a5d11a0bc9db568a33a9796b6d883bdcdbd239ef6a",
            "ac7e73f32d559c6a16efcfa4e2e4a08d7b815519e2751a36a74e81b3e715771c",
        ),
        (
            16000, "chirp", "viridis", (300, 4200),
            "7ffa8095a31e7e5037d449fca63494cee2c3472382b6b21b4c8a6ffb021c7b6b",
            "702f0567390b2b94ff79862ef3c6bcac1f298840405950c6a389a443c7e5595d",
        ),
        (
            48000, "silence", "magma", (0, 4200),
            "4f05a3c0831f10f62fc845217d8880b353e16f7391f4175269c51303d274b8e9",
            "03823a371c979053927f16f73ddb6ff94cffc5fda443f6c7b040d7127215e15f",
        ),
        (
            1000, "short", "viridis", (0, 500),
            "cd1ae95d96c2d73aa7bc061b30b48bc3ad7346e7c94b0bb046efef87de08cd70",
            "5063f8cc6f4d9925f6bd98815e46f5a395fa868a40d053fc170b8832a0a74bfa",
        ),
        (
            8000, "stereo", "viridis", (500, 3500),
            "6d82136c750beeb406ac716214243cc7c9b90b54cbaaceb86bb56a77c2880f09",
            "88f0bdda8525c51848e8934e234761633d8dc2ecb5e62d8384584f674b31dbf2",
        ),
    ],
)
def test_review_v1_pixel_and_db_parity(rate, kind, palette, band, rgb_sha, db_sha):
    n = 2 if kind == "short" else 29
    t = np.arange(n, dtype=np.float64) / rate
    if kind == "tone":
        audio = np.sin(2 * np.pi * 1000 * t)[None, :]
    elif kind == "chirp":
        audio = np.sin(2 * np.pi * (500 * t + 2500 * t * t))[None, :]
    elif kind == "silence":
        audio = np.zeros((1, n))
    elif kind == "short":
        audio = np.array([[-1.0, 1.0]])
    else:
        wave = np.sin(2 * np.pi * 1200 * t)
        audio = np.stack((wave, -wave))
    spectral = calculate_stft_power(
        audio,
        sample_rate_hz=rate,
        window_samples=8,
        hop_samples=3,
        n_fft=8,
        source_start_frame=101,
    )
    raster = rasterize_spectrogram(
        spectral,
        width=17,
        height=9,
        frequency_min_hz=band[0],
        frequency_max_hz=band[1],
        contrast_db=50,
        palette=palette,
    )
    assert raster.rgb.shape == (9, 17, 3)
    assert raster.rgb.dtype == np.uint8
    assert raster.relative_db.shape == (9, 17)
    assert raster.relative_db.dtype == np.float64
    assert np.isfinite(raster.relative_db).all()
    assert hashlib.sha256(raster.rgb_bytes).hexdigest() == rgb_sha
    assert hashlib.sha256(raster.relative_db_f32_bytes).hexdigest() == db_sha
    assert raster.frame_centers_x2[0] == 2 * 101 + spectral.window_samples - 1
    assert raster.pixel_edge_frame(0) == 101
    assert raster.pixel_edge_frame(17) == 101 + n
    assert raster.pixel_edge_time_seconds(17) == pytest.approx((101 + n) / rate)
    assert raster.pixel_edge_frequency_hz(0) == min(band[1], rate / 2)
    assert raster.pixel_edge_frequency_hz(9) == band[0]
    if kind == "silence":
        assert set(np.unique(raster.relative_db)) == {-50.0}
        assert set(map(tuple, raster.rgb.reshape(-1, 3))) == {(0, 0, 4)}


def test_frequency_crop_edges_nyquist_and_band_peak_reference():
    rate = 8_000
    t = np.arange(128) / rate
    audio = (
        0.1 * np.sin(2 * np.pi * 1_000 * t) + 1.0 * np.sin(2 * np.pi * 3_000 * t)
    )[None, :]
    spectrum = calculate_stft_power(
        audio, sample_rate_hz=rate, window_samples=32, hop_samples=16, source_start_frame=19
    )
    low = rasterize_spectrogram(
        spectrum, width=12, height=6, frequency_min_hz=750, frequency_max_hz=1250, palette="magma"
    )
    high = rasterize_spectrogram(
        spectrum, width=12, height=6, frequency_min_hz=2750, frequency_max_hz=3250, palette="viridis"
    )
    assert low.frequency_centers_hz == (750.0, 1000.0, 1250.0)
    assert low.frequency_edges_hz == (750.0, 875.0, 1125.0, 1250.0)
    assert low.relative_db.max() > -5  # Referenced to the low band's own peak.
    assert high.relative_db.max() > -5
    assert spectrum.power.max() > 10 * spectrum.power[3:6].max()
    assert low.pixel_edge_frequency_hz(3) == 1_000
    assert low.pixel_edge_frame(0) == 19
    assert low.pixel_edge_frame(12) == 147
    assert low.frame_centers_x2[0] == 2 * 19 + 31
    with pytest.raises(ValueError, match="pixel boundary"):
        low.pixel_edge_frame(13)
    with pytest.raises(ValueError, match="pixel boundary"):
        low.pixel_edge_frequency_hz(-1)
    clamped = rasterize_spectrogram(
        spectrum, width=4, height=4, frequency_max_hz=99_999
    )
    assert clamped.frequency_max_hz == 4_000
    assert clamped.frequency_edges_hz[-1] == 4_000


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate_hz": 0},
        {"window_samples": 0},
        {"hop_samples": True},
        {"n_fft": 3},
        {"source_start_frame": -1},
        {"remove_dc": 1},
    ],
)
def test_invalid_stft_parameters(kwargs):
    defaults = dict(sample_rate_hz=8000, window_samples=8, hop_samples=4)
    defaults.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        calculate_stft_power(np.ones((1, 8)), **defaults)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"height": True},
        {"contrast_db": 0},
        {"frequency_min_hz": -1},
        {"frequency_min_hz": 4000},
        {"frequency_max_hz": float("nan")},
        {"palette": "plasma"},
    ],
)
def test_invalid_raster_parameters(kwargs):
    spectrum = calculate_stft_power(
        np.ones((1, 8)), sample_rate_hz=8000, window_samples=8, hop_samples=4
    )
    defaults = dict(width=12, height=5)
    defaults.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        rasterize_spectrogram(spectrum, **defaults)


def test_headless_import_without_optional_visualization_dependencies():
    script = """
import builtins
import sys
real_import = builtins.__import__
before = set(sys.modules)
def import_without_extras(name, *args, **kwargs):
    if name.partition('.')[0] in {'matplotlib', 'PIL', 'torchvision', 'PySide6', 'PyQt6'}:
        raise ImportError('optional visualization extra blocked: ' + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = import_without_extras
from bioacoustics_dl_toolbox.visualization.spectrogram import calculate_stft_power, rasterize_spectrogram
import bioacoustics_dl_toolbox.visualization as viz
assert 'plot_spectrogram' in viz.__all__
assert callable(calculate_stft_power) and callable(rasterize_spectrogram)
assert not any(name.partition('.')[0] in {'matplotlib', 'PIL', 'torchvision'} for name in set(sys.modules) - before)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
