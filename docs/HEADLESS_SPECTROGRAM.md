# Headless inspection spectrograms

The toolbox exposes a **NumPy-only** spectral calculation and RGB raster API
at `bioacoustics_dl_toolbox.visualization.spectrogram`. This is an audio
inspection renderer: **do not** substitute its output for the PyTorch
`audio.transforms.Spectrogram` or `FeaturePreprocessor` training/inference
feature contract. No optional `[visualization]` extras are required.

```python
import numpy as np

from bioacoustics_dl_toolbox.visualization.spectrogram import (
    calculate_stft_power,
    rasterize_spectrogram,
)

sample_rate_hz = 8_000
seconds = np.arange(800, dtype=np.float64) / sample_rate_hz
samples = np.sin(2 * np.pi * 1_000 * seconds)[None, :]  # (channels, frames)
power = calculate_stft_power(
    samples,
    sample_rate_hz=sample_rate_hz,
    window_samples=160,
    hop_samples=80,
    source_start_frame=4_000,
)
raster = rasterize_spectrogram(
    power,
    width=320,
    height=120,
    frequency_min_hz=0,
    frequency_max_hz=3_000,
    contrast_db=50,
    palette="magma",
)
assert raster.rgb.shape == (120, 320, 3)
assert raster.rgb.dtype == np.uint8
assert raster.pixel_edge_frame(0) == 4_000
assert raster.pixel_edge_frame(320) == 4_800
```

## Explicit scientific and coordinate semantics

* Input is a finite `float64`-convertible waveform with shape
  `(channels, frames)`, at a positive integer sample rate. Each channel has
  its displayed clip mean subtracted by default. Independent-channel powers
  are **averaged before** the frequency band and peak are selected, so
  opposite-phase stereo channels do not cancel.
* `window_samples` is a requested window duration in samples, shortened when
  the clip is too short. `hop_samples` is the frame step; a tail shorter than a
  full window is omitted. Frames do not extend beyond clip edges.
  `n_fft` defaults to the smallest power of two covering the effective
  window (or accepts a larger explicit power of two); padding occurs only
  inside an FFT window. Windows with at least three samples use symmetric
  Hann weights; one- and two-sample clips use rectangular weights.
* `SpectralPower.power` is unscaled, nonnegative one-sided FFT power of shape
  `(frequency_bins, windows)`. `frame_centers_x2` stores exact doubled
  source-frame centers; `frequencies_hz` includes DC and Nyquist bins.
* The raster samples power by linear interpolation to pixel centers, then
  computes `10*log10(pixel_power / cropped_STFT_band_peak)`, clipped
  to `[-contrast_db, 0]`. Silence is uniformly `-contrast_db`, never
  negative infinity. This is a **relative** intensity scale, not calibrated
  sound pressure or absolute loudness.
* `frequency_min_hz` defaults to DC (0 Hz); the upper frequency bound
  defaults to Nyquist and is clamped to Nyquist. `frequency_centers_hz`
  and `frequency_edges_hz` describe included STFT bins, even when a band
  has only one bin or no bin. Pixels are RGB uint8 with shape
  `(height, width, 3)`, with time increasing rightward and frequency
  increasing upward. `relative_db` uses float64, and
  `relative_db_f32_bytes` gives little-endian float32 for compatibility.
* The caller selects pixel dimensions; a thumbnail can use 320×120 and
  a detail view 1200×450. `pixel_edge_frame(x)`,
  `pixel_edge_time_seconds(x)`, and `pixel_edge_frequency_hz(y)`
  map pixel boundaries to source coordinates without any candidate
  metadata, Qt drawing, image encoding, or cache.
* The built-in fixed-stop `magma` and `viridis` palettes use linear
  interpolation and nearest-even uint8 rounding, independent of Matplotlib.

**Downstream Studio integration:** The representative v1 RGB/float32 fixture
bytes are identical to the existing Studio renderer, including silent, short,
stereo, tone and chirp fixtures. The existing Studio content-key and renderer
version remain Studio-owned. When Studio starts consuming this released API,
retain its cache/version checks; any future raster change requires an
explicit Review renderer/cache version migration.
